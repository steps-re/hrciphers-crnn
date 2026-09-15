"""CRNN + CTC line recognizer for HR-Ciphers (same family as the organizers' winning LSTM baseline).

Splits match bench.py: the 300-line val pool never trains. val_pool[:25] is the report set
(the same lines the Gemini runs scored, so results are paired); val_pool[25:] picks the checkpoint.

  python3 crnn.py train --task 1 [--height 64] [--epochs 200]
  python3 crnn.py predict --task 1          # writes runs/<task>/{val,test}/crnn[+tag].jsonl
"""
import argparse, collections, fcntl, json, math, os, random, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
import bench  # noqa: E402  (splits, paths, scoring)


def vocab(gold):
    toks = sorted({t for v in gold.values() for t in v.split()})
    return {t: i + 1 for i, t in enumerate(toks)}  # 0 = CTC blank


def small(path, h=128):
    """Grayscale copy pre-shrunk to height h (cached on disk), so loaders stop resizing 2,000px scans every epoch."""
    out = path.replace("/raw/", f"/h{h}/").rsplit(".", 1)[0] + ".png"
    if not os.path.exists(out):
        im = Image.open(path).convert("L")
        im = im.resize((max(8, round(im.width * h / im.height)), h), Image.LANCZOS)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = f"{out}.{os.getpid()}.tmp.png"; im.save(tmp); os.replace(tmp, out)
    return out


def load_img(path, height, max_w, augment):
    im = Image.open(small(path)).convert("L")
    if augment:
        if random.random() < 0.5:  # mild scale jitter on height, keeps glyph shapes
            s = random.uniform(0.85, 1.15)
            im = im.resize((max(8, int(im.width * random.uniform(0.9, 1.1))), max(8, int(im.height * s))))
        if random.random() < 0.3:
            im = im.rotate(random.uniform(-1.5, 1.5), resample=Image.BILINEAR, expand=False, fillcolor=255)
        if random.random() < 0.2:
            im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
    w = max(16, min(max_w, round(im.width * height / im.height)))
    im = im.resize((w, height), Image.BILINEAR)
    a = 1.0 - np.asarray(im, dtype=np.float32) / 255.0  # ink = high
    if augment:
        a = np.clip(a * random.uniform(0.8, 1.2) + np.random.normal(0, 0.02, a.shape), 0, 1)
    return torch.from_numpy(a)[None]


class Lines(Dataset):
    def __init__(self, items, height, max_w, augment):
        self.items, self.h, self.mw, self.aug = items, height, max_w, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        lid, path, target = self.items[i]
        return lid, load_img(path, self.h, self.mw, self.aug), target


def collate(batch):
    ids, imgs, tgts = zip(*batch)
    W = max(x.shape[-1] for x in imgs)
    x = torch.zeros(len(imgs), 1, imgs[0].shape[1], W)
    for k, im in enumerate(imgs):
        x[k, :, :, : im.shape[-1]] = im
    widths = torch.tensor([im.shape[-1] for im in imgs])
    return list(ids), x, widths, list(tgts)


class CRNN(nn.Module):
    def __init__(self, n_classes, height):
        super().__init__()
        def block(i, o, pool):
            return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.1, inplace=True)] + (
                [nn.MaxPool2d(pool)] if pool else [])
        self.cnn = nn.Sequential(*block(1, 32, (2, 2)), *block(32, 64, (2, 2)), *block(64, 128, None),
                                 *block(128, 128, (2, 1)), *block(128, 256, None), *block(256, 256, (2, 1)),
                                 nn.Dropout2d(0.2))
        feat_h = height // 16
        self.proj = nn.Linear(256 * feat_h, 256)
        self.rnn = nn.LSTM(256, 256, num_layers=3, bidirectional=True, dropout=0.3, batch_first=True)
        self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n_classes))

    def forward(self, x):
        f = self.cnn(x)  # B, C, H/16, W/4
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)
        f, _ = self.rnn(F.leaky_relu(self.proj(f), 0.1))
        return self.out(f)  # B, T, K


def greedy(logits, inv):
    best = logits.argmax(-1).cpu().numpy()
    out = []
    for seq in best:
        toks, prev = [], 0
        for s in seq:
            if s != prev and s != 0:
                toks.append(inv[s])
            prev = s
        out.append(" ".join(toks))
    return out


def setup(a):
    gold, test_ids, tr_img, te_img, _ = bench.load(a.task)
    _, val_pool, _ = bench.split(a.task, 24)
    val = set(val_pool)
    train = [(i, tr_img(i), gold[i]) for i in sorted(gold) if i not in val]
    report = [(i, tr_img(i), gold[i]) for i in val_pool[:25]]
    select = [(i, tr_img(i), gold[i]) for i in val_pool[25:]]
    test = [(i, te_img(i), "") for i in test_ids]
    return gold, train, report, select, test


def run_eval(model, loader, inv, dev):
    model.eval(); preds = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        for ids, x, _, _ in loader:
            for i, p in zip(ids, greedy(model(x.to(dev)).float(), inv)):
                preds[i] = p
    return preds


def ckpt_path(a):
    d = f"{bench.ROOT}/models"; os.makedirs(d, exist_ok=True)
    return f"{d}/crnn_task{a.task}{('+' + a.tag) if a.tag else ''}.pt"


def cmd_train(a):
    lock = open(ckpt_path(a) + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # one trainer per task, ever
    except BlockingIOError:
        sys.exit(f"another trainer holds the lock for task {a.task}")
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gold, train, report, select, _ = setup(a)
    if a.all_data:  # final model: the held-out lines train too; fixed schedule, last epoch kept
        train, select = train + report + select, []
    v = vocab(gold); inv = {i: t for t, i in v.items()}
    enc = lambda s: torch.tensor([v[t] for t in s.split()], dtype=torch.long)
    tl = DataLoader(Lines(train, a.height, a.max_w, True), batch_size=a.bs, shuffle=True, num_workers=6,
                    collate_fn=collate, drop_last=True, persistent_workers=True)
    sl = DataLoader(Lines(select, a.height, a.max_w, False), batch_size=32, num_workers=4, collate_fn=collate)
    model = CRNN(len(v) + 1, a.height).to(dev)
    if a.init:  # transfer: reuse every layer whose shape matches (the output layer never does)
        src = torch.load(a.init, map_location=dev, weights_only=True)["model"]
        own = model.state_dict()
        keep = {k: w for k, w in src.items() if k in own and own[k].shape == w.shape}
        model.load_state_dict(keep, strict=False)
        print(f"init from {a.init}: {len(keep)}/{len(own)} tensors reused", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    best, bad, t0 = 9.9, 0, time.time()
    print(f"task {a.task}: {len(train)} train / {len(select)} select / {len(report)} report lines, "
          f"{len(v)} symbols, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for _, x, widths, tgts in tl:
            x = x.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(x)
            lp = logits.float().log_softmax(-1).permute(1, 0, 2)  # T, B, K
            T = logits.shape[1]
            in_len = torch.clamp(torch.ceil(widths.float() / 4).long(), max=T)
            ys = [enc(t) for t in tgts]
            loss = ctc(lp, torch.cat(ys).to(dev), in_len, torch.tensor([len(y) for y in ys]))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()  # step order warning is harmless under AMP
            tot += loss.item()
        if a.all_data:
            print(f"ep {ep:3d} loss {tot/len(tl):.3f} {(time.time()-t0)/60:.1f}min", flush=True)
            continue
        if ep % a.eval_every == 0 or ep == a.epochs - 1:
            preds = run_eval(model, sl, inv, dev)
            s, c = bench.cer([(preds[i], g) for i, _, g in select])
            flag = ""
            if s < best:
                best, bad, flag = s, 0, " *"
                torch.save({"model": model.state_dict(), "vocab": v, "height": a.height, "max_w": a.max_w,
                            "epoch": ep, "select_cer": s}, ckpt_path(a))
            else:
                bad += 1
            print(f"ep {ep:3d} loss {tot/len(tl):.3f} select symCER {s:.4f} charCER {c:.4f} "
                  f"{(time.time()-t0)/60:.1f}min{flag}", flush=True)
            if bad >= a.patience:
                print("early stop", flush=True); break
    last = {"model": model.state_dict(), "vocab": v, "height": a.height, "max_w": a.max_w, "epoch": ep,
            "select_cer": -1.0}
    if a.all_data:
        torch.save(last, ckpt_path(a))
        print(f"all-data model saved at epoch {ep} -> {ckpt_path(a)}", flush=True)
        return
    torch.save(last, ckpt_path(a).replace(".pt", "_last.pt"))
    preds = run_eval(model, sl, inv, dev)
    print(f"last-epoch select symCER {bench.cer([(preds[i], g) for i, _, g in select])[0]:.4f}", flush=True)
    print(f"best select symCER {best:.4f} -> {ckpt_path(a)}", flush=True)


def cmd_predict(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path(a), map_location=dev, weights_only=True)
    v = ck["vocab"]; inv = {i: t for t, i in v.items()}
    model = CRNN(len(v) + 1, ck["height"]).to(dev); model.load_state_dict(ck["model"])
    _, _, report, select, test = setup(a)
    name = "local:crnn" + (f"+{a.tag}" if a.tag else "")
    for split_name, items in (("val", report), ("test", test)):
        dl = DataLoader(Lines(items, ck["height"], ck["max_w"], False), batch_size=32, num_workers=4, collate_fn=collate)
        preds = run_eval(model, dl, inv, dev)
        path = bench.out_path(a.task, split_name, name)
        with open(path, "w") as f:
            for i, _, g in items:
                row = {"id": i, "pred": preds[i], "raw": preds[i], "usage": {"in": 0, "cached": 0, "out": 0},
                       "cost": 0.0, "secs": 0.0, "k": 0}
                if g:
                    row["gold"] = g
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(split_name, len(items), "->", path)
    print(f"checkpoint epoch {ck['epoch']}, select symCER {ck['select_cer']:.4f}")


def align(ref, hyp):
    """Levenshtein alignment: list of (ref_index or None, hyp_token or None)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (ref[i-1] != hyp[j-1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            ops.append((i - 1, hyp[j-1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i-1][j] + 1:
            ops.append((i - 1, None)); i -= 1
        else:
            ops.append((None, hyp[j-1], i)); j -= 1
    return ops[::-1]


def rover(hyps):
    """Majority vote over decoded symbol strings (ROVER-style), robust to CTC spike-timing differences."""
    toks = [h.split() for h in hyps]
    n = len(toks)
    dist = lambda a, b: bench.lev(a, b)
    backbone = min(range(n), key=lambda k: sum(dist(toks[k], toks[o]) for o in range(n)))
    ref = toks[backbone]
    sub = [collections.Counter({t: 1}) for t in ref]            # votes per backbone slot
    ins = collections.defaultdict(collections.Counter)            # (gap index) -> inserted token votes
    for k in range(n):
        if k == backbone: continue
        seen_gap = collections.Counter()
        for op in align(ref, toks[k]):
            if op[0] is not None:
                sub[op[0]][op[1] if op[1] is not None else ""] += 1
            else:
                gap = op[2]; ins[(gap, seen_gap[gap])][op[1]] += 1; seen_gap[gap] += 1
    out = []
    for gap in range(len(ref) + 1):
        r = 0
        while (gap, r) in ins:
            tok, c = ins[(gap, r)].most_common(1)[0]
            if c * 2 > n: out.append(tok)
            r += 1
        if gap < len(ref):
            best = max(sub[gap].items(), key=lambda kv: (kv[1], kv[0] == ref[gap]))[0]
            if best: out.append(best)
    return " ".join(out)


def cmd_ensemble(a):
    """Decode each checkpoint separately, then ROVER-vote the symbol strings line by line."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, report, select, test = setup(a)
    splits = [("test", test)] if a.test_only else [("select", select), ("val", report), ("test", test)]
    member_preds = []
    for tag in a.members.split(","):
        ck = torch.load(ckpt_path(argparse.Namespace(task=a.task, tag=tag)), map_location=dev, weights_only=True)
        m = CRNN(len(ck["vocab"]) + 1, ck["height"]).to(dev); m.load_state_dict(ck["model"])
        inv = {i: t for t, i in ck["vocab"].items()}
        per = {}
        for split_name, items in splits:
            dl = DataLoader(Lines(items, ck["height"], ck["max_w"], False), batch_size=32, num_workers=4, collate_fn=collate)
            per[split_name] = run_eval(m, dl, inv, dev)
            if split_name != "test":
                print(f"  member {tag or '(base)'} {split_name} symCER "
                      f"{bench.cer([(per[split_name][i], g) for i, _, g in items])[0]:.4f}", flush=True)
        member_preds.append(per)
    for split_name, items in splits:
        preds = {i: rover([mp[split_name][i] for mp in member_preds]) for i, _, _ in items}
        if split_name != "test":
            print(f"{split_name} ROVER symCER {bench.cer([(preds[i], g) for i, _, g in items])[0]:.4f} "
                  f"({len(member_preds)} members)", flush=True)
        if split_name == "select":
            continue
        path = bench.out_path(a.task, split_name, f"local:crnn+{a.out_tag}")
        with open(path, "w") as f:
            for i, _, g in items:
                row = {"id": i, "pred": preds[i], "raw": preds[i], "usage": {"in": 0, "cached": 0, "out": 0},
                       "cost": 0.0, "secs": 0.0, "k": 0}
                if g:
                    row["gold"] = g
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(split_name, len(items), "->", path, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict", "ensemble"])
    ap.add_argument("--task", required=True, choices=list(bench.TASKS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--max_w", type=int, default=2400)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--init", default="", help="checkpoint to warm-start from (transfer from a larger cipher)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all_data", action="store_true", help="train on every training line, keep the last epoch")
    ap.add_argument("--members", default="", help="ensemble: comma-separated checkpoint tags")
    ap.add_argument("--out_tag", default="ens")
    ap.add_argument("--test_only", action="store_true")
    a = ap.parse_args()
    {"train": cmd_train, "predict": cmd_predict, "ensemble": cmd_ensemble}[a.cmd](a)

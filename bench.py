"""HR-Ciphers (ICDAR 2024) line transcription benchmark, few-shot VLMs on GfS Vertex + Azure.

Data (never in git, research-only, no redistribution): ~/data/hr-ciphers/raw
Each call = fixed prefix (rules + alphabet + k example line images with gold) + one target line.
The prefix is identical across calls so provider prompt caching applies.

  python3 bench.py run  --task 2B --split val --n 20 --model vertex:gemini-3.8-flash
  python3 bench.py run  --task 1  --split test       --model azure:gpt-5.6-sol
  python3 bench.py score --task 2B --split val --model vertex:gemini-3.8-flash
  python3 bench.py submit --task 1 --model azure:gpt-5.6-sol     # writes RRC JSON
Output: ~/data/hr-ciphers/runs/<task>/<split>/<model>.jsonl (one row per line, resumable).
"""
import argparse, base64, collections, io, json, os, random, subprocess, sys, threading, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

ROOT = os.path.expanduser("~/data/hr-ciphers")
TASKS = {  # task -> (train dir, test dir, description)
    "1":  ("HR-Ciphers_task1_train/task1", "HR-Ciphers_task1_test/task1",
           "Vatican Secret Archive diplomatic ciphers. Mostly digits, often carrying diacritics above or below; some lines contain plaintext words (usually Italian)."),
    "2A": ("HR-Ciphers_task2A_train/Borg", "HR-Ciphers_task2A_test/Borg",
           "The Borg cipher (17th c., Latin plaintext). Invented glyphs, Latin letters and astrological/alchemical-looking signs. Symbols may touch."),
    "2B": ("HR-Ciphers_task2B_train/Copiale", "HR-Ciphers_task2B_test/Copiale",
           "The Copiale cipher (18th c., German plaintext). Latin and Greek letters, many with accents, plus ideograms."),
    "3A": ("HR-Ciphers_task3A_train/BNF", "HR-Ciphers_task3A_test/BNF",
           "16th c. French letters in cipher (Bibliotheque nationale de France). Each invented glyph is LABELLED with a letter A-Z or a code like _2, NOM3. The label is NOT the glyph's appearance: learn the glyph->label mapping only from the examples."),
    "3B": ("HR-Ciphers_task3B_train/Ramanacoil", "HR-Ciphers_task3B_test/Ramanacoil",
           "The Ramanacoil manuscript (1674, Dutch plaintext). Invented glyphs named after astrological signs. Transcribe only the main line of the image; ignore fragments of the lines above and below."),
}
# Conservative $/1M tokens (input, cached input, output). Unverified for the newest models: deliberately high.
PRICES = {
    "gemini-3.8-flash": (1.50, 0.15, 7.50),
    "gemini-3.1-pro-preview": (2.50, 0.25, 15.0),
    "gpt-5.6-sol": (5.00, 0.50, 40.0),
    "gpt-5.6-terra": (1.25, 0.125, 10.0),
    "gpt-5.6-luna": (0.25, 0.025, 2.0),
    "gpt-5.4-mini": (0.40, 0.04, 3.2),
}
AZ_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")  # https://<resource>.openai.azure.com
AZ_API = "2025-04-01-preview"
VX_PROJECT = os.environ.get("VERTEX_PROJECT_ID", "")  # GCP project billed for Vertex calls


def load(task):
    tr, te, desc = TASKS[task]
    gold = json.load(open(f"{ROOT}/raw/{tr}/{os.path.basename(glob_json(tr))}"))
    listing = {d: {f.rsplit(".", 1)[0]: f"{ROOT}/raw/{d}/img/{f}" for f in os.listdir(f"{ROOT}/raw/{d}/img")}
               for d in (tr, te)}
    test_ids = sorted(listing[te])
    return gold, test_ids, (lambda i: listing[tr][i]), (lambda i: listing[te][i]), desc


def glob_json(d):
    return next(f for f in os.listdir(f"{ROOT}/raw/{d}") if f.endswith(".json"))


def split(task, k, seed=0):
    """Fixed, disjoint example set (greedy symbol coverage) and validation pool."""
    gold, _, _, _, _ = load(task)
    ids = sorted(gold)
    rnd = random.Random(seed); rnd.shuffle(ids)
    val_pool, rest = ids[:300], ids[300:]
    freq = collections.Counter(t for i in ids for t in gold[i].split())
    common = {t for t, c in freq.items() if c >= 5}
    ex, covered = [], set()
    cand = [i for i in rest if len(gold[i].split()) <= 90]
    while len(ex) < k and cand:
        best = max(cand, key=lambda i: len((set(gold[i].split()) & common) - covered))
        ex.append(best); covered |= set(gold[best].split()); cand.remove(best)
    alphabet = [t for t, c in freq.most_common() if c >= 5]
    return ex, val_pool, alphabet


def png_b64(path, max_w=1600):
    im = Image.open(path).convert("L")
    if im.width > max_w:
        im = im.resize((max_w, max(1, round(im.height * max_w / im.width))))
    buf = io.BytesIO(); im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def rules(task, alphabet):
    return (
        f"You transcribe single text-line images from historical cipher manuscripts into a fixed symbol notation.\n"
        f"Manuscript: {TASKS[task][2]}\n\n"
        "Notation rules, learned from the examples that follow:\n"
        "- Output one line: the symbols left to right, separated by single spaces.\n"
        "- Use ONLY symbol names from the alphabet below, spelled exactly (case-sensitive).\n"
        "- Diacritic suffixes attach to the base symbol exactly as in the examples (e.g. 5^.. or u__).\n"
        "- If the notation uses <SPACE> for gaps between symbol groups, emit it where the examples would.\n"
        "- Transcribe every symbol, including punctuation-like marks. Do not decrypt, translate or explain.\n"
        "- Output the transcription only, with no preamble.\n\n"
        f"Alphabet ({len(alphabet)} symbols, most frequent first):\n{' '.join(alphabet)}\n\n"
        "Worked examples (image, then its exact transcription):"
    )


# ---------- providers ----------
_tok = {"t": None, "ts": 0}
_lock = threading.Lock()


def vx_token():
    with _lock:
        if not _tok["t"] or time.time() - _tok["ts"] > 1800:
            cmd = ["gcloud", "auth", "print-access-token"]
            if os.environ.get("VERTEX_ACCOUNT"):
                cmd += ["--account", os.environ["VERTEX_ACCOUNT"]]
            _tok["t"] = subprocess.check_output(cmd, text=True).strip()
            _tok["ts"] = time.time()
        return _tok["t"]


def post(url, body, headers, timeout=300):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


_vx_cache = {}
_cache_lock = threading.Lock()


def vertex_cache(model, prefix_text, examples):
    """Explicit context cache holding the rules + example images, so each call pays only for the target line."""
    key = (model, prefix_text)
    with _cache_lock:  # held across creation so parallel workers share one cache
        if key in _vx_cache:
            return _vx_cache[key]
        return _create_cache(key, model, prefix_text, examples)


def _create_cache(key, model, prefix_text, examples):
    parts = [{"text": prefix_text}]
    for b64, txt in examples:
        parts += [{"inlineData": {"mimeType": "image/png", "data": b64}}, {"text": f"Transcription: {txt}"}]
    body = {"model": f"projects/{VX_PROJECT}/locations/global/publishers/google/models/{model}",
            "contents": [{"role": "user", "parts": parts}], "ttl": "7200s"}
    r = post(f"https://aiplatform.googleapis.com/v1/projects/{VX_PROJECT}/locations/global/cachedContents", body,
             {"Authorization": f"Bearer {vx_token()}"})
    print(f"  created cache {r['name']} ({r.get('usageMetadata', {}).get('totalTokenCount')} tokens)", flush=True)
    _vx_cache[key] = r["name"]
    return r["name"]


def call_vertex(model, prefix_text, examples, target_b64):
    target = [{"text": "Now transcribe this line:"}, {"inlineData": {"mimeType": "image/png", "data": target_b64}},
              {"text": "Transcription:"}]
    lvl = os.environ.get("BENCH_THINK", "low")  # Gemini 3 thinkingLevel: low|medium|high
    body = {"generationConfig": {"temperature": 0, "maxOutputTokens": 12000, "thinkingConfig": {"thinkingLevel": lvl}}}
    if os.environ.get("BENCH_NOCACHE"):
        parts = [{"text": prefix_text}]
        for b64, txt in examples:
            parts += [{"inlineData": {"mimeType": "image/png", "data": b64}}, {"text": f"Transcription: {txt}"}]
        body["contents"] = [{"role": "user", "parts": parts + target}]
    else:
        body["cachedContent"] = vertex_cache(model, prefix_text, examples)
        body["contents"] = [{"role": "user", "parts": target}]
    url = f"https://aiplatform.googleapis.com/v1/projects/{VX_PROJECT}/locations/global/publishers/google/models/{model}:generateContent"
    r = post(url, body, {"Authorization": f"Bearer {vx_token()}"})
    c = r["candidates"][0]
    if c.get("finishReason") not in ("STOP", None):
        raise RuntimeError(f"finishReason={c.get('finishReason')}")
    text = "".join(p.get("text", "") for p in c["content"]["parts"] if not p.get("thought"))
    u = r.get("usageMetadata", {})
    return text, {"in": u.get("promptTokenCount", 0), "cached": u.get("cachedContentTokenCount", 0),
                  "out": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0),
                  "traffic": u.get("trafficType")}


def call_azure(model, prefix_text, examples, target_b64):
    content = [{"type": "text", "text": prefix_text}]
    for b64, txt in examples:
        content += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": f"Transcription: {txt}"}]
    content += [{"type": "text", "text": "Now transcribe this line:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_b64}", "detail": "high"}},
                {"type": "text", "text": "Transcription:"}]
    body = {"messages": [{"role": "user", "content": content}], "max_completion_tokens": 8000,
            "reasoning_effort": os.environ.get("BENCH_EFFORT", "low")}
    url = f"{AZ_ENDPOINT}/openai/deployments/{model}/chat/completions?api-version={AZ_API}"
    r = post(url, body, {"api-key": os.environ["AZURE_OPENAI_KEY"]})
    ch = r["choices"][0]
    if ch.get("finish_reason") != "stop":
        raise RuntimeError(f"finish_reason={ch.get('finish_reason')}")
    u = r.get("usage", {})
    return ch["message"]["content"] or "", {
        "in": u.get("prompt_tokens", 0), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "out": u.get("completion_tokens", 0)}


def cost(model, u):
    pi, pc, po = PRICES.get(model, (5, 0.5, 40))
    return ((u["in"] - u["cached"]) * pi + u["cached"] * pc + u["out"] * po) / 1e6


def normalise(text):
    text = text.strip().splitlines()[-1] if text.strip() else ""
    text = text.removeprefix("Transcription:").strip().strip("`").strip()
    return " ".join(text.split())


# ---------- scoring ----------
def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(pairs):
    """(symbol-level CER, character-level CER) over (pred, gold) pairs."""
    e = n = ec = nc = 0
    for p, g in pairs:
        e += lev(p.split(), g.split()); n += len(g.split())
        ec += lev(p, g); nc += len(g)
    return e / max(n, 1), ec / max(nc, 1)


# ---------- commands ----------
def out_path(task, split_name, model, tag=""):
    d = f"{ROOT}/runs/{task}/{split_name}"; os.makedirs(d, exist_ok=True)
    return f"{d}/{model.replace(':', '_')}{('+' + tag) if tag else ''}.jsonl"


def opaque_map(gold):
    """Neutral codes for every symbol, so label spelling (e.g. French letters in BNF) gives no language prior."""
    freq = collections.Counter(t for v in gold.values() for t in v.split())
    fwd = {t: f"g{n:03d}" for n, (t, _) in enumerate(freq.most_common(), 1)}
    return fwd, {v: k for k, v in fwd.items()}


def cmd_run(a):
    gold, test_ids, tr_img, te_img, _ = load(a.task)
    ex_ids, val_pool, alphabet = split(a.task, a.k)
    if a.split == "val":
        targets = [(i, tr_img(i)) for i in val_pool[a.offset:a.offset + a.n]]
    else:
        targets = [(i, te_img(i)) for i in test_ids][a.offset:a.offset + (a.n or 10**9)]
    prov, model = a.model.split(":", 1)
    fn = {"vertex": call_vertex, "azure": call_azure}[prov]
    if prov == "azure" and not os.environ.get("AZURE_OPENAI_KEY"):
        sys.exit("set AZURE_OPENAI_KEY (and AZURE_OPENAI_ENDPOINT) for the azure provider")
    enc = dec = None
    if a.opaque:
        enc, dec = opaque_map(gold)
        alphabet = [enc[t] for t in alphabet]
    tx = (lambda g: " ".join(enc[t] for t in g.split())) if enc else (lambda g: g)
    examples = [(png_b64(tr_img(i)), tx(gold[i])) for i in ex_ids]
    prefix = rules(a.task, alphabet)
    if enc:
        prefix = prefix.replace(TASKS[a.task][2], TASKS[a.task][2] + " Symbols are written as neutral codes g001, g002, ... that say nothing about their shape or meaning.")
    path = out_path(a.task, a.split, a.model, a.tag)
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    todo = [t for t in targets if t[0] not in done]
    print(f"task {a.task} {a.split} {a.model}: {len(targets)} targets, {len(done)} already done, k={len(ex_ids)}", flush=True)
    spent = [0.0]; wlock = threading.Lock()

    def work(t):
        i, p = t
        for attempt in range(6):
            try:
                t0 = time.time()
                raw, u = fn(model, prefix, examples, png_b64(p))
                pred = normalise(raw)
                if dec:
                    pred = " ".join(dec.get(t, t) for t in pred.split())
                row = {"id": i, "pred": pred, "raw": raw, "usage": u, "cost": cost(model, u),
                       "secs": round(time.time() - t0, 1), "k": len(ex_ids)}
                if a.split == "val":
                    row["gold"] = gold[i]
                with wlock:
                    open(path, "a").write(json.dumps(row, ensure_ascii=False) + "\n")
                    spent[0] += row["cost"]
                return
            except urllib.error.HTTPError as e:
                msg = e.read().decode()[:300]
                wait = 20 * (attempt + 1) if e.code in (429, 500, 503) else None
                print(f"  {i} HTTP {e.code} {msg}", flush=True)
                if wait is None: return
                time.sleep(wait)
            except Exception as e:
                print(f"  {i} error {e}", flush=True); time.sleep(10)
        print(f"  {i} FAILED after retries", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        for n, _ in enumerate(pool.map(work, todo), 1):
            if n % 10 == 0:
                print(f"  {n}/{len(todo)} done, ${spent[0]:.2f} this session", flush=True)
            if spent[0] > a.budget:
                print(f"BUDGET STOP at ${spent[0]:.2f}", flush=True); os._exit(2)
    for name in _vx_cache.values():
        try:
            req = urllib.request.Request(f"https://aiplatform.googleapis.com/v1/{name}", method="DELETE",
                                         headers={"Authorization": f"Bearer {vx_token()}"})
            urllib.request.urlopen(req, timeout=60)
        except Exception as e:
            print("  cache delete failed", name, e)
    print(f"finished: ${spent[0]:.3f} this session", flush=True)


def cmd_score(a):
    rows = [json.loads(l) for l in open(out_path(a.task, a.split, a.model, a.tag))]
    gold, *_ = load(a.task)
    s, c = cer([(r["pred"], gold[r["id"]]) for r in rows])
    usd = sum(r["cost"] for r in rows)
    u = collections.Counter()
    for r in rows: u.update({k: v for k, v in r["usage"].items() if isinstance(v, int)})
    print(json.dumps({"task": a.task, "model": a.model + (f"+{a.tag}" if a.tag else ""), "n": len(rows), "symbol_cer": round(s, 4),
                      "char_cer": round(c, 4), "usd": round(usd, 3), "usd_per_line": round(usd / len(rows), 5),
                      "avg_in": u["in"] // len(rows), "avg_cached": u["cached"] // len(rows),
                      "avg_out": u["out"] // len(rows), "avg_secs": round(sum(r["secs"] for r in rows) / len(rows), 1)}))


def cmd_submit(a):
    _, test_ids, *_ = load(a.task)
    rows = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(out_path(a.task, "test", a.model, a.tag))}
    missing = [i for i in test_ids if i not in rows]
    if missing:
        sys.exit(f"{len(missing)} test lines missing, e.g. {missing[:5]}")
    d = f"{ROOT}/submissions"; os.makedirs(d, exist_ok=True)
    f = f"{d}/task{a.task}_{a.model.replace(':', '_')}.json"
    json.dump({i: rows[i] for i in test_ids}, open(f, "w"), ensure_ascii=False, indent=0)
    print(f, len(test_ids), "lines")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score", "submit"])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--opaque", action="store_true", help="relabel symbols as neutral codes in the prompt")
    ap.add_argument("--tag", default="", help="variant name appended to the output file")
    a = ap.parse_args()
    {"run": cmd_run, "score": cmd_score, "submit": cmd_submit}[a.cmd](a)

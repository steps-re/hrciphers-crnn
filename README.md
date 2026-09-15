# HR-Ciphers: a small CRNN that reads historical cipher manuscripts

Line-level transcription of enciphered manuscripts, built for the [ICDAR 2024 Competition on
Handwriting Recognition of Historical Ciphers](https://rrc.cvc.uab.es/?ch=27) (HR-Ciphers) run by
the [DECRYPT project](https://de-crypt.org/) and the Computer Vision Center, Barcelona.

Submitted 14 September 2026 as **CRNN-CTC, 3-seed vote**. Character error rate on the organizers'
held-back test sets, scored by their server:

| Task | This method | Best entry before it |
|---|---|---|
| 1 Vatican digit ciphers | **0.0504** | 0.0783 |
| 2A Borg | **0.0613** | 0.0676 |
| 2B Copiale | **0.0162** | 0.0162 |
| 3A BNF French letters | **0.0081** | 0.0089 |
| 3B Ramanacoil | **0.0388** | 0.0561 |

Lower is better. Live tables: [task 1](https://rrc.cvc.uab.es/?ch=27&com=evaluation&task=1) ·
[2A](https://rrc.cvc.uab.es/?ch=27&com=evaluation&task=2) ·
[2B](https://rrc.cvc.uab.es/?ch=27&com=evaluation&task=3) ·
[3A](https://rrc.cvc.uab.es/?ch=27&com=evaluation&task=4) ·
[3B](https://rrc.cvc.uab.es/?ch=27&com=evaluation&task=5).

Total cloud spend for the whole project, including the dead ends, was about $17. Each model trains
in 10 to 30 minutes on one NVIDIA L4.

## What is here

- `crnn.py` - the recognizer: CNN + 3-layer BiLSTM + CTC, 5.6-6.9M parameters, trained per cipher.
  Also holds the ROVER vote and the warm-start transfer.
- `bench.py` - splits, symbol- and character-level CER, and a few-shot vision-language-model
  baseline (Gemini on Vertex, GPT on Azure) for comparison.

No data and no weights are included: see [Data](#data).

## Method, in short

1. **Train one model per cipher** on the competition's own training lines. Images are grayscale,
   scaled to a fixed height (64px, or 96px for the taller Copiale and Ramanacoil lines), with mild
   scale, rotation, blur and brightness jitter.
2. **Warm-start the low-data ciphers.** BNF has 788 training lines. Starting from the model trained
   on the 5,496-line Vatican task and re-learning only the output layer cut its error roughly in
   half (2.5% to 1.3% on held-out lines). Ramanacoil starts from Copiale.
3. **Train three seeds on every training line** for the final models, for a fixed 60 epochs, keeping
   the last epoch. In validation runs the last epoch scored within 0.05 points of the best epoch, so
   nothing is lost by dropping the checkpoint-selection split.
4. **Vote.** Decode each model separately, then take a majority vote per symbol using ROVER-style
   alignment against the median hypothesis. This beat the best single model on all five tasks.

### Do not average CTC posteriors across separately trained models

Averaging per-frame log-probabilities of three independently trained models took Borg from about 6%
CER to **68%**. CTC models put their probability spikes at slightly different frames, so one model's
symbol lands against another's blank and the blank wins. Voting on decoded strings has no such
problem. The same code path with one member reproduces that member exactly, which is the check worth
running before trusting any ensemble number.

### A frontier model with worked examples was not competitive

Before training anything, the pipeline in `bench.py` gave Gemini 3.1 Pro 24 example line images with
their gold transcriptions in a cached context, then asked it to read a new line. On the same 25
held-out lines per cipher it made 3 to 12 times more errors than the trained CRNN, at roughly $0.011
per line against zero marginal cost:

| Task | Few-shot VLM | Trained CRNN (single model) |
|---|---|---|
| 1 Vatican | 8.9% | 2.5% |
| 2A Borg | 19.6% | 5.9% |
| 2B Copiale | 13.7% | 1.1% |
| 3A BNF | 6.5% | 1.3% |
| 3B Ramanacoil | 25.7% | 4.2% |

One quirk worth knowing: BNF's symbol labels *are* the plaintext letters they stand for, so the model
wrote plausible French instead of reading glyphs. Replacing every label with a neutral code
(`--opaque`) took it from 26.7% to 6.5%. The same trick makes the other ciphers much worse, because
their label names describe the shapes.

## Data

Get the line images and transcriptions from the
[competition downloads page](https://rrc.cvc.uab.es/?ch=27&com=downloads) (free account required)
and accept the organizers' terms. Their images are not redistributable, so nothing here ships data
or trained weights. Unpack to `~/data/hr-ciphers/raw/HR-Ciphers_task*_{train,test}/`.

**Submission gotcha:** the task 1 test set contains 22 lines with no cipher at all, just plaintext
Italian. The evaluation server rejects a submission that includes them.

## Use

```bash
pip install torch torchvision pillow numpy

# train one cipher, then predict its test set
python3 crnn.py train   --task 1  --height 64 --epochs 60 --patience 40
python3 crnn.py predict --task 1

# the submitted recipe: three seeds on all data, then vote
for s in 0 1 2; do
  python3 crnn.py train --task 2A --height 64 --epochs 60 --seed $s --tag all$s --all_data \
    --init ~/data/hr-ciphers/models/crnn_task1.pt        # warm start, low-data tasks only
done
python3 crnn.py ensemble --task 2A --members all0,all1,all2 --out_tag vote --test_only

# few-shot VLM baseline (optional; needs your own cloud credentials)
VERTEX_PROJECT_ID=<project> python3 bench.py run --task 2B --n 25 --model vertex:gemini-3.1-pro-preview
python3 bench.py score --task 2B --model vertex:gemini-3.1-pro-preview
```

Heights used: 64 for tasks 1, 2A and 3A; 96 for 2B and 3B. Warm starts: 2A and 3A from task 1, 3B
from 2B.

## Citing the benchmark

Megyesi et al., *Decryption of historical manuscripts: the DECRYPT project*, Cryptologia, 2020.
<https://doi.org/10.1080/01611194.2020.1716410>. The competition report is published in the ICDAR
2024 proceedings.

## License

MIT, see [LICENSE](LICENSE). The competition data is licensed separately by its holders.

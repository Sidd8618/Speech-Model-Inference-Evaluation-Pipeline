# Speech Model Inference & Evaluation Pipeline

A reproducible pipeline that downloads a pretrained speech recognition model from
Hugging Face, runs inference on a public speech dataset, and evaluates the results
with standard ASR metrics (WER, CER, latency).

Built for an internship selection assignment. See `research/wav2vec2_summary.md` for
the paper-summary component (Part 1).

## What it does

1. Downloads a pretrained model + processor/tokenizer from Hugging Face
   (default: `facebook/wav2vec2-base-960h`; also supports `facebook/hubert-base-ls960`
   and `openai/whisper-small`).
2. Loads a public speech dataset from Hugging Face
   (default: `hf-internal-testing/librispeech_asr_dummy`, a small public slice of
   LibriSpeech — no auth token required).
3. Runs inference on N audio samples (default 30, configurable 20-50+).
4. Saves predictions to `results/predictions.csv`.
5. Computes WER, CER, average latency, and sample counts.
6. Writes `results/metrics.json` and a human-readable `results/report.md`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires internet access to `huggingface.co` to download the model and dataset on
first run (both are cached locally afterward under `~/.cache/huggingface`).

## Run

```bash
python run.py
```

That's it — this downloads everything, runs inference on 30 samples, and populates
`results/`.

### Optional arguments

```bash
python run.py \
  --model facebook/wav2vec2-base-960h \
  --num_samples 40 \
  --dataset hf-internal-testing/librispeech_asr_dummy \
  --dataset_config clean \
  --split validation \
  --output_dir results
```

| Flag | Default | Notes |
|---|---|---|
| `--model` | `facebook/wav2vec2-base-960h` | one of the three assignment-approved models |
| `--num_samples` | `30` | number of audio clips to run inference on |
| `--dataset` | `hf-internal-testing/librispeech_asr_dummy` | any HF dataset with an `audio` column + text column |
| `--dataset_config` | `clean` | dataset config/subset name |
| `--split` | `validation` | dataset split |
| `--output_dir` | `results` | where predictions/metrics/report are written |

To evaluate a different model, just swap `--model`:

```bash
python run.py --model facebook/hubert-base-ls960 --num_samples 30
python run.py --model openai/whisper-small --num_samples 30
```

## Output

**`results/predictions.csv`**

```
audio_id,ground_truth,prediction,latency_sec
1272-141231-0000,MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES,MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES,0.0421
...
```

**`results/metrics.json`**

```json
{
  "model": "facebook/wav2vec2-base-960h",
  "model_type": "ctc",
  "dataset": "hf-internal-testing/librispeech_asr_dummy",
  "dataset_config": "clean",
  "split": "validation",
  "num_samples": 30,
  "wer": 0.045,
  "cer": 0.018,
  "avg_inference_latency_sec": 0.0512,
  "total_inference_time_sec": 1.536,
  "device": "cpu"
}
```

**`results/report.md`** — a readable summary table plus a preview of predictions.

## Design notes

- **Dataset choice:** `hf-internal-testing/librispeech_asr_dummy` was chosen over full
  LibriSpeech or Common Voice because it's publicly accessible with no auth token,
  small enough to download in seconds, and is the same dataset used in Hugging Face's
  own official model documentation/examples for Wav2Vec2, HuBERT, and Whisper — making
  this pipeline easy to reproduce on any machine. Swap `--dataset` to point at full
  LibriSpeech (`librispeech_asr`) for a larger-scale run.
- **Fair WER/CER across model families:** CTC models (Wav2Vec2, HuBERT) output
  upper-case, unpunctuated text; Whisper outputs cased, punctuated text. Both
  references and hypotheses are normalized (upper-cased, punctuation stripped) before
  scoring so that metrics are comparable across all three model choices.
- **Latency measurement:** timed strictly around the model forward/generate call
  (excludes feature extraction and I/O), so numbers reflect model compute cost.

## Reproducibility

- All configuration is exposed as CLI flags with sane defaults — `python run.py` with
  zero arguments reproduces the full pipeline end to end.
- Model and dataset are pulled by fixed Hugging Face ids/revisions rather than local
  files, so results are reproducible on any machine with internet access and the
  packages in `requirements.txt`.
- No hidden state: each run overwrites `results/` from scratch.

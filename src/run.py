"""
run.py — single entry point for the whole assignment.

Usage:
    python run.py
    python run.py --model openai/whisper-small --num_samples 40

What it does, end to end:
  1. Downloads a pretrained speech model + processor from Hugging Face.
  2. Loads a public speech dataset (LibriSpeech slice) from Hugging Face.
  3. Runs inference on N audio samples.
  4. Writes results/predictions.csv
  5. Computes WER, CER, latency stats.
  6. Writes results/metrics.json and results/report.md
"""

import argparse
import csv
import json
import os

import torch

from src.data_loader import load_speech_dataset
from src.model_loader import load_model_and_processor, MODEL_TYPES
from src.inference import run_ctc_inference, run_whisper_inference
from src.evaluate import compute_wer, compute_cer
from src.utils import normalize_text, get_ground_truth, get_audio_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproducible ASR inference + evaluation pipeline"
    )
    parser.add_argument(
        "--model",
        default="facebook/wav2vec2-base-960h",
        choices=list(MODEL_TYPES.keys()),
        help="Hugging Face model id to evaluate.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Number of audio samples to run inference on (20-50 recommended).",
    )
    parser.add_argument(
        "--dataset",
        default="hf-internal-testing/librispeech_asr_dummy",
        help="Hugging Face dataset id. Must expose an 'audio' column + a text column.",
    )
    parser.add_argument("--dataset_config", default="clean")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output_dir", default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run.py] Using device: {device}")

    print(f"[run.py] Loading model + processor: {args.model}")
    model, processor, model_type = load_model_and_processor(args.model, device)

    print(
        f"[run.py] Loading dataset: {args.dataset} "
        f"({args.dataset_config}/{args.split}), {args.num_samples} samples requested"
    )
    dataset = load_speech_dataset(
        num_samples=args.num_samples,
        dataset_name=args.dataset,
        config=args.dataset_config,
        split=args.split,
    )
    print(f"[run.py] Loaded {len(dataset)} samples")

    rows = []
    latencies = []

    for i, example in enumerate(dataset):
        audio = example["audio"]
        audio_array = audio["array"]
        sampling_rate = audio["sampling_rate"]
        ground_truth = get_ground_truth(example)
        audio_id = get_audio_id(example, i)

        if model_type == "ctc":
            prediction, latency = run_ctc_inference(
                model, processor, audio_array, sampling_rate, device
            )
        else:
            prediction, latency = run_whisper_inference(
                model, processor, audio_array, sampling_rate, device
            )

        latencies.append(latency)
        rows.append(
            {
                "audio_id": audio_id,
                "ground_truth": ground_truth.strip(),
                "prediction": prediction.strip(),
                "latency_sec": round(latency, 4),
            }
        )
        print(
            f"[{i + 1}/{len(dataset)}] "
            f"GT: {ground_truth.strip()[:60]!r}  PRED: {prediction.strip()[:60]!r}"
        )

    # ---- Part 2 deliverable: predictions.csv ----
    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio_id", "ground_truth", "prediction", "latency_sec"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[run.py] Saved predictions -> {predictions_path}")

    # ---- Part 3 deliverable: metrics.json ----
    references_norm = [normalize_text(r["ground_truth"]) for r in rows]
    hypotheses_norm = [normalize_text(r["prediction"]) for r in rows]

    wer = compute_wer(references_norm, hypotheses_norm) if rows else float("nan")
    cer = compute_cer(references_norm, hypotheses_norm) if rows else float("nan")
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    metrics = {
        "model": args.model,
        "model_type": model_type,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "num_samples": len(rows),
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "avg_inference_latency_sec": round(avg_latency, 4),
        "total_inference_time_sec": round(sum(latencies), 4),
        "device": device,
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[run.py] Saved metrics -> {metrics_path}")

    # ---- Part 3 deliverable: report.md ----
    report_path = os.path.join(args.output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(generate_report(metrics, rows))
    print(f"[run.py] Saved report -> {report_path}")

    print("\n[run.py] Done. Summary:")
    print(json.dumps(metrics, indent=2))


def generate_report(metrics: dict, rows: list) -> str:
    lines = []
    lines.append("# Evaluation Report\n")
    lines.append(f"**Model:** `{metrics['model']}` ({metrics['model_type']})  ")
    lines.append(
        f"**Dataset:** `{metrics['dataset']}` "
        f"({metrics['dataset_config']}/{metrics['split']})  "
    )
    lines.append(f"**Device:** {metrics['device']}  ")
    lines.append(f"**Samples processed:** {metrics['num_samples']}\n")

    lines.append("## Summary Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Word Error Rate (WER) | {metrics['wer']:.2%} |")
    lines.append(f"| Character Error Rate (CER) | {metrics['cer']:.2%} |")
    lines.append(f"| Avg. inference latency (s) | {metrics['avg_inference_latency_sec']:.4f} |")
    lines.append(f"| Total inference time (s) | {metrics['total_inference_time_sec']:.4f} |")
    lines.append(f"| Samples processed | {metrics['num_samples']} |\n")

    lines.append("## Sample Predictions (first 10)\n")
    lines.append("| audio_id | ground_truth | prediction |")
    lines.append("|---|---|---|")
    for r in rows[:10]:
        gt = r["ground_truth"].replace("|", " ")
        pred = r["prediction"].replace("|", " ")
        lines.append(f"| {r['audio_id']} | {gt} | {pred} |")

    lines.append("\n## Notes\n")
    lines.append(
        "- WER/CER are computed after normalizing both ground truth and "
        "predictions (uppercase, punctuation stripped) so that CTC models "
        "(which output unpunctuated uppercase text) and Whisper (which "
        "outputs cased, punctuated text) can be compared fairly."
    )
    lines.append(
        "- Latency is measured around the model forward/generate call only, "
        "excluding feature extraction and I/O."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

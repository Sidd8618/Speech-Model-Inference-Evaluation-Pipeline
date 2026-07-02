"""
Runs a single forward pass / generation call for one audio sample and
times it. Latency is measured around only the model compute (not the
feature-extraction preprocessing), so numbers are comparable to
published inference-speed benchmarks.
"""

import time
import torch


@torch.no_grad()
def run_ctc_inference(model, processor, audio_array, sampling_rate, device):
    inputs = processor(
        audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True
    )
    input_values = inputs.input_values.to(device)

    start = time.perf_counter()
    logits = model(input_values).logits
    latency = time.perf_counter() - start

    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]
    return transcription, latency


@torch.no_grad()
def run_whisper_inference(model, processor, audio_array, sampling_rate, device):
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    start = time.perf_counter()
    predicted_ids = model.generate(input_features)
    latency = time.perf_counter() - start

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription, latency

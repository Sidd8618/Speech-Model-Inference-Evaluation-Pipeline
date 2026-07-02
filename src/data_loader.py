"""
Loads a public speech dataset from the Hugging Face Hub.

Default: hf-internal-testing/librispeech_asr_dummy
This is a small (~73 sample), publicly accessible, non-gated slice of
LibriSpeech that the transformers/datasets libraries use in their own
official Wav2Vec2/HuBERT/Whisper documentation examples. It is a good
fit for this assignment because:
  - No authentication token is required (unlike Common Voice).
  - It downloads in seconds instead of minutes/hours (unlike full
    LibriSpeech, which is multiple GB).
  - It ships with 16kHz audio + text transcripts already aligned.

You can point --dataset / --dataset_config / --split at any other
Hugging Face ASR dataset that exposes an "audio" column and a text
transcript column (e.g. "librispeech_asr", "mozilla-foundation/common_voice_11_0").
"""

from datasets import load_dataset, Audio


def load_speech_dataset(
    num_samples: int = 30,
    dataset_name: str = "hf-internal-testing/librispeech_asr_dummy",
    config: str = "clean",
    split: str = "validation",
):
    if config:
        ds = load_dataset(dataset_name, config, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    # Ensure audio is decoded at 16kHz, the sampling rate all three
    # candidate models (Wav2Vec2 / HuBERT / Whisper) expect.
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    n = min(num_samples, len(ds))
    ds = ds.select(range(n))
    return ds

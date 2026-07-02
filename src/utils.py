"""
Loads a pretrained speech model + its matching processor/tokenizer
from the Hugging Face Hub.

Two model families are supported, matching the assignment's
allowed choices:

  - "ctc"     : facebook/wav2vec2-base-960h, facebook/hubert-base-ls960
                (encoder-only, CTC greedy-decoded)
  - "whisper" : openai/whisper-small
                (encoder-decoder, autoregressive generation)
"""

import torch
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    HubertForCTC,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

MODEL_TYPES = {
    "facebook/wav2vec2-base-960h": "ctc",
    "facebook/hubert-base-ls960": "ctc",
    "openai/whisper-small": "whisper",
}


def load_model_and_processor(model_name: str, device: str):
    if model_name not in MODEL_TYPES:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose one of: {list(MODEL_TYPES)}"
        )
    model_type = MODEL_TYPES[model_name]

    if model_type == "ctc":
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        if "hubert" in model_name:
            model = HubertForCTC.from_pretrained(model_name)
        else:
            model = Wav2Vec2ForCTC.from_pretrained(model_name)
    else:  # whisper
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)

    model.to(device)
    model.eval()
    return model, processor, model_type

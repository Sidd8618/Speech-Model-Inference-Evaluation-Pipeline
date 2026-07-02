"""
Evaluation metrics for ASR output: Word Error Rate and Character Error
Rate, computed with the `jiwer` library over the full list of
references/hypotheses (corpus-level WER/CER, the standard reporting
convention for ASR benchmarks).
"""

import jiwer


def compute_wer(references, hypotheses) -> float:
    return jiwer.wer(references, hypotheses)


def compute_cer(references, hypotheses) -> float:
    return jiwer.cer(references, hypotheses)

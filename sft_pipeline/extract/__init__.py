"""Extract subpackage: LLM-based PKVC extraction + model output parsing.

Modules
-------
llm       Single-text PKVC extractor (``extract_pkvc``) + multi-family LLM clients
          (OpenAI / Anthropic / DeepSeek). Defines ``ExtractionResult`` schema.
batch     Async batch driver over JSONL (concurrency-controlled, cached, optional
          κ pilot for inter-rater agreement). CLI entry for production extraction.
parser    Parse model SFT-time generation output back into structured records
          (used at inference time + by some eval flows).

CLI entries
-----------
* ``python -m sft_pipeline.extract.batch --input q7_segments.jsonl --output q7_extracted.jsonl``
* ``python -m sft_pipeline.extract.llm --text "..."``  (single-segment smoke)
"""
from .llm import (
    ExtractionResult, extract_pkvc, extract_pkvc_batch_async,
    compute_extraction_kappa,
)

__all__ = [
    "ExtractionResult", "extract_pkvc", "extract_pkvc_batch_async",
    "compute_extraction_kappa",
]

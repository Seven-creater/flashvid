"""Unified multiple-choice evaluation utilities for long-video benchmarks."""

from .answers import extract_answer_letter, extract_strict_answer_letter
from .datasets import load_samples
from .schemas import Sample

__all__ = ["Sample", "extract_answer_letter", "extract_strict_answer_letter", "load_samples"]

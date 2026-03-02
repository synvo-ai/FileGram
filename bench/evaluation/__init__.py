"""Evaluation methods for FileGramBench.

Two complementary evaluation types:
1. LLM-as-Judge: per-attribute scoring of inferred vs ground-truth profiles
2. Trajectory MCQ: auto-generated multiple-choice questions from trajectory data
"""

from .judge_scoring import JudgeScorer
from .mcq_generator import MCQGenerator

__all__ = ["JudgeScorer", "MCQGenerator"]

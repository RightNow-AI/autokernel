"""Correctness verification for structured kernel outputs.

This package generalizes the benchmark harness beyond single-tensor outputs:

* :mod:`autokernel.verification.outputs` flattens and compares arbitrary
  output trees (tensors, tuples, lists, dictionaries, named tuples and
  nested combinations) leaf by leaf, with stable diagnostic paths.

Modules here never initialize a GPU at import time; ``torch`` is imported
lazily inside the functions that need it.
"""

from __future__ import annotations

from .corpus import (
    CORPUS_SCHEMA_VERSION,
    CorpusCase,
    CorpusError,
    ShapeCorpus,
    load_shape_corpus,
    validate_corpus_against_spec,
    weighted_aggregate,
)
from .outputs import (
    DEFAULT_TOLERANCE,
    LeafRecord,
    OutputTreeError,
    TreeComparison,
    compare_deterministic,
    compare_output_trees,
    flatten_output_tree,
    tree_has_nan_or_inf,
)

__all__ = [
    "CORPUS_SCHEMA_VERSION",
    "CorpusCase",
    "CorpusError",
    "DEFAULT_TOLERANCE",
    "LeafRecord",
    "OutputTreeError",
    "ShapeCorpus",
    "TreeComparison",
    "compare_deterministic",
    "compare_output_trees",
    "flatten_output_tree",
    "load_shape_corpus",
    "tree_has_nan_or_inf",
    "validate_corpus_against_spec",
    "weighted_aggregate",
]

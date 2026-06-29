"""Learned search guidance scaffold.

This package is intentionally separate from the production solver path. It
contains the plumbing for our approach: candidate child-path features, expert
training-row generation, and a lightweight ranker interface that can later be
backed by logistic regression, gradient boosting, or another model.
"""

from solver.learned_search.features import CandidateFeatures, candidate_features
from solver.learned_search.linear_ranker import LinearRanker

__all__ = ["CandidateFeatures", "LinearRanker", "candidate_features"]

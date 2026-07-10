"""
hungarian.py
============
Stage 7a: Thin wrapper around scipy's implementation of the Hungarian
(Kuhn-Munkres) algorithm for optimal bipartite assignment between
before-regions and after-regions, given a precomputed cost matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from modules.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AssignmentResult:
    before_indices: List[int]
    after_indices: List[int]
    costs: List[float]


def solve_assignment(cost_matrix: np.ndarray, max_cost: float = 1.0) -> AssignmentResult:
    """
    Solve the optimal assignment problem for a (N_before x N_after) cost
    matrix using the Hungarian algorithm.

    Args:
        cost_matrix: lower cost = better match. Values should already be
                     bounded/normalized (e.g. in [0, 1]) by the caller.
        max_cost: pairs with cost >= max_cost are treated as "do not match"
                  and are filtered out of the returned assignment even if
                  the solver paired them (this handles rectangular /
                  unequal-size matrices gracefully).

    Returns:
        AssignmentResult with parallel lists of matched before/after
        indices and their costs.
    """
    if cost_matrix.size == 0:
        return AssignmentResult([], [], [])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    before_indices: List[int] = []
    after_indices: List[int] = []
    costs: List[float] = []

    for r, c in zip(row_ind, col_ind):
        cost = float(cost_matrix[r, c])
        if cost >= max_cost:
            continue
        before_indices.append(int(r))
        after_indices.append(int(c))
        costs.append(cost)

    logger.debug(
        "Hungarian assignment solved: matrix_shape=%s -> %d valid matches",
        cost_matrix.shape, len(before_indices),
    )
    return AssignmentResult(before_indices, after_indices, costs)

import unittest

import numpy as np

from flytris.placeholder import W, features as legacy_features
from flytris.tetris import COLS, H
from flytris.v3_policy import (
    LinearAfterstatePolicy, afterstate_features, enumerate_afterstates,
    enumerate_transitions, play_policy, racing_elites,
)


class V3PolicyTests(unittest.TestCase):
    def test_first_four_features_preserve_proven_teacher(self):
        rng = np.random.default_rng(4)
        boards = (rng.random((12, H, COLS)) < 0.18).astype(np.uint8)
        lines = rng.integers(0, 5, len(boards))
        old = legacy_features(boards, lines) @ W
        new = afterstate_features(boards, lines)[:, :4] @ W
        np.testing.assert_allclose(new, old)

    def test_racing_only_promotes_shortlisted_candidates(self):
        primary = [10, 9, 8, 7, 6]
        extra = [[0], [0], [100], [1000]]
        elite = racing_elites(primary, extra, shortlist=4, elites=2)
        self.assertEqual(elite.tolist(), [3, 2])

    def test_policy_replay_is_deterministic(self):
        weights = np.zeros(9)
        weights[:4] = W
        policy = LinearAfterstatePolicy(np.zeros(9), np.ones(9), weights, 0.0)
        first = play_policy(policy, seed=712345, cap=40)
        second = play_policy(policy, seed=712345, cap=40)
        self.assertEqual((first["lines"], first["pieces"], first["shaping"]),
                         (second["lines"], second["pieces"], second["shaping"]))
        np.testing.assert_array_equal(first["final_board"], second["final_board"])

    def test_transition_afterstate_matches_existing_drop(self):
        board = np.zeros((H, COLS), np.uint8)
        board[-1, :6] = 1
        for piece in range(7):
            _, after, lines, dead = enumerate_transitions(board, piece)
            old_after, old_lines, old_dead = enumerate_afterstates(board, piece)
            np.testing.assert_array_equal(after, old_after)
            np.testing.assert_array_equal(lines, old_lines)
            np.testing.assert_array_equal(dead, old_dead)


if __name__ == "__main__":
    unittest.main()

"""Tests for deterministic seed derivation."""

from __future__ import annotations

from gwmock.simulator.seeds import derive_seed


def test_derive_seed_is_stable_across_runs() -> None:
    """The same parent and labels should always map to the same seed."""
    assert derive_seed(42, "signal", 0) == 8512427659370527145


def test_derive_seed_smoke_test_for_uniqueness() -> None:
    """Nearby derivation inputs should produce distinct bounded seeds."""
    seeds = (
        {derive_seed(42, "signal", segment_index) for segment_index in range(8)}
        | {derive_seed(42, "noise", segment_index) for segment_index in range(8)}
        | {derive_seed(42, label) for label in ("population", "signal", "noise", "third_party")}
    )

    assert len(seeds) == 20
    assert all(0 <= seed < 2**63 for seed in seeds)

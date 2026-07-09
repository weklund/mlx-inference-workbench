import numpy as np

from workbench.comparability import GateResult
from workbench.models import DistributionStats
from workbench.statistics_compare import compare_distributions


def _dist(values: list[float]) -> DistributionStats:
    arr = np.asarray(values, dtype=float)
    return DistributionStats(
        n=len(values),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)),
        cov=float(arr.std(ddof=1) / arr.mean()),
        trimmed_mean=float(arr.mean()),
        percentiles={"p50": float(np.median(arr))},
        values=tuple(values),
    )


def test_no_significant_difference_identical():
    vals = [10.0, 10.1, 9.9, 10.0, 10.05]
    gate = GateResult(comparable=True, violations=())
    r = compare_distributions(_dist(vals), _dist(vals), metric_name="decode_tok_s", gate=gate)
    assert r.comparable
    assert r.verdict == "no_significant_difference"
    assert r.significant_at_0_05 is False


def test_blocked_gate():
    gate = GateResult(comparable=False, violations=("prompt_dataset_hash: a != b",))
    r = compare_distributions(_dist([1, 2, 3]), _dist([4, 5, 6]), metric_name="x", gate=gate)
    assert r.verdict == "blocked_by_comparability_gate"

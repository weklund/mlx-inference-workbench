"""Statistical comparison of two runs (after comparability gate passes)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats

from workbench.comparability import GateResult, check_runs_comparable
from workbench.models import DistributionStats, RunRecord


@dataclass(frozen=True)
class ComparisonResult:
    metric_name: str
    comparable: bool
    violations: tuple[str, ...]
    n_a: int
    n_b: int
    mean_a: float | None
    mean_b: float | None
    mean_diff: (
        float | None
    )  # b - a (positive => b faster for tok/s; for latency invert interpretation)
    p_value: float | None
    test_name: str | None
    cohens_d: float | None
    ci95_low: float | None
    ci95_high: float | None
    significant_at_0_05: bool | None
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled)


def _values(dist: DistributionStats | None) -> np.ndarray | None:
    if dist is None or dist.n == 0:
        return None
    if dist.values:
        return np.asarray(dist.values, dtype=np.float64)
    return None


def compare_distributions(
    a: DistributionStats | None,
    b: DistributionStats | None,
    *,
    metric_name: str,
    gate: GateResult,
    alpha: float = 0.05,
) -> ComparisonResult:
    if not gate.comparable:
        return ComparisonResult(
            metric_name=metric_name,
            comparable=False,
            violations=gate.violations,
            n_a=0,
            n_b=0,
            mean_a=None,
            mean_b=None,
            mean_diff=None,
            p_value=None,
            test_name=None,
            cohens_d=None,
            ci95_low=None,
            ci95_high=None,
            significant_at_0_05=None,
            verdict="blocked_by_comparability_gate",
        )

    va, vb = _values(a), _values(b)
    if va is None or vb is None or len(va) < 2 or len(vb) < 2:
        return ComparisonResult(
            metric_name=metric_name,
            comparable=True,
            violations=(),
            n_a=0 if va is None else len(va),
            n_b=0 if vb is None else len(vb),
            mean_a=None if va is None else float(np.mean(va)),
            mean_b=None if vb is None else float(np.mean(vb)),
            mean_diff=None,
            p_value=None,
            test_name=None,
            cohens_d=None,
            ci95_low=None,
            ci95_high=None,
            significant_at_0_05=None,
            verdict="insufficient_data_for_test",
        )

    # Normality: if either side fails Shapiro (n>=3), use Mann-Whitney
    use_mw = False
    if len(va) >= 3 and len(vb) >= 3:
        _, pa = stats.shapiro(va)
        _, pb = stats.shapiro(vb)
        if pa < 0.05 or pb < 0.05:
            use_mw = True
    elif len(va) < 3 or len(vb) < 3:
        use_mw = True

    if use_mw:
        stat = stats.mannwhitneyu(va, vb, alternative="two-sided")
        p_value = float(stat.pvalue)
        test_name = "mannwhitney_u"
        # CI via bootstrap on mean difference
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(2000):
            sa = rng.choice(va, size=len(va), replace=True)
            sb = rng.choice(vb, size=len(vb), replace=True)
            boots.append(float(np.mean(sb) - np.mean(sa)))
        ci_low, ci_high = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    else:
        # Welch t-test
        t_res = stats.ttest_ind(va, vb, equal_var=False)
        p_value = float(t_res.pvalue)
        test_name = "welch_t"
        mean_diff = float(np.mean(vb) - np.mean(va))
        se = np.sqrt(np.var(va, ddof=1) / len(va) + np.var(vb, ddof=1) / len(vb))
        # Welch–Satterthwaite df
        v1, v2 = np.var(va, ddof=1), np.var(vb, ddof=1)
        n1, n2 = len(va), len(vb)
        df_num = (v1 / n1 + v2 / n2) ** 2
        df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = df_num / df_den if df_den > 0 else max(n1 + n2 - 2, 1)
        t_crit = float(stats.t.ppf(1 - alpha / 2, df))
        ci_low, ci_high = mean_diff - t_crit * se, mean_diff + t_crit * se

    mean_a, mean_b = float(np.mean(va)), float(np.mean(vb))
    mean_diff = mean_b - mean_a
    d = _cohens_d(va, vb)
    significant = p_value < alpha

    if not significant:
        verdict = "no_significant_difference"
    else:
        direction = "higher" if mean_diff > 0 else "lower"
        verdict = f"significant_{direction}_b_vs_a"

    return ComparisonResult(
        metric_name=metric_name,
        comparable=True,
        violations=(),
        n_a=len(va),
        n_b=len(vb),
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_diff,
        p_value=p_value,
        test_name=test_name,
        cohens_d=d if d == d else None,  # NaN check
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        significant_at_0_05=significant,
        verdict=verdict,
    )


def compare_runs(
    run_a: RunRecord,
    run_b: RunRecord,
    *,
    metric_name: str = "decode_tok_s",
) -> ComparisonResult:
    gate = check_runs_comparable(run_a, run_b)
    dist_a = getattr(run_a.metrics, metric_name, None)
    dist_b = getattr(run_b.metrics, metric_name, None)
    return compare_distributions(dist_a, dist_b, metric_name=metric_name, gate=gate)

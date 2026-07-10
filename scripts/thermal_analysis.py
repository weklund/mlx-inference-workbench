"""Phase 0.5: Analyze thermal validation data and produce verdict.

Reads docs/spikes/005_thermal_data.jsonl and computes:
- Overall CoV (pass/fail against <5% threshold)
- Per-session breakdown (morning vs afternoon vs evening)
- Day-to-day variance
- Thermal correlation (if data available)

Usage:
    uv run python scripts/thermal_analysis.py
"""

import json
from pathlib import Path

import click
import numpy as np

RESULTS_FILE = Path(__file__).parent.parent / "docs" / "spikes" / "005_thermal_data.jsonl"
COV_THRESHOLD = 5.0


def _is_protocol_run(r: dict) -> bool:
    """AC + powermetrics + not explicitly excluded from the official gate."""
    if r.get("excluded_from_protocol_gate") is True:
        return False
    if r.get("cohort") == "exploratory":
        return False
    return (
        r.get("power_source") == "ac"
        and r.get("thermal_before", {}).get("method") == "powermetrics"
    )


def _cov_pct(rates: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(rates, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    cov = (std / mean) * 100 if mean > 0 else float("inf")
    return mean, std, cov


@click.command()
@click.option(
    "--valid-only",
    is_flag=True,
    help="Protocol gate only: AC + powermetrics; drop exploratory/battery rows",
)
def main(valid_only: bool):
    """Analyze thermal reproducibility validation data."""
    if not RESULTS_FILE.exists():
        click.echo(f"No data found at {RESULTS_FILE}. Run thermal_validation.py first.")
        raise SystemExit(1)

    all_records = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    if not all_records:
        click.echo("No records found.")
        raise SystemExit(1)

    protocol = [r for r in all_records if _is_protocol_run(r)]
    exploratory = [r for r in all_records if r not in protocol]

    records = protocol if valid_only else all_records
    if not records:
        click.echo("No valid records after filtering. Need AC + thermal data.")
        raise SystemExit(1)

    click.echo("Phase 0.5: Thermal Reproducibility Analysis")
    click.echo(f"  Total rows in file: {len(all_records)}")
    click.echo(f"  Analyzing: {len(records)} ({'protocol only' if valid_only else 'all cohorts'})")
    click.echo(f"  Model: {records[0].get('model_id', 'unknown')}")
    click.echo()

    all_rates = [r["tok_per_sec"] for r in records]
    mean_all = np.mean(all_rates)
    std_all = np.std(all_rates)
    cov_all = (std_all / mean_all) * 100 if mean_all > 0 else float("inf")

    click.echo("=" * 60)
    click.echo("OVERALL RESULTS")
    click.echo(f"  Mean:     {mean_all:.2f} tok/s")
    click.echo(f"  Std:      {std_all:.2f} tok/s")
    click.echo(f"  CoV:      {cov_all:.2f}%")
    click.echo(f"  p50:      {np.percentile(all_rates, 50):.2f} tok/s")
    click.echo(f"  p90:      {np.percentile(all_rates, 90):.2f} tok/s")
    click.echo(f"  p99:      {np.percentile(all_rates, 99):.2f} tok/s")
    click.echo(f"  Min:      {np.min(all_rates):.2f} tok/s")
    click.echo(f"  Max:      {np.max(all_rates):.2f} tok/s")
    click.echo()
    verdict = "✓ PASS" if cov_all < COV_THRESHOLD else "✗ FAIL"
    click.echo(f"  VERDICT: {verdict} (CoV {cov_all:.2f}% vs threshold {COV_THRESHOLD}%)")
    click.echo("=" * 60)
    click.echo()

    # Per-session breakdown
    sessions = {}
    for r in records:
        key = f"Day {r['day']}, {r['session']}"
        sessions.setdefault(key, []).append(r["tok_per_sec"])

    click.echo("PER-SESSION BREAKDOWN")
    click.echo(f"{'Session':<25} {'N':>4} {'Mean':>8} {'Std':>8} {'CoV%':>8}")
    click.echo("-" * 60)
    for key in sorted(sessions.keys()):
        rates = sessions[key]
        m = np.mean(rates)
        s = np.std(rates)
        c = (s / m) * 100 if m > 0 else 0
        click.echo(f"{key:<25} {len(rates):>4} {m:>8.2f} {s:>8.2f} {c:>8.2f}")
    click.echo()

    # Day-to-day comparison
    days = {}
    for r in records:
        days.setdefault(r["day"], []).append(r["tok_per_sec"])

    if len(days) >= 2:
        click.echo("DAY-TO-DAY COMPARISON")
        for day_num in sorted(days.keys()):
            rates = days[day_num]
            click.echo(
                f"  Day {day_num}: mean={np.mean(rates):.2f}, std={np.std(rates):.2f}, n={len(rates)}"
            )

        day_means = [np.mean(days[d]) for d in sorted(days.keys())]
        drift = abs(day_means[0] - day_means[1]) / np.mean(day_means) * 100
        click.echo(f"  Day-to-day drift: {drift:.2f}%")
        click.echo()

    # Time-of-day comparison
    time_sessions = {}
    for r in records:
        time_sessions.setdefault(r["session"], []).append(r["tok_per_sec"])

    if len(time_sessions) >= 2:
        click.echo("TIME-OF-DAY COMPARISON")
        for sess in ["morning", "afternoon", "evening"]:
            if sess in time_sessions:
                rates = time_sessions[sess]
                click.echo(
                    f"  {sess:>10}: mean={np.mean(rates):.2f}, std={np.std(rates):.2f}, n={len(rates)}"
                )
        click.echo()

    # Thermal correlation (power draw vs tok/s)
    cpu_power_before = []
    combined_power_before = []
    rates_with_power = []
    for r in records:
        tb = r.get("thermal_before", {})
        if tb.get("method") == "powermetrics" and "cpu_power_mw" in tb:
            cpu_power_before.append(tb["cpu_power_mw"])
            combined_power_before.append(tb.get("combined_power_mw", 0))
            rates_with_power.append(r["tok_per_sec"])

    if len(cpu_power_before) >= 5:
        corr_cpu = np.corrcoef(cpu_power_before, rates_with_power)[0, 1]
        corr_combined = np.corrcoef(combined_power_before, rates_with_power)[0, 1]
        click.echo("POWER CORRELATION (pre-run power draw vs tok/s)")
        click.echo(f"  Pearson(cpu_power_before, tok/s):      {corr_cpu:.3f}")
        click.echo(f"  Pearson(combined_power_before, tok/s):  {corr_combined:.3f}")

        def interpret(c):
            return "strong" if abs(c) > 0.7 else "moderate" if abs(c) > 0.4 else "weak"

        click.echo(
            f"  Interpretation: CPU={interpret(corr_cpu)}, combined={interpret(corr_combined)}"
        )
        click.echo()

    # Cohort summary (always from full file so exploratories stay visible)
    click.echo("DATA COHORTS")
    click.echo(f"  Protocol (AC + powermetrics): {len(protocol)}")
    if protocol:
        m, s, c = _cov_pct([r["tok_per_sec"] for r in protocol])
        click.echo(f"    mean={m:.2f} tok/s  CoV={c:.2f}%  → gate uses this with --valid-only")
    click.echo(f"  Exploratory / non-protocol:  {len(exploratory)}")
    if exploratory:
        m, s, c = _cov_pct([r["tok_per_sec"] for r in exploratory])
        click.echo(f"    mean={m:.2f} tok/s  CoV={c:.2f}%  (stability under different inputs)")
        reasons: set[str] = set()
        for r in exploratory:
            for reason in r.get("exclusion_reasons") or []:
                reasons.add(str(reason))
            if r.get("power_source") != "ac":
                reasons.add("battery")
            if r.get("thermal_before", {}).get("method") != "powermetrics":
                reasons.add("no thermal data")
        if reasons:
            click.echo(f"    tags/reasons: {', '.join(sorted(reasons))}")
        # Per-session exploratory CoV (e.g. battery evening still tight)
        exp_sessions: dict[str, list[float]] = {}
        for r in exploratory:
            key = f"Day {r['day']}, {r['session']}"
            exp_sessions.setdefault(key, []).append(r["tok_per_sec"])
        for key in sorted(exp_sessions):
            m, s, c = _cov_pct(exp_sessions[key])
            click.echo(f"    {key}: n={len(exp_sessions[key])} mean={m:.2f} CoV={c:.2f}%")
    click.echo()


if __name__ == "__main__":
    main()

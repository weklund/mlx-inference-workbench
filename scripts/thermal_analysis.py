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


@click.command()
@click.option(
    "--valid-only", is_flag=True, help="Exclude battery runs and runs without thermal data"
)
def main(valid_only: bool):
    """Analyze thermal reproducibility validation data."""
    if not RESULTS_FILE.exists():
        click.echo(f"No data found at {RESULTS_FILE}. Run thermal_validation.py first.")
        raise SystemExit(1)

    records = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        click.echo("No records found.")
        raise SystemExit(1)

    if valid_only:
        records = [
            r
            for r in records
            if r.get("power_source") == "ac"
            and r.get("thermal_before", {}).get("method") == "powermetrics"
        ]
        if not records:
            click.echo("No valid records after filtering. Need AC + thermal data.")
            raise SystemExit(1)

    click.echo("Phase 0.5: Thermal Reproducibility Analysis")
    click.echo(f"  Total runs: {len(records)}")
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

    # Data quality summary
    valid_records = [
        r
        for r in records
        if r.get("power_source") == "ac"
        and r.get("thermal_before", {}).get("method") == "powermetrics"
    ]
    invalid_records = [r for r in records if r not in valid_records]
    click.echo("DATA QUALITY")
    click.echo(f"  Valid runs (AC + thermal):   {len(valid_records)}")
    click.echo(f"  Invalid runs (excluded):     {len(invalid_records)}")
    if invalid_records:
        reasons = set()
        for r in invalid_records:
            if r.get("power_source") != "ac":
                reasons.add("battery")
            if r.get("thermal_before", {}).get("method") != "powermetrics":
                reasons.add("no thermal data")
        click.echo(f"  Exclusion reasons: {', '.join(sorted(reasons))}")
    click.echo()


if __name__ == "__main__":
    main()

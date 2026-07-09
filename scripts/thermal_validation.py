"""Phase 0.5: Thermal Reproducibility Validation Script.

Runs N inference iterations with cooldown between each, logging timing
and thermal data. Designed to be run multiple times across different
sessions (morning/afternoon/evening) over 2 days.

Usage:
    uv run python scripts/thermal_validation.py --runs 5 --session morning --day 1
    uv run python scripts/thermal_validation.py --runs 5 --session afternoon --day 1
    uv run python scripts/thermal_validation.py --runs 5 --session evening --day 1
    # ... repeat day 2

Results append to: docs/spikes/005_thermal_data.jsonl
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import mlx.core as mx
import numpy as np

RESULTS_FILE = Path(__file__).parent.parent / "docs" / "spikes" / "005_thermal_data.jsonl"
MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
PROMPT = (
    "Write a Python function that implements binary search on a sorted list. "
    "Include type hints, docstring, and handle edge cases."
)
MAX_TOKENS = 256
COOLDOWN_SEC = 30
SEED = 42


def get_power_mode() -> dict:
    """Read macOS power mode: 0=low_power, 1=automatic, 2=high_performance."""
    try:
        result = subprocess.run(
            ["pmset", "-g"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.split("\n"):
            if "powermode" in line:
                val = int(line.strip().split()[-1])
                labels = {0: "low_power", 1: "automatic", 2: "high_performance"}
                return {"powermode": val, "powermode_label": labels.get(val, f"unknown_{val}")}
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return {"powermode": None, "powermode_label": "unknown"}


def get_power_source() -> str:
    """Check if running on battery or AC power."""
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout.lower()
        if "ac power" in output:
            return "ac"
        elif "battery power" in output:
            return "battery"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def get_thermal_state() -> dict:
    """Read chip temperature via powermetrics (requires sudo) or fallback."""
    try:
        result = subprocess.run(
            ["sudo", "powermetrics", "--samplers", "smc", "-i", "1", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = result.stdout.strip().split("\n")
        temps = {}
        for line in lines:
            if "die temperature" in line.lower() or "cpu die" in line.lower():
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        temps["die_temp_c"] = float(parts[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
        if temps:
            return {"method": "powermetrics", **temps}
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        pass

    try:
        result = subprocess.run(
            ["sysctl", "machdep.xcpm.cpu_thermal_level"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            level = result.stdout.strip().split(":")[-1].strip()
            return {"method": "sysctl_thermal_level", "thermal_level": int(level)}
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return {"method": "unavailable"}


def run_single_inference(model, tokenizer) -> dict:
    """Run one inference pass and return timing data."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt_tokens = tokenizer.encode(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    prompt_len = len(prompt_tokens)

    sampler = make_sampler(temp=0.0)

    t_start = time.perf_counter()
    output = generate(
        model,
        tokenizer,
        prompt=PROMPT,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
    )
    mx.synchronize()
    t_end = time.perf_counter()

    output_tokens = tokenizer.encode(output)
    num_output_tokens = len(output_tokens)
    total_time = t_end - t_start
    tok_per_sec = num_output_tokens / total_time if total_time > 0 else 0

    return {
        "prompt_tokens": prompt_len,
        "output_tokens": num_output_tokens,
        "total_time_s": round(total_time, 4),
        "tok_per_sec": round(tok_per_sec, 2),
    }


@click.command()
@click.option("--runs", default=5, help="Number of runs this session")
@click.option("--session", type=click.Choice(["morning", "afternoon", "evening"]), required=True)
@click.option("--day", type=int, required=True, help="Day number (1 or 2)")
@click.option("--cooldown", default=COOLDOWN_SEC, help="Seconds between runs")
@click.option("--model-id", default=MODEL_ID, help="HuggingFace model ID")
def main(runs: int, session: str, day: int, cooldown: int, model_id: str):
    """Run thermal reproducibility validation measurements."""
    from mlx_lm import load

    power_mode = get_power_mode()
    power_source = get_power_source()

    click.echo("Phase 0.5: Thermal Reproducibility Validation")
    click.echo(f"  Model: {model_id}")
    click.echo(f"  Session: Day {day}, {session}")
    click.echo(f"  Runs: {runs}, Cooldown: {cooldown}s")
    click.echo(f"  Power mode: {power_mode['powermode_label']} ({power_mode['powermode']})")
    click.echo(f"  Power source: {power_source}")
    click.echo()

    if power_mode["powermode"] == 0:
        click.echo("⚠️  WARNING: Low Power Mode is active. Results will be throttled and invalid.")
        click.echo("   Switch to High Performance mode: System Settings → Battery → Low Power Mode → Never")
        if not click.confirm("Continue anyway?"):
            raise SystemExit(1)

    if power_source == "battery":
        click.echo("⚠️  WARNING: Running on battery. Plug in for reproducible results.")
        click.echo("   (Proceeding anyway — results will be tagged as battery)")
        click.echo()

    click.echo("Loading model...")
    model, tokenizer = load(model_id)
    click.echo("Model loaded. Running warmup (3 iterations)...")

    for _ in range(3):
        run_single_inference(model, tokenizer)

    click.echo("Warmup complete. Starting timed measurements.")
    click.echo()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for i in range(runs):
        if i > 0:
            click.echo(f"  Cooling down ({cooldown}s)...")
            time.sleep(cooldown)

        thermal_before = get_thermal_state()
        run_data = run_single_inference(model, tokenizer)
        thermal_after = get_thermal_state()

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "day": day,
            "session": session,
            "run_index": i,
            "model_id": model_id,
            "cooldown_sec": cooldown,
            "power_mode": power_mode["powermode"],
            "power_mode_label": power_mode["powermode_label"],
            "power_source": power_source,
            "thermal_before": thermal_before,
            "thermal_after": thermal_after,
            **run_data,
        }

        results.append(record)

        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        click.echo(
            f"  Run {i+1}/{runs}: {run_data['tok_per_sec']:.1f} tok/s "
            f"({run_data['output_tokens']} tokens in {run_data['total_time_s']:.2f}s)"
        )

    tok_rates = [r["tok_per_sec"] for r in results]
    mean_rate = np.mean(tok_rates)
    std_rate = np.std(tok_rates)
    cov = (std_rate / mean_rate) * 100 if mean_rate > 0 else float("inf")

    click.echo()
    click.echo(f"Session Summary (Day {day}, {session}):")
    click.echo(f"  Mean: {mean_rate:.2f} tok/s")
    click.echo(f"  Std:  {std_rate:.2f} tok/s")
    click.echo(f"  CoV:  {cov:.2f}%")
    click.echo(f"  {'✓ PASS' if cov < 5 else '✗ FAIL'} (threshold: <5%)")
    click.echo(f"  Results appended to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()

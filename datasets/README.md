# Prompt datasets

Immutable JSONL corpora for the benchmarking harness. Paths are referenced from experiment configs (`benchmark.prompt_dataset` + `benchmark.prompt_checksum`).

## Format

One JSON object per line:

| Field | Required | Description |
|-------|----------|-------------|
| `id` | yes | Stable unique id within the file |
| `category` | yes | Label for stratification / reporting |
| `prompt` | yes | User-facing prompt text (single-request) |
| `expected_tokens_approx` | no | Soft size hint for planning |
| `reference` | no | Gold output for `require_correctness` (temp=0 / fixed seed) |

Encoding: UTF-8. The harness verifies integrity with a **SHA-256** sidecar in `sha256sum` form:

```text
<hex_digest>  <basename.jsonl>
```

## Datasets

| File | Role | Prompts |
|------|------|--------:|
| `smoke_v1.jsonl` | Offline stub CI smoke (tiny) | 3 |
| `agentic_coding_v1.jsonl` | Coding-agent proxy workloads (issue #6) | ≥20 |

### agentic_coding_v1

Categories (proxy, not production-validated):

- `tool_call` — JSON / tool-shaped agent I/O  
- `multi_turn` — continuation given prior thread context (still **one** generate call)  
- `code_generation` — implement a function/migration/handler  
- `refactor` — improve structure without changing intent  
- `explain_failure` — diagnose CI/prod failures  
- `debug` — root-cause + minimal fix sketches  

Designed for **single-request** runs (HLD). Not multi-agent concurrent sessions.

Config entrypoint: `configs/experiments/agentic_coding_v1_stub.yaml` (stub backend for harness checks). Point mlx-lm configs at the same dataset path for real baselines (#7).

## Versioning policy (immutable once published)

1. **Never mutate** a dataset file that has appeared in a published / comparable run. Changing bytes changes the hash and breaks comparability.
2. To evolve content, mint a **new** versioned basename (`agentic_coding_v2.jsonl` + matching `.sha256`).
3. Keep v1 files forever (or archive with the same bytes); configs pin an explicit path + checksum.
4. Optional `reference` fields may be added only in a new version if gold outputs are curated later — do not silently rewrite v1 lines.

### Mint a new version

```bash
# edit a copy, never the live published file in place for official runs
cp datasets/agentic_coding_v1.jsonl datasets/agentic_coding_v2.jsonl
# ... edit v2 ...
shasum -a 256 datasets/agentic_coding_v2.jsonl | awk '{print $1"  agentic_coding_v2.jsonl"}' \
  > datasets/agentic_coding_v2.sha256
```

### Verify hash

```bash
shasum -a 256 datasets/agentic_coding_v1.jsonl | diff - datasets/agentic_coding_v1.sha256
# or
make sync && uv run python -c "
from pathlib import Path
from workbench.prompts import load_dataset
load_dataset(
    Path('datasets/agentic_coding_v1.jsonl'),
    checksum_path=Path('datasets/agentic_coding_v1.sha256'),
)
print('ok')
"
```

Tampering the JSONL without updating the checksum must fail closed before generation (Prompt Manager).

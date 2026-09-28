#!/usr/bin/env python3
import csv, json, re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
raw = root / "results" / "raw"
rows = []

def metric_total(path, expected_name):
    total = 0.0
    if not path.exists(): return None
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("#"): continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name == expected_name:
            try: total += float(line.rsplit(" ", 1)[-1])
            except ValueError: pass
    return total

for path in sorted(raw.glob("*/*.json")):
    obj = json.loads(path.read_text())
    row = {k: obj.get(k) for k in (
        "label", "context_tokens_actual", "qps", "requests", "max_tokens",
        "ttft_p50_s", "ttft_p95_s", "tpot_p50_s", "tpot_p95_s", "output_tok_s", "wall_s")}
    row["measurement_valid"] = not (
        row["label"] == "pd_disaggregated" and (row["output_tok_s"] or 0) <= 0
    )
    before = path.parent / "metrics_before.txt"; after = path.parent / "metrics_after.txt"
    accepted_before = metric_total(before, "vllm:spec_decode_num_accepted_tokens_total") or 0
    accepted_after = metric_total(after, "vllm:spec_decode_num_accepted_tokens_total") or 0
    drafted_before = metric_total(before, "vllm:spec_decode_num_draft_tokens_total") or 0
    drafted_after = metric_total(after, "vllm:spec_decode_num_draft_tokens_total") or 0
    accepted = accepted_after - accepted_before; drafted = drafted_after - drafted_before
    row["accepted_tokens"] = accepted if drafted else None
    row["drafted_tokens"] = drafted if drafted else None
    row["acceptance_rate"] = accepted / drafted if drafted > 0 else None
    rows.append(row)

out = root / "results" / "benchmark.csv"
out.parent.mkdir(parents=True, exist_ok=True)
fields = list(rows[0]) if rows else []
with out.open("w", newline="") as f:
    if fields:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(rows)
(root / "results" / "benchmark.json").write_text(json.dumps(rows, indent=2))

domain_rows = []
for path in sorted((root / "results" / "domain").glob("*/gsm8k_qps4.json")):
    obj = json.loads(path.read_text())
    row = {k: obj.get(k) for k in (
        "label", "requests", "qps", "max_tokens", "ttft_p50_s", "ttft_p95_s",
        "tpot_p50_s", "tpot_p95_s", "output_tok_s", "wall_s", "exact_match")}
    before = path.parent / "metrics_before.txt"; after = path.parent / "metrics_after.txt"
    accepted_before = metric_total(before, "vllm:spec_decode_num_accepted_tokens_total") or 0
    accepted_after = metric_total(after, "vllm:spec_decode_num_accepted_tokens_total") or 0
    drafted_before = metric_total(before, "vllm:spec_decode_num_draft_tokens_total") or 0
    drafted_after = metric_total(after, "vllm:spec_decode_num_draft_tokens_total") or 0
    accepted = accepted_after - accepted_before; drafted = drafted_after - drafted_before
    row["accepted_tokens"] = accepted if drafted else None
    row["drafted_tokens"] = drafted if drafted else None
    row["acceptance_rate"] = accepted / drafted if drafted > 0 else None
    domain_rows.append(row)

domain_fields = list(domain_rows[0]) if domain_rows else []
with (root / "results" / "domain_benchmark.csv").open("w", newline="") as f:
    if domain_fields:
        w = csv.DictWriter(f, fieldnames=domain_fields, lineterminator="\n"); w.writeheader(); w.writerows(domain_rows)
(root / "results" / "domain_benchmark.json").write_text(json.dumps(domain_rows, indent=2))

baseline = {(r["context_tokens_actual"], r["qps"]): r for r in rows if r["label"] == "baseline_bf16"}
lines = ["# H200 LLM inference optimization — measured results", "", "All values below are generated from preserved raw request records.", "",
         "| Configuration | Context | QPS | TTFT p50 (s) | TPOT p50 (s) | output tok/s | acceptance | valid |",
         "|---|---:|---:|---:|---:|---:|---:|---|"]
for r in rows:
    ar = "—" if r["acceptance_rate"] is None else f'{r["acceptance_rate"]:.3f}'
    valid = "yes" if r["measurement_valid"] else "no — response framing yielded zero parsed tokens"
    lines.append(f'| {r["label"]} | {r["context_tokens_actual"]} | {r["qps"]} | {r["ttft_p50_s"]:.4f} | {r["tpot_p50_s"]:.5f} | {r["output_tok_s"]:.1f} | {ar} | {valid} |')
lines += ["", "## Interpretation guardrail", "", "Custom, pretrained, and baseline results are separate runs. Missing configurations were not completed before the service-restoration deadline and are not inferred."]
(root / "results" / "MEASURED_RESULTS.md").write_text("\n".join(lines) + "\n")

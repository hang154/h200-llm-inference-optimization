#!/usr/bin/env python3
"""Build portable tables and figures from the preserved benchmark outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

bench = pd.read_csv(RESULTS / "benchmark.csv")
plot_bench = bench[bench.measurement_valid.astype(str).str.lower() == "true"]
bench.to_parquet(RESULTS / "benchmark.parquet", index=False)
bench.to_csv(RESULTS / "latency.csv", index=False)
acceptance = (
    plot_bench.dropna(subset=["acceptance_rate"])
    .groupby("label", as_index=False)[["accepted_tokens", "drafted_tokens"]]
    .max()
)
acceptance["acceptance_rate"] = acceptance.accepted_tokens / acceptance.drafted_tokens
acceptance.to_csv(RESULTS / "acceptance.csv", index=False)

telemetry_path = ROOT / "telemetry" / "gpu_2_3.csv"
if telemetry_path.exists():
    telemetry = pd.read_csv(telemetry_path, skipinitialspace=True)
    telemetry["utilization_gpu_pct"] = pd.to_numeric(telemetry.utilization_gpu_pct, errors="coerce")
    for column in ["memory_used_mib", "memory_total_mib", "power_w", "temperature_c"]:
        telemetry[column] = pd.to_numeric(telemetry[column], errors="coerce")
    telemetry.to_csv(RESULTS / "telemetry.csv", index=False)
    memory = telemetry.groupby("index", as_index=False).agg(
        peak_hbm_mib=("memory_used_mib", "max"),
        peak_utilization_pct=("utilization_gpu_pct", "max"),
        peak_power_w=("power_w", "max"),
    )
    memory.to_csv(RESULTS / "memory.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 6))
qps_slice = plot_bench[plot_bench.context_tokens_actual == 8192]
for label, group in qps_slice.groupby("label"):
    group = group.sort_values("qps")
    ax.plot(group.qps, group.tpot_p50_s, marker="o", label=label)
ax.set_xlabel("Offered QPS")
ax.set_ylabel("Median TPOT (s), 8K input")
ax.grid(alpha=.25)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIGURES / "tp_ot_vs_qps.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 6))
qps8 = plot_bench[plot_bench.qps == 8]
for label, group in qps8.groupby("label"):
    group = group.sort_values("context_tokens_actual")
    ax.plot(group.context_tokens_actual, group.ttft_p50_s, marker="o", label=label)
ax.set_xscale("log", base=2)
ax.set_xlabel("Input context tokens")
ax.set_ylabel("Median TTFT (s), QPS 8")
ax.grid(alpha=.25)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIGURES / "latency_vs_context.png", dpi=180)
plt.close(fig)

if not acceptance.empty:
    labels = acceptance.label.tolist()
    values = (acceptance.acceptance_rate * 100).tolist()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, values)
    ax.set_ylabel("Accepted draft tokens (%)")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "acceptance_vs_draft_length.png", dpi=180)
    plt.close(fig)

if telemetry_path.exists():
    fig, ax = plt.subplots(figsize=(11, 6))
    for gpu, group in telemetry.groupby("index"):
        ax.plot(range(len(group)), group.memory_used_mib / 1024, label=f"GPU {gpu}")
    ax.set_xlabel("2-second telemetry samples")
    ax.set_ylabel("HBM used (GiB)")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "gpu_hbm_timeline.png", dpi=180)
    plt.close(fig)

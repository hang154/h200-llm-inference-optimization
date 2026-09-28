# H200 LLM Inference Optimization

Measured Qwen3-8B inference experiments on two NVIDIA H200 GPUs:

- vanilla vLLM decoding;
- the public Red Hat EAGLE-3 drafter;
- a custom EAGLE-3 drafter trained on on-policy Qwen3-8B responses to the public GSM8K training split;
- short, medium, and long prompts under low and higher offered load;
- BF16 and FP8 weight/KV-cache configurations;
- an attempted two-GPU prefill/decode disaggregation smoke test.

The repository intentionally excludes model weights and generated training responses. It keeps configuration, hashes, raw aggregate measurements, telemetry, failure/recovery evidence, and scripts needed to repeat the experiment.

## Reproduce

The remote runner expects two available CUDA devices and `uv`:

```bash
bash scripts/setup_remote.sh
nohup bash scripts/remote_guardian.sh orchestrator \
  taskset -c 0-47 bash scripts/run_all.sh > logs/orchestrator_launcher.log 2>&1 &
```

Override `EXPERIMENT_ROOT` and GPU assignments in the scripts for another host. The run uses at most 48 CPU cores, splits serving and training affinity, writes atomic state files, records 2-second GPU telemetry, and retries failed workers with bounded cooldown. The deployment-specific service-restoration hook is intentionally excluded because it contains unrelated host paths; the published artifacts retain the verified restoration outcome.

## Evidence policy

- Every result is reported from preserved request-level JSON and telemetry.
- Baseline, pretrained, and custom-drafter runs are never merged.
- Missing deadline-limited configurations stay missing; they are not inferred.
- A custom drafter that regresses is retained as a negative result.
- The on-policy corpus is derived from `openai/gsm8k`; redistribution is intentionally omitted.

See `results/MEASURED_RESULTS.md` and `reports/INFERENCE_OPTIMIZATION_REPORT.md` after completion.

## Headline measured result

At 32K input tokens and QPS 8, pretrained EAGLE-3 with four draft tokens reduced median TPOT from 10.636 ms to 2.282 ms and increased output throughput from 461.2 to 805.3 tokens/s. The newly trained one-epoch drafter was a documented negative result. FP8 KV cache increased the modeled 40K-request concurrency from 18.96× to 37.92×, with a throughput trade-off at the saturated 32K point.

The NIXL prefill/decode request path completed, but its toy-proxy streaming response was incompatible with the generic benchmark parser, so no P/D performance claim is made. Read the [final report](reports/INFERENCE_OPTIMIZATION_REPORT.md) before reusing any number.

## Upstream references

- [vLLM Speculators](https://github.com/vllm-project/speculators)
- [Speculators training tutorial](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/tutorials/train.md)
- [vLLM disaggregated prefill](https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md)
- [vLLM NIXL connector usage](https://github.com/vllm-project/vllm/blob/main/docs/features/nixl_connector_usage.md)

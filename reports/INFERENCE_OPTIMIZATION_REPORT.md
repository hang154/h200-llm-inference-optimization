# H200 LLM inference optimization — final report

Generated: 2026-09-29 KST

## Outcome

This run completed a measured Qwen3-8B inference study on two NVIDIA H200 GPUs: vanilla vLLM, a public pretrained EAGLE-3 drafter, a newly trained domain drafter, FP8 weights, FP8 KV cache, held-out quality checks, GPU telemetry, guardian recovery, and a two-GPU NIXL prefill/decode smoke path.

The strongest production candidate was the pretrained EAGLE-3 drafter with four draft tokens. The one-epoch custom drafter was a negative result: it trained and loaded successfully, but its low draft acceptance erased most speculative-decoding benefit. FP8 KV cache approximately doubled modeled 40K-request capacity, but did not improve every latency/throughput point.

## Hardware and software

- 2 × NVIDIA H200, 143,771 MiB each; driver 580.126.20.
- vLLM 0.30.0, PyTorch 2.13.0, Transformers 5.17.0, NIXL 1.4.1.
- vLLM Speculators source commit `e3041ddb4e58d8d0fad86f938f44423bdf424ae9`.
- Target: `Qwen/Qwen3-8B`; FP8 target: `RedHatAI/Qwen3-8B-FP8-dynamic`.
- CPU execution was bounded to 48 cores, split 0–23 and 24–47 for concurrent lanes.

The full environment fingerprint is in `artifacts/environment.txt`.

## Experiment design

- Raw workload: deterministic prompts at 512, 8,192, and 32,768 actual input tokens; QPS 1 and 8; 128 generated tokens.
- Domain workload: the first 32 held-out questions from `openai/gsm8k/main/test`, QPS 4.
- Speculative configurations: pretrained and custom EAGLE-3, each at k=1,2,3,4.
- Quantization configurations: FP8 weights + BF16 KV, and FP8 weights + FP8 E4M3 KV.
- Quality check: identical held-out GSM8K-32 prompts with deterministic decoding for BF16 and FP8 weights.
- Telemetry: two-second GPU utilization, HBM, power, and temperature samples.

The target and FP8 model both advertise 40,960 native positions. The requested 64K and 128K points therefore exceed the supported context without an unvalidated RoPE override. They were recorded as unsupported rather than converted into misleading quality-sensitive measurements; see `artifacts/context_limit_probe.json`.

## EAGLE-3 results

At 32K context and QPS 8, pretrained EAGLE-3 k=4 reduced median TPOT from 10.636 ms to 2.282 ms (78.5%) and increased output throughput from 461.2 to 805.3 tokens/s (74.6%). At 8K/QPS 8 it reduced median TPOT by 77.7% and increased throughput by 35.3%. At 512/QPS 8 it reduced median TPOT by 75.6% and increased throughput by 31.8%.

On the held-out GSM8K-shaped workload, pretrained acceptance declined as draft length increased: 69.3%, 56.5%, 44.7%, and 36.0% for k=1..4. Even so, k=4 had the lowest median TPOT: 2.664 ms versus the 5.576 ms baseline.

The custom drafter completed 226 training steps over 1,000 on-policy Qwen3-8B GSM8K responses. Validation full-token accuracy by predicted position was 46.3%, 21.4%, and 10.1%. On raw prompts its aggregate acceptance fell from 17.0% at k=1 to 4.9% at k=4. On held-out GSM8K it fell from 30.4% to 10.3%. At 32K/QPS 8, custom k=1 slightly increased TPOT by 3.2% relative to baseline; longer draft lengths were worse. This is retained as evidence that a short, small on-policy run did not match the mature pretrained drafter.

## FP8 and KV-cache results

FP8 weights improved median TPOT across the tested points. At 8K/QPS 8:

| Configuration | Median TPOT | Output tok/s | Change vs BF16 baseline |
|---|---:|---:|---:|
| BF16 weights + BF16 KV | 6.922 ms | 692.8 | baseline |
| FP8 weights + BF16 KV | 4.883 ms | 766.9 | +10.7% throughput |
| FP8 weights + FP8 KV | 4.732 ms | 772.1 | +11.4% throughput |

At 32K/QPS 8, FP8 weights + BF16 KV produced 515.5 tok/s (+11.8%), while FP8 KV produced 485.8 tok/s (+5.3%). Thus FP8 KV increased capacity but was not the fastest saturated long-context configuration.

The vLLM capacity profiler reported:

| Configuration | KV tokens | Maximum 40K concurrency |
|---|---:|---:|
| BF16 baseline | 703,248 | 17.58× |
| FP8 weights + BF16 KV | 758,304 | 18.96× |
| FP8 weights + FP8 KV | 1,516,624 | 37.92× |

On the 32-item quality check, BF16 scored 12/32 and FP8 weights scored 14/32. Only 12/32 final predictions agreed exactly, so this sample does not establish numerical equivalence; it only shows no aggregate accuracy regression on this small fixed set.

## Prefill/decode disaggregation

Both Qwen3-8B workers started successfully with vLLM's NIXL connector, the official toy proxy initialized one prefill and one decode client, and all eight requests returned HTTP 200 through the prefill→decode path. However, the generic streaming benchmark observed zero decoded tokens because the toy proxy's response framing was not interpreted by that client. The recorded 30.86-second pseudo-TTFT and zero throughput are therefore **invalid as performance measurements**. This stage proves integration/startup and request traversal only; it does not support a latency-speedup claim.

## Reliability evidence

- All long-running stages used parent guardians with atomic state and bounded exponential cooldown.
- A serving worker was deliberately killed with `SIGKILL`; the guardian advanced from attempt 0 to 1 and restored an HTTP-healthy endpoint.
- An interrupted orchestrator was replaced by a resumable remaining-work runner; completed configurations were not rerun.
- A stopped telemetry lane was corrected to append across restarts and relaunched under its own guardian.
- The original GPU2/GPU3 services were restored after the experiment and health-checked separately.

See `artifacts/guardian_recovery_test.json` and `reports/FAILURE_AND_RECOVERY_LOG.md`.

## Claim boundary

The evidence supports direct H200 operation, vLLM integration, speculative-decoder training/deployment, latency and throughput benchmarking, quantization/KV-capacity trade-off analysis, NIXL integration smoke testing, and process recovery. It does not support a claim that the custom drafter improved production performance, that 64K/128K were benchmarked natively, or that the P/D path achieved a measured speedup.

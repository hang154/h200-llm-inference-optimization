# H200 LLM inference optimization — measured results

All values below are generated from preserved raw request records.

| Configuration | Context | QPS | TTFT p50 (s) | TPOT p50 (s) | output tok/s | acceptance | valid |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_bf16 | 32768 | 1.0 | 0.1550 | 0.00616 | 130.0 | — | yes |
| baseline_bf16 | 32768 | 8.0 | 0.1238 | 0.01064 | 461.2 | — | yes |
| baseline_bf16 | 512 | 1.0 | 0.0289 | 0.00514 | 131.5 | — | yes |
| baseline_bf16 | 512 | 8.0 | 0.0263 | 0.00552 | 741.4 | — | yes |
| baseline_bf16 | 8192 | 1.0 | 0.0590 | 0.00539 | 130.8 | — | yes |
| baseline_bf16 | 8192 | 8.0 | 0.0598 | 0.00692 | 692.8 | — | yes |
| custom_eagle3_k1 | 32768 | 1.0 | 0.1980 | 0.00554 | 129.5 | 0.170 | yes |
| custom_eagle3_k1 | 32768 | 8.0 | 0.2108 | 0.01098 | 463.5 | 0.170 | yes |
| custom_eagle3_k1 | 512 | 1.0 | 0.0280 | 0.00460 | 132.2 | 0.170 | yes |
| custom_eagle3_k1 | 512 | 8.0 | 0.0272 | 0.00477 | 767.5 | 0.170 | yes |
| custom_eagle3_k1 | 8192 | 1.0 | 0.0882 | 0.00518 | 130.8 | 0.170 | yes |
| custom_eagle3_k1 | 8192 | 8.0 | 0.0850 | 0.00678 | 699.6 | 0.170 | yes |
| custom_eagle3_k2 | 32768 | 1.0 | 0.2168 | 0.00610 | 128.8 | 0.100 | yes |
| custom_eagle3_k2 | 32768 | 8.0 | 0.1819 | 0.01131 | 463.2 | 0.100 | yes |
| custom_eagle3_k2 | 512 | 1.0 | 0.0303 | 0.00503 | 131.6 | 0.100 | yes |
| custom_eagle3_k2 | 512 | 8.0 | 0.0293 | 0.00519 | 750.0 | 0.100 | yes |
| custom_eagle3_k2 | 8192 | 1.0 | 0.0891 | 0.00553 | 130.1 | 0.100 | yes |
| custom_eagle3_k2 | 8192 | 8.0 | 0.0541 | 0.00715 | 689.8 | 0.100 | yes |
| custom_eagle3_k3 | 32768 | 1.0 | 0.2040 | 0.00643 | 127.6 | 0.064 | yes |
| custom_eagle3_k3 | 32768 | 8.0 | 0.1982 | 0.01174 | 445.7 | 0.064 | yes |
| custom_eagle3_k3 | 512 | 1.0 | 0.0297 | 0.00524 | 131.3 | 0.064 | yes |
| custom_eagle3_k3 | 512 | 8.0 | 0.0305 | 0.00547 | 736.0 | 0.064 | yes |
| custom_eagle3_k3 | 8192 | 1.0 | 0.0967 | 0.00587 | 129.7 | 0.064 | yes |
| custom_eagle3_k3 | 8192 | 8.0 | 0.0760 | 0.00747 | 673.0 | 0.064 | yes |
| custom_eagle3_k4 | 32768 | 1.0 | 0.2049 | 0.00666 | 127.4 | 0.049 | yes |
| custom_eagle3_k4 | 32768 | 8.0 | 0.2024 | 0.01160 | 444.9 | 0.049 | yes |
| custom_eagle3_k4 | 512 | 1.0 | 0.0322 | 0.00576 | 130.5 | 0.049 | yes |
| custom_eagle3_k4 | 512 | 8.0 | 0.0311 | 0.00610 | 713.1 | 0.049 | yes |
| custom_eagle3_k4 | 8192 | 1.0 | 0.0966 | 0.00645 | 128.8 | 0.049 | yes |
| custom_eagle3_k4 | 8192 | 8.0 | 0.0840 | 0.00801 | 643.1 | 0.049 | yes |
| fp8_weights_bf16_kv | 32768 | 1.0 | 0.1285 | 0.00485 | 132.3 | — | yes |
| fp8_weights_bf16_kv | 32768 | 8.0 | 0.1081 | 0.00866 | 515.5 | — | yes |
| fp8_weights_bf16_kv | 512 | 1.0 | 0.0245 | 0.00378 | 133.4 | — | yes |
| fp8_weights_bf16_kv | 512 | 8.0 | 0.0183 | 0.00393 | 813.2 | — | yes |
| fp8_weights_bf16_kv | 8192 | 1.0 | 0.0635 | 0.00403 | 132.6 | — | yes |
| fp8_weights_bf16_kv | 8192 | 8.0 | 0.0682 | 0.00488 | 766.9 | — | yes |
| fp8_weights_fp8_kv | 32768 | 1.0 | 0.1292 | 0.00437 | 132.8 | — | yes |
| fp8_weights_fp8_kv | 32768 | 8.0 | 0.1240 | 0.00962 | 485.8 | — | yes |
| fp8_weights_fp8_kv | 512 | 1.0 | 0.0238 | 0.00375 | 133.6 | — | yes |
| fp8_weights_fp8_kv | 512 | 8.0 | 0.0224 | 0.00392 | 818.7 | — | yes |
| fp8_weights_fp8_kv | 8192 | 1.0 | 0.0704 | 0.00391 | 132.8 | — | yes |
| fp8_weights_fp8_kv | 8192 | 8.0 | 0.0606 | 0.00473 | 772.1 | — | yes |
| pd_disaggregated | 8192 | 1.0 | 30.8596 | 0.00000 | 0.0 | — | no — response framing yielded zero parsed tokens |
| pretrained_eagle3_k1 | 32768 | 1.0 | 0.1301 | 0.00314 | 136.2 | 0.986 | yes |
| pretrained_eagle3_k1 | 32768 | 8.0 | 0.1260 | 0.00450 | 688.2 | 0.986 | yes |
| pretrained_eagle3_k1 | 512 | 1.0 | 0.0302 | 0.00275 | 135.1 | 0.986 | yes |
| pretrained_eagle3_k1 | 512 | 8.0 | 0.0251 | 0.00293 | 875.8 | 0.986 | yes |
| pretrained_eagle3_k1 | 8192 | 1.0 | 0.0510 | 0.00286 | 134.5 | 0.986 | yes |
| pretrained_eagle3_k1 | 8192 | 8.0 | 0.0527 | 0.00393 | 710.3 | 0.986 | yes |
| pretrained_eagle3_k2 | 32768 | 1.0 | 0.1464 | 0.00234 | 137.6 | 0.966 | yes |
| pretrained_eagle3_k2 | 32768 | 8.0 | 0.1342 | 0.00299 | 764.5 | 0.966 | yes |
| pretrained_eagle3_k2 | 512 | 1.0 | 0.0268 | 0.00193 | 136.2 | 0.966 | yes |
| pretrained_eagle3_k2 | 512 | 8.0 | 0.0281 | 0.00197 | 931.3 | 0.966 | yes |
| pretrained_eagle3_k2 | 8192 | 1.0 | 0.0756 | 0.00198 | 135.5 | 0.966 | yes |
| pretrained_eagle3_k2 | 8192 | 8.0 | 0.0689 | 0.00217 | 916.0 | 0.966 | yes |
| pretrained_eagle3_k3 | 32768 | 1.0 | 0.1343 | 0.00200 | 138.8 | 0.893 | yes |
| pretrained_eagle3_k3 | 32768 | 8.0 | 0.1379 | 0.00269 | 799.7 | 0.893 | yes |
| pretrained_eagle3_k3 | 512 | 1.0 | 0.0290 | 0.00159 | 136.8 | 0.893 | yes |
| pretrained_eagle3_k3 | 512 | 8.0 | 0.0260 | 0.00165 | 955.6 | 0.893 | yes |
| pretrained_eagle3_k3 | 8192 | 1.0 | 0.0569 | 0.00166 | 136.2 | 0.893 | yes |
| pretrained_eagle3_k3 | 8192 | 8.0 | 0.0442 | 0.00179 | 936.4 | 0.893 | yes |
| pretrained_eagle3_k4 | 32768 | 1.0 | 0.1359 | 0.00178 | 139.0 | 0.827 | yes |
| pretrained_eagle3_k4 | 32768 | 8.0 | 0.1340 | 0.00228 | 805.3 | 0.827 | yes |
| pretrained_eagle3_k4 | 512 | 1.0 | 0.0310 | 0.00131 | 137.2 | 0.827 | yes |
| pretrained_eagle3_k4 | 512 | 8.0 | 0.0232 | 0.00135 | 976.9 | 0.827 | yes |
| pretrained_eagle3_k4 | 8192 | 1.0 | 0.0739 | 0.00148 | 136.4 | 0.827 | yes |
| pretrained_eagle3_k4 | 8192 | 8.0 | 0.0715 | 0.00155 | 937.4 | 0.827 | yes |

## Interpretation guardrail

Custom, pretrained, and baseline results are separate runs. Missing configurations were not completed before the service-restoration deadline and are not inferred.

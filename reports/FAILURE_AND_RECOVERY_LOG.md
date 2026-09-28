# Failure and recovery log

This file is completed from runtime state and logs after the measured run. Failures are retained as engineering evidence rather than removed from the public record.

## Preflight failures already reproduced

1. The default `/home/work` loop mount had only 328 MiB free, causing uv wheel extraction to fail. All uv, Hugging Face, XDG, vLLM, TorchInductor, and temporary caches were moved to the experiment volume.
2. A duplicated setup launcher wrote the same atomic temporary state path. The duplicate process tree was removed and guardian signal handling was corrected to terminate its child process group and exit.
3. The legacy GPU2 guardian could restart a child after a child-first shutdown. Shutdown ordering was changed to kill the restart-owning parent first, then its captured descendants.
4. vLLM memory profiling overlapped the final release of that legacy process and rejected an inconsistent free-memory snapshot. The guarded retry succeeded after isolation.
5. A graceful `SIGTERM` recovery test exited with status zero and correctly did not trigger a restart. The destructive test was corrected to use `SIGKILL`; the guardian then incremented its attempt counter and restored a healthy endpoint.
6. The first telemetry worker was tied to the superseded orchestrator and stopped when that process tree was retired. Telemetry was changed from truncate-on-start to append-on-restart, relaunched under its own guardian, and duplicate launcher descendants were removed by exact PID before collection continued.
7. The NIXL P/D workers and official proxy returned HTTP 200 for all eight requests, but the generic SSE benchmark parsed zero output tokens from the proxy framing. The result is retained as an integration smoke test and explicitly invalidated as a performance measurement.
8. GPU2 restoration initially encountered a stale guardian lock whose recorded PID no longer existed. The dead PID was verified, only that exact lock file was removed, and the original supervised launch then resumed normally.

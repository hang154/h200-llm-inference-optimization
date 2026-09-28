#!/usr/bin/env python3
import argparse, asyncio, hashlib, json, math, statistics, time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--label", required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--qps", type=float, required=True)
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    seed = "Analyze this numbered sequence and continue with concise mathematical reasoning. 1, 1, 2, 3, 5, 8. "
    ids = tok.encode(seed * math.ceil(args.context / 20), add_special_tokens=False)[: args.context]
    prompt = tok.decode(ids)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    timeout = aiohttp.ClientTimeout(total=1200)
    results = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def request(i):
            await asyncio.sleep(i / args.qps)
            payload = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            start = time.perf_counter(); first = None; text = ""; usage = None
            async with session.post(args.endpoint + "/v1/completions", json=payload) as r:
                r.raise_for_status()
                async for raw in r.content:
                    for line in raw.decode(errors="ignore").splitlines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        obj = json.loads(line[6:])
                        chunk = obj.get("choices", [{}])[0].get("text", "") if obj.get("choices") else ""
                        if chunk and first is None: first = time.perf_counter()
                        text += chunk
                        usage = obj.get("usage") or usage
            end = time.perf_counter()
            n = (usage or {}).get("completion_tokens") or len(tok.encode(text, add_special_tokens=False))
            ttft = (first or end) - start
            tpot = (end - (first or end)) / max(1, n - 1)
            results.append({"request": i, "ttft_s": ttft, "total_s": end-start, "tpot_s": tpot, "output_tokens": n})

        wall_start = time.perf_counter()
        await asyncio.gather(*(request(i) for i in range(args.requests)))
        wall = time.perf_counter() - wall_start

    def pct(values, p):
        values = sorted(values)
        return values[min(len(values)-1, max(0, math.ceil(p*len(values))-1))]
    summary = {
        "label": args.label, "context_tokens_requested": args.context,
        "context_tokens_actual": len(ids), "qps": args.qps, "requests": args.requests,
        "max_tokens": args.max_tokens, "prompt_sha256": prompt_hash,
        "ttft_p50_s": statistics.median(x["ttft_s"] for x in results),
        "ttft_p95_s": pct([x["ttft_s"] for x in results], .95),
        "tpot_p50_s": statistics.median(x["tpot_s"] for x in results),
        "tpot_p95_s": pct([x["tpot_s"] for x in results], .95),
        "output_tok_s": sum(x["output_tokens"] for x in results) / wall,
        "wall_s": wall, "requests_detail": sorted(results, key=lambda x: x["request"]),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2)); tmp.replace(output)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
import argparse, asyncio, json, os, time
from pathlib import Path

import aiohttp
from datasets import load_dataset


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:18800/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = list(load_dataset("openai/gsm8k", "main", split="train"))[: args.samples]
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=600)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(index, row):
            prompt = (
                "Solve the following math problem carefully. Show a concise derivation, "
                "then put only the final answer after 'Final answer:'.\n\n" + row["question"]
            )
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 384,
            }
            async with sem:
                for attempt in range(6):
                    try:
                        async with session.post(args.endpoint + "/chat/completions", json=payload) as r:
                            if r.status == 429:
                                await asyncio.sleep(min(60, 2 ** attempt))
                                continue
                            r.raise_for_status()
                            answer = (await r.json())["choices"][0]["message"]["content"]
                            return index, {
                                "conversations": [
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": answer},
                                ],
                                "source": "openai/gsm8k-main",
                                "source_index": index,
                            }
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        if attempt == 5:
                            raise
                        await asyncio.sleep(min(60, 2 ** attempt))
            raise RuntimeError("unreachable")

        completed = await asyncio.gather(*(one(i, row) for i, row in enumerate(rows)))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w") as f:
        for _, row in sorted(completed):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(output)
    meta = {
        "dataset": "openai/gsm8k/main/train",
        "samples": len(completed),
        "model": args.model,
        "endpoint": args.endpoint,
        "generated_at_epoch": time.time(),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

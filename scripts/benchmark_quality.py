#!/usr/bin/env python3
import argparse
import asyncio
import json
import re
from pathlib import Path

import aiohttp
from datasets import load_dataset


def final_number(text):
    values = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return values[-1].replace(",", "") if values else None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = list(load_dataset("openai/gsm8k", "main", split="test"))[:32]
    timeout = aiohttp.ClientTimeout(total=1200)
    results = []
    semaphore = asyncio.Semaphore(4)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(index, row):
            async with semaphore:
                prompt = "Solve carefully. End with only 'Final answer: <number>'.\n\n" + row["question"]
                payload = {"model": "Qwen/Qwen3-8B", "prompt": prompt, "max_tokens": 512,
                           "temperature": 0.0, "stream": False}
                async with session.post(args.endpoint + "/v1/completions", json=payload) as response:
                    response.raise_for_status()
                    text = (await response.json())["choices"][0]["text"]
                predicted = final_number(text)
                expected = final_number(row["answer"].split("####")[-1])
                results.append({"index": index, "predicted": predicted, "expected": expected,
                                "correct": predicted == expected, "output": text})
        await asyncio.gather(*(one(i, row) for i, row in enumerate(rows)))
    results.sort(key=lambda item: item["index"])
    out = {"label": args.label, "dataset": "openai/gsm8k-main-test-first32",
           "exact_numeric_accuracy": sum(x["correct"] for x in results) / len(results),
           "correct": sum(x["correct"] for x in results), "total": len(results),
           "samples": results}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(out, indent=2)); tmp.replace(path)


if __name__ == "__main__":
    asyncio.run(main())

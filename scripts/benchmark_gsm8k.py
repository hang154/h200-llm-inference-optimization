#!/usr/bin/env python3
import argparse, asyncio, json, math, statistics, time
from pathlib import Path

import aiohttp
from datasets import load_dataset
from transformers import AutoTokenizer


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--label", required=True)
    ap.add_argument("--qps", type=float, default=4)
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = list(load_dataset("openai/gsm8k", "main", split="test"))[: args.requests]
    tok = AutoTokenizer.from_pretrained(args.model)
    timeout = aiohttp.ClientTimeout(total=1200)
    result = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(i, row):
            await asyncio.sleep(i / args.qps)
            prompt = "Solve carefully and end with 'Final answer:'.\n\n" + row["question"]
            payload = {"model": args.model, "prompt": prompt, "max_tokens": 192,
                       "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True}}
            start=time.perf_counter(); first=None; text=""; usage=None
            async with session.post(args.endpoint + "/v1/completions", json=payload) as r:
                r.raise_for_status()
                async for raw in r.content:
                    for line in raw.decode(errors="ignore").splitlines():
                        if not line.startswith("data: ") or line == "data: [DONE]": continue
                        obj=json.loads(line[6:]); choices=obj.get("choices") or []
                        chunk=choices[0].get("text","") if choices else ""
                        if chunk and first is None: first=time.perf_counter()
                        text += chunk; usage=obj.get("usage") or usage
            end=time.perf_counter(); n=(usage or {}).get("completion_tokens") or len(tok.encode(text,add_special_tokens=False))
            result.append({"request":i,"ttft_s":(first or end)-start,
                           "tpot_s":(end-(first or end))/max(1,n-1),"total_s":end-start,"output_tokens":n})
        wall_start=time.perf_counter(); await asyncio.gather(*(one(i,r) for i,r in enumerate(rows))); wall=time.perf_counter()-wall_start
    def pctl(vals,p):
        vals=sorted(vals); return vals[min(len(vals)-1,math.ceil(p*len(vals))-1)]
    out={"label":args.label,"workload":"openai/gsm8k-main-test-first32","qps":args.qps,"requests":len(result),
         "ttft_p50_s":statistics.median(x["ttft_s"] for x in result),"ttft_p95_s":pctl([x["ttft_s"] for x in result],.95),
         "tpot_p50_s":statistics.median(x["tpot_s"] for x in result),"tpot_p95_s":pctl([x["tpot_s"] for x in result],.95),
         "output_tok_s":sum(x["output_tokens"] for x in result)/wall,"wall_s":wall,"requests_detail":sorted(result,key=lambda x:x["request"])}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out,indent=2)); tmp.replace(path)


if __name__ == "__main__": asyncio.run(main())

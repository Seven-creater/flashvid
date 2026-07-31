# FlashVID server validation

Validation date: 2026-07-31, server `10.1.4.86`, eight NVIDIA A6000 GPUs.

## Scope

This run validates the before-LLM vision path only: last-layer QKV attention is
scored per frame, ADTS selects anchors, and TSTM/DPC-kNN merges redundant
spatiotemporal tokens. No Inner-LLM pruning is enabled. The model is the local
ModelScope copy at `models/Qwen3.5-4B` and the service is vLLM 0.25.1.

## Functional checks

All requests below returned HTTP 200 with zero failed requests:

| Input | Result |
| --- | --- |
| text-only | passed |
| one image (`method.png`) | passed |
| one video (`Qgr4dcsY-60.mp4`) | passed |
| two videos in one request | passed |
| dummy-weight plugin startup | passed |
| real-weight plugin startup | passed |

For the same sampled video, API prompt token counts were:

| Vision retention ratio | Prompt tokens |
| ---: | ---: |
| 1.00 | 11,671 |
| 0.50 | 6,163 |
| 0.25 | 3,409 |
| 0.10 | 1,756 |

The processor samples only a small number of temporal positions for this clip.
The implementation keeps at least one spatial token group per sampled time
position, so the 0.10 case is a safe lower-bound case rather than exactly 10%
of this clip's tokens. Longer videos use the requested ratio more closely.

The ratio-1.0 plugin response matched the native Qwen3.5 response for the
tested greedy video request (same prompt token count and generated content).

## Throughput checks

The benchmark used the same video URL, `temperature=0`, 32 output tokens, and
16 concurrent clients where shown. Repeated identical media benefits from
vLLM's multimodal cache; these numbers are service throughput, not an isolated
vision-encoder microbenchmark.

| Data parallel replicas | Requests | Concurrency | Successful | Requests/s | Output tok/s | TTFT p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 8 | 16 | 1.81 | 57.95 | 2.59 s |
| 2 | 16 | 8 | 16 | 1.65 | 52.74 | 3.83 s |
| 4 | 32 | 16 | 32 | 2.63 | 84.24 | 4.86 s |
| 8 | 32 | 16 | 32 | **3.29** | **105.15** | 2.95 s |

The fastest tested deployment is therefore `DP=8`, `TP=1`, one 4B replica per
A6000. At startup each replica used approximately 8.6 GiB for weights and the
DP8 service reached approximately 40.8 GiB per GPU including runtime/cache
allocations. All DP2/4/8 benchmark requests completed without OOM or failures.

## Reproduce

```bash
cd /data02/usr/wangqihao/Demo/test/flashvid
bash scripts/serve_dp8.sh 0.10
python scripts/smoke_openai.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3.5-4b-flashvid \
  --video-url http://127.0.0.1:8765/Qgr4dcsY-60.mp4
python scripts/benchmark_openai.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3.5-4b-flashvid \
  --video-url http://127.0.0.1:8765/Qgr4dcsY-60.mp4 \
  --requests 32 --concurrency 16 --max-tokens 32
```

The source paper is [FlashVID](https://arxiv.org/abs/2602.08024), and the
reference implementation is [Fanziyang-v/FlashVID](https://github.com/Fanziyang-v/FlashVID).

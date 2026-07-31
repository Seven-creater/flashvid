from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.5-4b-flashvid")
    parser.add_argument("--video-url", required=True)
    parser.add_argument("--prompt", default="Describe this video concisely.")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": args.video_url}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
    }
    request = urllib.request.Request(
        f"{args.base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


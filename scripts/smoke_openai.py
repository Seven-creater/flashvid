from __future__ import annotations

import argparse
import json
import urllib.request
from urllib.parse import urlsplit


def validate_media_reference(value: str, *, allow_remote: bool) -> None:
    if urlsplit(value).scheme.lower() in {"http", "https"} and not allow_remote:
        raise ValueError(
            "remote HTTP(S) media is disabled by default; use a local file URI "
            "or pass --allow-remote-media explicitly"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.5-4b-flashvid")
    media = parser.add_mutually_exclusive_group()
    media.add_argument(
        "--video-url",
        action="append",
        help="Video URL; repeat the flag to exercise multi-video requests.",
    )
    media.add_argument("--image-url")
    parser.add_argument("--allow-remote-media", action="store_true")
    parser.add_argument("--prompt", default="Describe this video concisely.")
    args = parser.parse_args()

    try:
        media_references = [
            *(args.video_url or []),
            *([args.image_url] if args.image_url else []),
        ]
        for value in media_references:
            validate_media_reference(value, allow_remote=args.allow_remote_media)
    except ValueError as exc:
        parser.error(str(exc))

    content = []
    if args.video_url:
        content.extend(
            {"type": "video_url", "video_url": {"url": url}}
            for url in args.video_url
        )
    elif args.image_url:
        content.append(
            {"type": "image_url", "image_url": {"url": args.image_url}}
        )
    content.append({"type": "text", "text": args.prompt})

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": content,
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

#!/usr/bin/env python3
"""Extract selected ZIP members through HTTP ranges from hf-mirror only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile


class HttpRangeReader(io.RawIOBase):
    def __init__(self, url: str, *, block_size: int = 64 * 1024 * 1024) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "hf-mirror.com":
            raise ValueError("archive URL must use https://hf-mirror.com")
        request = Request(url, method="HEAD", headers={"User-Agent": "flashvid-recovery/1"})
        with urlopen(request, timeout=60) as response:
            self.url = response.geturl()
            self.length = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
        self.block_size = block_size
        self.position = 0
        self._cache_start = 0
        self._cache = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = min(position, self.length)
        return self.position

    def _fetch(self, start: int, size: int) -> bytes:
        end = min(self.length, start + max(size, self.block_size)) - 1
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                request = Request(
                    self.url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "User-Agent": "flashvid-recovery/1",
                    },
                )
                with urlopen(request, timeout=180) as response:
                    if getattr(response, "status", None) != 206:
                        raise RuntimeError("mirror/CDN ignored the HTTP Range request")
                    content_range = str(response.headers.get("Content-Range") or "")
                    if not content_range.startswith(f"bytes {start}-"):
                        raise RuntimeError(f"unexpected Content-Range: {content_range}")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise RuntimeError("short HTTP Range response")
                return data
            except Exception as error:  # bounded network retry
                last_error = error
                if attempt < 2:
                    time.sleep(2**attempt)
        assert last_error is not None
        raise last_error

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length or size == 0:
            return b""
        if size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            cache_end = self._cache_start + len(self._cache)
            if not (self._cache_start <= self.position < cache_end):
                self._cache_start = self.position
                self._cache = self._fetch(self.position, remaining)
                cache_end = self._cache_start + len(self._cache)
            take = min(remaining, cache_end - self.position)
            offset = self.position - self._cache_start
            chunks.append(self._cache[offset : offset + take])
            self.position += take
            remaining -= take
        return b"".join(chunks)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_member(
    url: str,
    member_name: str,
    output: Path,
    *,
    ffprobe: str = "ffprobe",
) -> dict[str, object]:
    reader = HttpRangeReader(url)
    with zipfile.ZipFile(reader) as archive:
        matches = [
            info
            for info in archive.infolist()
            if info.filename == member_name or Path(info.filename).name == member_name
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one ZIP member named {member_name}, found {len(matches)}")
        info = matches[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and output.stat().st_size == info.file_size:
            temporary = None
        else:
            with tempfile.NamedTemporaryFile(
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".partial",
                delete=False,
            ) as target:
                temporary = Path(target.name)
                with archive.open(info) as source:
                    while block := source.read(8 * 1024 * 1024):
                        target.write(block)
                target.flush()
                os.fsync(target.fileno())
            if temporary.stat().st_size != info.file_size:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("extracted member size differs from ZIP metadata")
            os.replace(temporary, output)
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(completed.stdout.strip())
    if duration <= 0:
        raise RuntimeError("ffprobe returned a non-positive duration")
    return {
        "status": "passed",
        "archive_url": url,
        "resolved_url_host": urlparse(reader.url).hostname,
        "archive_bytes": reader.length,
        "archive_etag": reader.etag,
        "member": info.filename,
        "member_crc32": f"{info.CRC:08x}",
        "member_bytes": info.file_size,
        "compressed_bytes": info.compress_size,
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "duration_s": duration,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    try:
        payload = extract_member(
            args.url,
            args.member,
            args.output,
            ffprobe=args.ffprobe,
        )
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        if args.report.exists() and args.report.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to overwrite changed report: {args.report}")
        args.report.write_text(text, encoding="utf-8")
        print(text, end="")
        return 0
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

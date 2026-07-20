"""Download official C2Seg-AB full-scene MAT files from Google Drive."""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests


FILES = [
    ("augsburg_multimodal.mat", "1y0TqnKb2xxiiJy838z-bapqy2VOaTlrH", 475149799),
    ("berlin_multimodal.mat", "1onnRf22V__nmdKZc-RWqe26RC8Z8Gw18", 1404259213),
]


def extract_download_form(text: str):
    action_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', text)
    if not action_match:
        action_match = re.search(r'<form[^>]+action="([^"]*download[^"]*)"', text)
    if not action_match:
        return None
    action = html.unescape(action_match.group(1))
    params = {}
    for name, value in re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', text):
        params[html.unescape(name)] = html.unescape(value)
    return action, params


def confirm_url(resp: requests.Response, file_id: str):
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            return "https://drive.google.com/uc", {
                "export": "download",
                "id": file_id,
                "confirm": value,
            }

    form = extract_download_form(resp.text)
    if form:
        return form

    match = re.search(r'href="([^"]*confirm=[^"]+)"', resp.text)
    if match:
        href = html.unescape(match.group(1)).replace("&amp;", "&")
        if href.startswith("/"):
            href = "https://drive.google.com" + href
        parsed = urlparse(href)
        return parsed.scheme + "://" + parsed.netloc + parsed.path, {
            key: values[-1] for key, values in parse_qs(parsed.query).items()
        }

    return None


def download_one(session: requests.Session, dest_dir: Path, name: str, file_id: str, expected_size: int) -> None:
    final_path = dest_dir / name
    temp_path = dest_dir / f"{name}.part"

    if final_path.exists() and final_path.stat().st_size == expected_size:
        print(f"skip complete {name} ({expected_size} bytes)")
        return

    url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}
    resp = session.get(url, params=params, stream=True, timeout=60)
    resp.raise_for_status()
    if "text/html" in resp.headers.get("content-type", ""):
        confirmed = confirm_url(resp, file_id)
        if not confirmed:
            raise RuntimeError(f"Could not find confirmation token for {name}")
        url, params = confirmed
        resp.close()
        resp = session.get(url, params=params, stream=True, timeout=60)
        resp.raise_for_status()

    written = 0
    last_log = time.time()
    with temp_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            fh.write(chunk)
            written += len(chunk)
            now = time.time()
            if now - last_log >= 30:
                pct = written / expected_size * 100
                print(f"{name}: {written}/{expected_size} bytes ({pct:.1f}%)", flush=True)
                last_log = now

    actual = temp_path.stat().st_size
    if actual != expected_size:
        raise RuntimeError(f"Size mismatch for {name}: got {actual}, expected {expected_size}")
    os.replace(temp_path, final_path)
    print(f"downloaded {name} ({actual} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest_dir", type=Path)
    args = parser.parse_args()
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        for name, file_id, size in FILES:
            download_one(session, args.dest_dir, name, file_id, size)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

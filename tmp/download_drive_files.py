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
    ("beijing_label.mat", "1qPsyIHYvVISioFChm9FWVRTRJcNFe9hg", 6529937),
    ("beijing.mat", "12vLNQNcsMDpGds8RaGXJ5kB9_v3J-_iT", 6695252038),
    ("wuhan_label.mat", "15xJq_uGOYHVp04lSP9OTMQnyV_L1qmj6", 5022446),
    ("wuhan.mat", "1fzPUaPjEspbKrnd5ckIOLCrvcfD-tySQ", 3100719518),
]


def _extract_download_form(text):
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


def _confirm_url_from_response(resp, file_id):
    token = None
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token:
        return "https://drive.google.com/uc", {
            "export": "download",
            "id": file_id,
            "confirm": token,
        }

    form = _extract_download_form(resp.text)
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


def download_file(session, name, file_id, expected_size, dest_dir):
    final_path = dest_dir / name
    temp_path = dest_dir / (name + ".part")

    if final_path.exists() and final_path.stat().st_size == expected_size:
        print(f"skip complete {name} ({expected_size} bytes)")
        return

    if final_path.exists():
        print(f"existing size mismatch, re-downloading {name}: {final_path.stat().st_size} != {expected_size}")

    url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}
    resp = session.get(url, params=params, stream=True, timeout=60)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type:
        confirmed = _confirm_url_from_response(resp, file_id)
        if not confirmed:
            raise RuntimeError(f"Could not find Google Drive confirmation token for {name}")
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
                pct = written / expected_size * 100 if expected_size else 0
                print(f"{name}: {written}/{expected_size} bytes ({pct:.1f}%)", flush=True)
                last_log = now

    actual = temp_path.stat().st_size
    if actual != expected_size:
        raise RuntimeError(f"Size mismatch for {name}: got {actual}, expected {expected_size}")

    os.replace(temp_path, final_path)
    print(f"downloaded {name} ({actual} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dest_dir", type=Path)
    args = parser.parse_args()
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        for name, file_id, size in FILES:
            download_file(session, name, file_id, size, args.dest_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

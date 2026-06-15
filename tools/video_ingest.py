"""Offline batch ingest: YouTube channel -> vidsum summaries -> extracted
scenarios -> append-only corpus. Mirrors tools/build_universe.py (a heavy,
manual, resumable step). The page never runs this; it only reads the corpus.

  .venv/bin/python -m tools.video_ingest --limit 10
  .venv/bin/python -m tools.video_ingest --since 20260101 --retry-failed

Needs: ollama running (the extract model) + the vidsum venv (transcription).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

# ── Config (fill CHANNEL_URL before first run) ────────────────────────────────
CHANNEL_URL = "https://www.youtube.com/@AndreiJikh/videos"  # e.g. "https://www.youtube.com/@AndreiJikh/videos"
VIDSUM_PY = "/Users/cxmc/code/vidsum/.venv/bin/python"
EXTRACT_MODEL = "gemma4-abliterated:Q4_K_M"
EXTRACT_TEMPERATURE = 0.1

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "local" / "scenarios" / "corpus.jsonl"


# ── Corpus helpers ────────────────────────────────────────────────────────────

def processed_ids(corpus_path: Path, retry_failed: bool) -> set:
    """IDs already in the corpus that should be skipped. With retry_failed,
    records whose status is 'failed' are excluded so they get another attempt."""
    done: set = set()
    if not Path(corpus_path).exists():
        return done
    for line in Path(corpus_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "failed" and retry_failed:
            continue
        done.add(rec.get("video_id"))
    return done


def append_record(corpus_path: Path, rec: dict) -> None:
    p = Path(corpus_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_summary(summaries_dir, base: dict, summary: str) -> str:
    """Dump one raw vidsum summary as markdown into a folder, named to sort by
    upload date. Returns the path written."""
    d = Path(summaries_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{base.get('upload_date') or 'NA'}_{base['video_id']}.md"
    header = (f"# {base.get('title') or base['video_id']}\n\n"
              f"- video: {base.get('url')}\n"
              f"- uploaded: {base.get('upload_date')}\n\n---\n\n")
    path.write_text(header + (summary or ""), encoding="utf-8")
    return str(path)


# ── Pure orchestration (I/O steps injected → fully testable) ──────────────────

def ingest(channel_url: str, corpus_path, *, list_videos, run_vidsum, extract,
           limit: int | None = None, since: str | None = None,
           retry_failed: bool = False, now_iso: str | None = None,
           summaries_dir=None) -> dict:
    """List channel videos, skip ones already in the corpus, and for each new
    one run vidsum + extraction, appending a record. A single video's failure is
    recorded and the batch continues. If summaries_dir is set, each raw summary is
    also dumped there as markdown."""
    now_iso = now_iso or datetime.now().isoformat(timespec="seconds")
    done = processed_ids(corpus_path, retry_failed)
    videos = list_videos(channel_url, limit=limit, since=since)

    results = {"processed": [], "failed": [], "skipped": [], "summaries": []}
    for v in videos:
        vid = v["video_id"]
        if vid in done:
            results["skipped"].append(vid)
            continue
        base = {"video_id": vid, "title": v.get("title"), "url": v.get("url"),
                "upload_date": v.get("upload_date"), "ingested_at": now_iso}
        try:
            summary = run_vidsum(v)
            if summaries_dir:
                results["summaries"].append(write_summary(summaries_dir, base, summary))
            ext = extract(summary, v.get("upload_date"))
            status = "extract_failed" if ext.get("extraction_failed") else "ok"
            append_record(corpus_path, {**base, "status": status, "summary": summary,
                                        "assets": ext.get("assets", []),
                                        "macro_theses": ext.get("macro_theses", [])})
            results["processed"].append(vid)
        except Exception as e:  # one bad video must not abort the batch
            append_record(corpus_path, {**base, "status": "failed", "error": str(e),
                                        "summary": "", "assets": [], "macro_theses": []})
            results["failed"].append(vid)
    return results


# ── Production I/O wrappers ────────────────────────────────────────────────────

def _yt_dlp_bin() -> str:
    return shutil.which("yt-dlp") or "/opt/homebrew/bin/yt-dlp"


def list_channel_videos(channel_url: str, *, limit: int | None = None,
                        since: str | None = None, dateafter: str | None = None) -> list[dict]:
    """Recent uploads via yt-dlp (metadata only, no download). --skip-download
    keeps upload_date populated (unlike --flat-playlist). --dateafter lets yt-dlp
    skip older uploads server-side; --playlist-end is a safety cap."""
    fmt = "%(id)s\t%(upload_date)s\t%(title)s\t%(webpage_url)s"
    cmd = [_yt_dlp_bin(), "--skip-download", "--ignore-errors", "--print", fmt]
    if dateafter:
        cmd += ["--dateafter", dateafter]
    if limit:
        cmd += ["--playlist-end", str(limit)]
    cmd.append(channel_url)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        vid, up, title, url = parts[0], parts[1], parts[2], parts[3]
        if since and up not in ("NA", "") and up < since:
            continue
        out.append({"video_id": vid, "upload_date": None if up in ("NA", "") else up,
                    "title": title, "url": url})
    return out


def clean_summary(text: str) -> str:
    """Strip gemma 'harmony' control tokens that leak into the summary, e.g. a
    leading `<|channel>thought ... <channel|>` reasoning preamble plus any stray
    `<|token|>` markers, leaving just the readable summary."""
    t = text.strip()
    m = re.match(r"\s*<\|?channel[^>]*>.*?<channel\|>\s*", t, re.DOTALL)
    if m:
        t = t[m.end():]
    t = re.sub(r"<\|?[a-zA-Z_]+\|?>", "", t)
    return t.strip()


def run_vidsum(video: dict) -> str:
    """Run the vidsum CLI (its own venv) on one video URL; return the summary md."""
    tmp = tempfile.mkdtemp(prefix="vidsum_ingest_")
    try:
        cmd = [VIDSUM_PY, "-m", "vidsum", video["url"], "--output-dir", tmp]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        mds = glob.glob(os.path.join(tmp, "*.md"))
        if not mds:
            raise RuntimeError(
                f"vidsum produced no summary: {proc.stderr.strip()[:300] or proc.stdout.strip()[:300]}")
        return clean_summary(Path(mds[0]).read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _default_extract(summary: str, upload_date: str | None) -> dict:
    from tools import ollama_client
    from tools.scenario_extract import extract_scenarios
    gen = lambda p: ollama_client.generate(p, EXTRACT_MODEL, temperature=EXTRACT_TEMPERATURE)
    return extract_scenarios(summary, gen, upload_date=upload_date)


def main():
    ap = argparse.ArgumentParser(description="Ingest market-analysis videos into the scenario corpus.")
    ap.add_argument("--channel", default=CHANNEL_URL, help="channel/playlist URL")
    ap.add_argument("--limit", type=int, default=10, help="safety cap: most-recent N uploads to consider")
    ap.add_argument("--since", default=None, help="only uploads on/after YYYYMMDD (post-filter)")
    ap.add_argument("--dateafter", default=None, help="yt-dlp --dateafter YYYYMMDD (server-side)")
    ap.add_argument("--summaries-dir", default=None, help="also dump each raw summary .md into this folder")
    ap.add_argument("--retry-failed", action="store_true", help="re-attempt previously failed videos")
    args = ap.parse_args()

    if not args.channel:
        ap.error("no channel URL — set CHANNEL_URL in tools/video_ingest.py or pass --channel")

    # thread --dateafter into the lister without changing ingest's injected-fn contract
    lister = lambda url, *, limit, since: list_channel_videos(
        url, limit=limit, since=since, dateafter=args.dateafter)

    print(f"listing up to {args.limit} videos from {args.channel} …")
    res = ingest(args.channel, CORPUS_PATH,
                 list_videos=lister, run_vidsum=run_vidsum,
                 extract=_default_extract, limit=args.limit, since=args.since,
                 retry_failed=args.retry_failed, summaries_dir=args.summaries_dir)
    print(f"processed={len(res['processed'])} failed={len(res['failed'])} "
          f"skipped={len(res['skipped'])}")
    if res["failed"]:
        print("  failed:", ", ".join(res["failed"]))
    if args.summaries_dir:
        print(f"summaries → {args.summaries_dir} ({len(res['summaries'])} files)")
    print(f"corpus → {CORPUS_PATH}")


if __name__ == "__main__":
    main()

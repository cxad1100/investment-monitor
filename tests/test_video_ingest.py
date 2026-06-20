"""No-network tests for the ingest orchestrator.

The three heavy I/O steps (channel listing, vidsum, Ollama extraction) are
injected, so dedup / failure-isolation / retry are tested deterministically.
"""
import json

from tools.video_ingest import ingest, clean_summary


def _read_ids(path):
    return [json.loads(l)["video_id"] for l in path.read_text().splitlines()]


def _ok_extract(summary, upload_date):
    return {"assets": [{"name": "A"}], "macro_theses": [], "extraction_failed": False}


def test_ingest_dedups_existing_and_appends_new(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"video_id": "X", "status": "ok"}) + "\n")

    videos = [
        {"video_id": "X", "url": "uX", "upload_date": "20251201", "title": "tx"},
        {"video_id": "Y", "url": "uY", "upload_date": "20260101", "title": "ty"},
    ]
    run_calls = []

    def run_vidsum(v):
        run_calls.append(v["video_id"])
        return "summary for " + v["video_id"]

    res = ingest("chan", corpus,
                 list_videos=lambda url, limit, since: videos,
                 run_vidsum=run_vidsum, extract=_ok_extract, now_iso="t")

    assert res["processed"] == ["Y"]
    assert "X" in res["skipped"]
    assert run_calls == ["Y"]                 # X never re-transcribed
    assert _read_ids(corpus) == ["X", "Y"]    # append-only

    yrec = json.loads(corpus.read_text().splitlines()[1])
    assert yrec["status"] == "ok"
    assert yrec["summary"] == "summary for Y"
    assert yrec["assets"] == [{"name": "A"}]
    assert yrec["upload_date"] == "20260101"


def test_ingest_isolates_failure_and_continues(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    videos = [
        {"video_id": "BAD", "url": "u1", "upload_date": "20260101", "title": "b"},
        {"video_id": "OK", "url": "u2", "upload_date": "20260102", "title": "o"},
    ]

    def run_vidsum(v):
        if v["video_id"] == "BAD":
            raise RuntimeError("yt-dlp boom")
        return "good summary"

    res = ingest("chan", corpus,
                 list_videos=lambda url, limit, since: videos,
                 run_vidsum=run_vidsum, extract=_ok_extract, now_iso="t")

    assert res["failed"] == ["BAD"]
    assert res["processed"] == ["OK"]         # loop did not abort on BAD
    recs = {r["video_id"]: r for r in (json.loads(l) for l in corpus.read_text().splitlines())}
    assert recs["BAD"]["status"] == "failed"
    assert "boom" in recs["BAD"]["error"]
    assert recs["OK"]["status"] == "ok"


def test_ingest_extract_failure_marks_status(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    videos = [{"video_id": "Z", "url": "u", "upload_date": "20260101", "title": "z"}]

    def bad_extract(summary, upload_date):
        return {"assets": [], "macro_theses": [], "extraction_failed": True, "raw": "junk"}

    res = ingest("chan", corpus,
                 list_videos=lambda url, limit, since: videos,
                 run_vidsum=lambda v: "summary", extract=bad_extract, now_iso="t")

    assert res["processed"] == ["Z"]          # we got a summary; it's still ingested
    rec = json.loads(corpus.read_text())
    assert rec["status"] == "extract_failed"
    assert rec["summary"] == "summary"        # raw summary preserved for later re-extract


def test_ingest_retry_failed_flag(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"video_id": "BAD", "status": "failed"}) + "\n")
    videos = [{"video_id": "BAD", "url": "u", "upload_date": "20260101", "title": "b"}]

    common = dict(list_videos=lambda url, limit, since: videos,
                  run_vidsum=lambda v: "s", extract=_ok_extract, now_iso="t")

    skipped = ingest("chan", corpus, retry_failed=False, **common)
    assert skipped["processed"] == []         # failed record blocks re-run by default

    retried = ingest("chan", corpus, retry_failed=True, **common)
    assert retried["processed"] == ["BAD"]     # retry_failed re-attempts it


def test_ingest_writes_summaries_dir(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    sdir = tmp_path / "summaries"
    videos = [{"video_id": "V1", "url": "u", "upload_date": "20260101", "title": "Hello World"}]
    res = ingest("chan", corpus,
                 list_videos=lambda url, limit, since: videos,
                 run_vidsum=lambda v: "the summary body",
                 extract=_ok_extract, now_iso="t", summaries_dir=sdir)
    assert res["processed"] == ["V1"]
    files = list(sdir.glob("*.md"))
    assert len(files) == 1
    assert files[0].name == "20260101_V1.md"   # sorts by upload date
    text = files[0].read_text()
    assert "the summary body" in text
    assert "Hello World" in text                # header carries the title


def test_clean_summary_strips_harmony_preamble():
    raw = "<|channel>thought\nlet me think about this\n<channel|>\n### Summary\nGold up, dollar down."
    assert clean_summary(raw) == "### Summary\nGold up, dollar down."


def test_clean_summary_removes_stray_tokens():
    assert clean_summary("<|start|>Real content<|end|>") == "Real content"


def test_clean_summary_leaves_normal_text():
    assert clean_summary("Just a normal market summary.") == "Just a normal market summary."

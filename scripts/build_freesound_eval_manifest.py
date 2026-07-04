"""Build freesound.org query URLs for the real-audio evaluation corpus.

Emits the search-API URLs (with a token placeholder) that the user can fetch
to assemble an evaluation-only corpus of native high-sample-rate, CC0
recordings. Per the CAPB data policy, real audio is NOT used for training;
it serves as an out-of-distribution holdout for the flatness/gain/LB gates.

Usage:
    uv run python scripts/build_freesound_eval_manifest.py \
        --output reports/freesound_eval/query_manifest.json
    # Then fetch each URL with your API token substituted for {TOKEN}.

After downloading, verify each file header's sample rate (metadata can lie)
and re-run the gate evaluation with the real-audio corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

API_BASE = "https://freesound.org/apiv2/search/text/"
DEFAULT_QUERIES = (
    "music",
    "drums",
    "piano",
    "guitar",
    "orchestra",
    "vocal",
    "field-recording",
)
FILTER_EXPR = (
    'samplerate:[88200 TO 192000] type:("wav" OR "flac") '
    'duration:[10 TO 120] license:"Creative Commons 0"'
)
FIELDS = "id,name,samplerate,duration,license,download,username"


def main() -> None:
    """Write the query manifest JSON and print the URLs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/freesound_eval/query_manifest.json"),
    )
    parser.add_argument("--queries", nargs="*", default=list(DEFAULT_QUERIES))
    parser.add_argument("--page-size", type=int, default=30)
    args = parser.parse_args()

    entries = []
    for query in args.queries:
        params = {
            "query": query,
            "filter": FILTER_EXPR,
            "fields": FIELDS,
            "sort": "rating_desc",
            "page_size": str(args.page_size),
            "token": "{TOKEN}",
        }
        url = f"{API_BASE}?{urlencode(params)}"
        entries.append({"query": query, "url": url})
        print(url)

    payload = {
        "purpose": "evaluation-only real-audio corpus (no training use)",
        "license_filter": "CC0",
        "min_samplerate_hz": 88_200,
        "post_download_checks": [
            "verify header sample rate >= 88200 (metadata can lie)",
            "trim to 10-30 s excerpts, curate ~15 min total",
        ],
        "queries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

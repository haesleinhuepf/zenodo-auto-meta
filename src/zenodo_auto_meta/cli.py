"""Command-line interface for zenodo-auto-meta."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .llm import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MAX_EXAMPLES, DEFAULT_MODEL, predict_tags
from .zenodo import (
    ZENODO_API_URL,
    fetch_community_records,
    get_description,
    get_link,
    get_tags,
    separate_records,
    update_record_keywords,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zenodo-auto-meta",
        description=(
            "Auto-complete Zenodo record metadata using a local LLM.\n\n"
            "USAGE — two modes:\n"
            "  1. Predict tags:  zenodo-auto-meta <community> [<community> ...] [options]\n"
            "  2. Apply curated tags: zenodo-auto-meta <community> --update <json_file>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "community",
        nargs="+",
        help="One or more Zenodo community slugs (e.g. 'nfdi4bioimage')",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached records and re-fetch from Zenodo",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ZENODO_AUTO_META_MODEL", DEFAULT_MODEL),
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ZENODO_AUTO_META_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ZENODO_AUTO_META_API_KEY", DEFAULT_API_KEY),
        help="API key for the LLM service",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help=f"Maximum number of few-shot examples to send (default: {DEFAULT_MAX_EXAMPLES})",
    )
    parser.add_argument(
        "--output",
        default="proposed_tags.json",
        help=(
            "Output JSON file for proposed tags (default: proposed_tags.json). "
            "When multiple communities are given, the community slug is inserted "
            "before the extension, e.g. 'proposed_tags_<community>.json'."
        ),
    )
    parser.add_argument(
        "--update",
        metavar="JSON_FILE",
        help=(
            "Curated JSON file produced by a previous run.  "
            "When supplied, the tool updates Zenodo records instead of predicting tags."
        ),
    )
    parser.add_argument(
        "--zenodo-token",
        default=os.environ.get("ZENODO_TOKEN"),
        help="Zenodo API token (required for --update).  Can also be set via ZENODO_TOKEN env var.",
    )
    parser.add_argument(
        "--zenodo-url",
        default=os.environ.get("ZENODO_URL", ZENODO_API_URL),
        help=f"Zenodo API base URL (default: {ZENODO_API_URL})",
    )
    return parser


def _run_update(args: argparse.Namespace) -> int:
    """Apply curated tags from a JSON file to Zenodo records."""
    if not args.zenodo_token:
        print(
            "Error: a Zenodo API token is required to update records.\n"
            "Set the ZENODO_TOKEN environment variable or use --zenodo-token.",
            file=sys.stderr,
        )
        return 1

    with open(args.update) as fh:
        entries = json.load(fh)

    errors = 0
    for entry in entries:
        link = entry.get("link", "")
        proposed = entry.get("proposed_tags", [])
        if not link or not proposed:
            print(f"  Skipping entry with missing link or tags: {entry}", file=sys.stderr)
            continue
        print(f"Updating {link} with tags: {proposed}")
        try:
            update_record_keywords(link, proposed, args.zenodo_token, zenodo_url=args.zenodo_url)
            print("  → Done")
        except Exception as exc:  # noqa: BLE001
            print(f"  → Error: {exc}", file=sys.stderr)
            errors += 1

    if errors:
        print(f"\n{errors} record(s) could not be updated.", file=sys.stderr)
        return 1
    return 0


def _output_path_for(community: str, output: str, *, multiple: bool) -> str:
    """Return the output path for *community*, disambiguating when multiple
    communities share a single ``--output`` template."""
    if not multiple:
        return output
    base, ext = os.path.splitext(output)
    return f"{base}_{community}{ext or '.json'}"


def _run_predict_one(
    community: str,
    output: str,
    without_tags: list,
    examples: list[tuple[str, list[str]]],
    args: argparse.Namespace,
) -> int:
    """Predict tags for a single community's untagged records, save JSON."""
    print(f"\nCommunity '{community}':")
    print(f"  Records without tags: {len(without_tags)}")

    # Records to predict: have a description but no tags
    to_predict = [r for r in without_tags if get_description(r)]
    print(f"  Records to predict tags for: {len(to_predict)}")

    if not to_predict:
        print("  Nothing to do — all records with descriptions already have tags.")
        return 0

    results: list[dict] = []
    for i, record in enumerate(to_predict, start=1):
        desc = get_description(record)
        link = get_link(record)
        print(f"  [{i}/{len(to_predict)}] Predicting tags for {link} …", end=" ", flush=True)
        try:
            tags = predict_tags(
                desc,
                examples,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                max_examples=args.max_examples,
            )
            print(", ".join(tags))
            results.append({"link": link, "description": desc, "proposed_tags": tags})
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}", file=sys.stderr)

        with open(output, "w") as fh:
            json.dump(results, fh, indent=2)
    print(f"  Saved {len(results)} prediction(s) to '{output}'.")
    print("  Please review and curate the file, then run:")
    print(f"    zenodo-auto-meta {community} --update {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.update:
        return _run_update(args)

    multiple = len(args.community) > 1

    # Fetch records for every community first, then pool the tagged records
    # from all of them together as few-shot examples.
    without_tags_by_community: dict[str, list] = {}
    examples: list[tuple[str, list[str]]] = []
    for community in args.community:
        print(f"Fetching records for Zenodo community '{community}' …")
        records = fetch_community_records(
            community,
            force_refresh=args.force_refresh,
            zenodo_url=args.zenodo_url,
        )
        print(f"  Total records: {len(records)}")

        with_tags, without_tags = separate_records(records)
        print(f"  Records with tags:    {len(with_tags)}")
        print(f"  Records without tags: {len(without_tags)}")
        without_tags_by_community[community] = without_tags

        for record in with_tags:
            desc = get_description(record)
            tags = get_tags(record)
            if desc:
                examples.append((desc, tags))

    print(f"\nFew-shot examples pooled from all communities (tagged + described): {len(examples)}")

    exit_code = 0
    for community in args.community:
        output = _output_path_for(community, args.output, multiple=multiple)
        result = _run_predict_one(
            community, output, without_tags_by_community[community], examples, args
        )
        if result != 0:
            exit_code = result
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

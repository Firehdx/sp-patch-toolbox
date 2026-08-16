"""Small stable command line for discovery, integrity gates and legacy runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline.integrity import scan_manifest_strict
from .pipeline.manifest import load_manifest
from .profiles.reviewed_cases import reviewed_case_summary


def _integrity_command(args: argparse.Namespace) -> int:
    specs = load_manifest(args.manifest)
    report = scan_manifest_strict(specs, data_root=args.data_root, thumbnail_max_size=args.thumbnail_max_size)
    payload = report.to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if report.failed else 0


def _profiles_command(_: argparse.Namespace) -> int:
    print(json.dumps(reviewed_case_summary(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _segment_command(argv: list[str]) -> int:
    """Forward unchanged arguments to the validated compatibility CLI."""
    from .compat import legacy_cli

    # ``argparse.REMAINDER`` keeps the conventional separator.  Remove only
    # that leading separator so downstream flags (including ``--help``) retain
    # their normal meaning to the compatibility CLI.
    if argv and argv[0] == "--":
        argv = argv[1:]
    old_argv = sys.argv
    try:
        sys.argv = ["sppatch-segment", *argv]
        legacy_cli.main()
    finally:
        sys.argv = old_argv
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="sppatch", description="Spatial-proteomics patch toolbox")
    subparsers = parser.add_subparsers(dest="command", required=True)

    integrity = subparsers.add_parser("integrity", help="Strictly validate source images before patch generation")
    integrity.add_argument("--manifest", required=True)
    integrity.add_argument("--data-root", default=None)
    integrity.add_argument("--thumbnail-max-size", type=int, default=1600)
    integrity.add_argument("--out", default=None, help="Optional JSON report path")
    integrity.set_defaults(handler=_integrity_command)

    profiles = subparsers.add_parser("profiles", help="List reviewed special-case rule groups")
    profiles.set_defaults(handler=_profiles_command)

    segment = subparsers.add_parser("segment", help="Run the compatible segmentation/coordinate pipeline")
    segment.add_argument("args", nargs=argparse.REMAINDER, help="Arguments accepted by sppatch-segment")
    segment.set_defaults(handler=lambda parsed: _segment_command(parsed.args))

    parsed = parser.parse_args(argv)
    return int(parsed.handler(parsed))


if __name__ == "__main__":
    raise SystemExit(main())

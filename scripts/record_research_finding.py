#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


STATUS_ORDER = {
    "cleared": 0,
    "partially_causal": 1,
    "open": 2,
    "confounded": 3,
    "falsified": 4,
    "note": 5,
}


@dataclass(frozen=True)
class FindingRecord:
    timestamp_utc: str
    claim_id: str
    topic: str
    status: str
    question: str
    conclusion: str
    summary: str
    branch: str
    commit: str
    note_path: str
    artifact_paths: list[str]
    next_checks: list[str]
    tags: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _findings_path(repo_root: Path) -> Path:
    return repo_root / "docs" / "research" / "findings.jsonl"


def _current_map_path(repo_root: Path) -> Path:
    return repo_root / "docs" / "research" / "current_hypotheses.md"


def _load_records(path: Path) -> list[FindingRecord]:
    if not path.exists():
        return []
    records: list[FindingRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(FindingRecord(**json.loads(line)))
    return records


def _write_current_map(path: Path, records: list[FindingRecord]) -> None:
    latest_by_claim: dict[str, FindingRecord] = {}
    for record in records:
        latest_by_claim[record.claim_id] = record

    grouped: dict[str, list[FindingRecord]] = defaultdict(list)
    for record in latest_by_claim.values():
        grouped[record.status].append(record)

    for status_records in grouped.values():
        status_records.sort(key=lambda r: (r.topic, r.claim_id))

    lines = [
        "# Current Hypotheses",
        "",
        "This file is generated from `docs/research/findings.jsonl`.",
        "It is the current research surface for scaling decisions, not an append-only history.",
        "",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## How To Read This",
        "",
        "- `cleared`: a hypothesis has strong supporting evidence and can be treated as established locally",
        "- `partially_causal`: the factor is real and mattered, but does not fully explain the observed outcome",
        "- `open`: still a live question for the next experiment",
        "- `confounded`: result was informative, but interpretation depended on a known flaw or missing control",
        "- `falsified`: do not keep spending on this version of the claim without a new mechanism",
        "",
    ]

    sections = [
        ("cleared", "Cleared"),
        ("partially_causal", "Partially Causal"),
        ("open", "Open"),
        ("confounded", "Confounded"),
        ("falsified", "Falsified"),
        ("note", "Notes"),
    ]
    for status, title in sections:
        entries = grouped.get(status, [])
        if not entries:
            continue
        lines.extend([f"## {title}", ""])
        lines.append("| Claim ID | Topic | Conclusion | Branch | Note |")
        lines.append("|---|---|---|---|---|")
        for record in entries:
            note_label = Path(record.note_path).name if record.note_path else ""
            lines.append(
                f"| `{record.claim_id}` | `{record.topic}` | {record.conclusion} | `{record.branch}` | `{note_label}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Recent Findings",
            "",
            "| Timestamp | Claim ID | Status | Summary |",
            "|---|---|---|---|",
        ]
    )
    for record in sorted(records, key=lambda r: r.timestamp_utc, reverse=True)[:12]:
        lines.append(
            f"| `{record.timestamp_utc}` | `{record.claim_id}` | `{record.status}` | {record.summary} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_record(path: Path, record: FindingRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record a durable research finding for GraphDrone.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--claim-id", required=True)
    record.add_argument("--topic", required=True)
    record.add_argument(
        "--status",
        required=True,
        choices=tuple(STATUS_ORDER.keys()),
    )
    record.add_argument("--question", required=True)
    record.add_argument("--conclusion", required=True)
    record.add_argument("--summary", required=True)
    record.add_argument("--branch", required=True)
    record.add_argument("--commit", required=True)
    record.add_argument("--note-path", required=True)
    record.add_argument("--artifact-path", action="append", default=[])
    record.add_argument("--next-check", action="append", default=[])
    record.add_argument("--tag", action="append", default=[])

    subparsers.add_parser("rebuild")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    repo_root = _repo_root()
    findings_path = _findings_path(repo_root)
    current_map_path = _current_map_path(repo_root)

    if args.command == "record":
        record = FindingRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            claim_id=args.claim_id,
            topic=args.topic,
            status=args.status,
            question=args.question,
            conclusion=args.conclusion,
            summary=args.summary,
            branch=args.branch,
            commit=args.commit,
            note_path=args.note_path,
            artifact_paths=args.artifact_path,
            next_checks=args.next_check,
            tags=args.tag,
        )
        _append_record(findings_path, record)

    records = _load_records(findings_path)
    records.sort(key=lambda r: (r.timestamp_utc, STATUS_ORDER.get(r.status, 999)))
    _write_current_map(current_map_path, records)
    print(current_map_path)


if __name__ == "__main__":
    main()

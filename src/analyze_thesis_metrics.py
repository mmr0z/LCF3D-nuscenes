#!/usr/bin/env python3
"""Convert LCF3D evaluation output into thesis-friendly tables.

The script accepts official nuScenes ``metrics.json`` files, MMEngine JSONL
metric logs, regular JSON metric dictionaries, or directories containing any
of those files.  Multiple inputs can be supplied for an ablation comparison.
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


COCO_OVERALL_KEYS = {
    "bbox_mAP": "AP",
    "bbox_mAP_50": "AP50",
    "bbox_mAP_75": "AP75",
    "bbox_mAP_s": "AP_small",
    "bbox_mAP_m": "AP_medium",
    "bbox_mAP_l": "AP_large",
    "bbox_AR@100": "AR100",
    "bbox_AR_s@1000": "AR_small",
    "bbox_AR_m@1000": "AR_medium",
    "bbox_AR_l@1000": "AR_large",
}
NUSC_ERRORS = {
    "trans_err": "mATE",
    "scale_err": "mASE",
    "orient_err": "mAOE",
    "vel_err": "mAVE",
    "attr_err": "mAAE",
}
SUMMARY_FIELDS = [
    "run",
    "protocol",
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
    "NDS",
    "mATE",
    "mASE",
    "mAOE",
    "mAVE",
    "mAAE",
    "latency_ms",
    "throughput_per_s",
    "step",
    "source",
]
CLASS_FIELDS = [
    "run",
    "protocol",
    "class",
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AP_0.5m",
    "AP_1.0m",
    "AP_2.0m",
    "AP_4.0m",
    "ATE",
    "ASE",
    "AOE",
    "AVE",
    "AAE",
]


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _find_key(record: Dict[str, Any], suffix: str) -> Any:
    if suffix in record:
        return record[suffix]
    matches = [value for key, value in record.items() if key.endswith("/" + suffix)]
    return matches[-1] if matches else None


def _has_metrics(record: Dict[str, Any]) -> bool:
    keys = record.keys()
    return (
        "mean_ap" in keys
        or "nd_score" in keys
        or any("bbox_mAP" in key for key in keys)
        or any(key.endswith("/NDS") for key in keys)
        or any("_AP_dist_" in key for key in keys)
    )


def _read_json_records(path: Path) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        records: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _parse_coco_tables(log_path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """Read the last MMDetection classwise AP table from a text log."""
    rows: Dict[str, Dict[str, Optional[float]]] = {}
    headers: Optional[List[str]] = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == "category":
            headers = cells
            rows = {}
            continue
        if headers is None or len(cells) != len(headers):
            continue
        values = [_number(value) for value in cells[1:]]
        if all(value is None for value in values):
            continue
        rows[cells[0]] = dict(zip(headers[1:], values))
    return rows


def _candidate_sources(path: Path) -> Tuple[List[Path], List[Path]]:
    if path.is_file():
        if path.suffix == ".log":
            return [], [path]
        return [path], []
    if not path.is_dir():
        raise FileNotFoundError(path)

    json_candidates: List[Path] = []
    for candidate in path.rglob("*.json"):
        if candidate.name.endswith(".bbox.json") or candidate.name == "results_nusc.json":
            continue
        json_candidates.append(candidate)
    log_candidates = list(path.rglob("*.log"))
    return (
        sorted(json_candidates, key=lambda item: item.stat().st_mtime),
        sorted(log_candidates, key=lambda item: item.stat().st_mtime),
    )


def _select_record(paths: Iterable[Path]) -> Tuple[Dict[str, Any], Optional[Path]]:
    selected: Dict[str, Any] = {}
    source: Optional[Path] = None
    for path in paths:
        try:
            records = _read_json_records(path)
        except (OSError, UnicodeDecodeError):
            continue
        for record in records:
            if _has_metrics(record):
                selected = record
                source = path
    return selected, source


def _parse_official_nuscenes(
    record: Dict[str, Any], run: str, source: Path
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary = {field: None for field in SUMMARY_FIELDS}
    summary.update(
        run=run,
        protocol="nuScenes-3D",
        AP=_number(record.get("mean_ap")),
        NDS=_number(record.get("nd_score")),
        source=str(source),
    )
    for raw_name, output_name in NUSC_ERRORS.items():
        summary[output_name] = _number(record.get("tp_errors", {}).get(raw_name))
    runtime = record.get("lcf3d_runtime", {})
    summary["latency_ms"] = _number(runtime.get("mean_seconds_per_sample"))
    if summary["latency_ms"] is not None:
        summary["latency_ms"] *= 1000.0
    summary["throughput_per_s"] = _number(runtime.get("samples_per_second"))

    class_names = list(record.get("mean_dist_aps", {}).keys())
    class_names += [
        name for name in record.get("label_aps", {}).keys() if name not in class_names
    ]
    per_class: List[Dict[str, Any]] = []
    for name in class_names:
        row = {field: None for field in CLASS_FIELDS}
        row.update(
            run=run,
            protocol="nuScenes-3D",
            **{"class": name},
            AP=_number(record.get("mean_dist_aps", {}).get(name)),
        )
        distance_aps = record.get("label_aps", {}).get(name, {})
        for distance in ("0.5", "1.0", "2.0", "4.0"):
            row[f"AP_{distance}m"] = _number(distance_aps.get(distance))
        tp_errors = record.get("label_tp_errors", {}).get(name, {})
        for raw_name, output_name in {
            "trans_err": "ATE",
            "scale_err": "ASE",
            "orient_err": "AOE",
            "vel_err": "AVE",
            "attr_err": "AAE",
        }.items():
            row[output_name] = _number(tp_errors.get(raw_name))
        per_class.append(row)
    return summary, per_class


def _parse_flat_nuscenes(
    record: Dict[str, Any], run: str, source: Path
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary = {field: None for field in SUMMARY_FIELDS}
    summary.update(
        run=run,
        protocol="nuScenes-3D",
        AP=_number(_find_key(record, "mAP")),
        NDS=_number(_find_key(record, "NDS")),
        step=record.get("step"),
        source=str(source),
    )
    for output_name in ("mATE", "mASE", "mAOE", "mAVE", "mAAE"):
        summary[output_name] = _number(_find_key(record, output_name))
    latency = _number(record.get("time"))
    if latency and latency > 0:
        summary["latency_ms"] = 1000.0 * latency
        summary["throughput_per_s"] = 1.0 / latency

    class_rows: Dict[str, Dict[str, Any]] = {}
    class_pattern = re.compile(
        r"/([^/]+)_(AP_dist_(0\.5|1\.0|2\.0|4\.0)|"
        r"trans_err|scale_err|orient_err|vel_err|attr_err)$"
    )
    output_names = {
        "trans_err": "ATE",
        "scale_err": "ASE",
        "orient_err": "AOE",
        "vel_err": "AVE",
        "attr_err": "AAE",
    }
    for key, value in record.items():
        match = class_pattern.search(key)
        if not match:
            continue
        name, metric_name, distance = match.groups()
        row = class_rows.setdefault(name, {field: None for field in CLASS_FIELDS})
        row.update(run=run, protocol="nuScenes-3D", **{"class": name})
        if distance:
            row[f"AP_{distance}m"] = _number(value)
        else:
            row[output_names[metric_name]] = _number(value)
    for row in class_rows.values():
        distance_values = [row[f"AP_{distance}m"] for distance in ("0.5", "1.0", "2.0", "4.0")]
        valid = [value for value in distance_values if value is not None]
        row["AP"] = sum(valid) / len(valid) if valid else None
    return summary, list(class_rows.values())


def _parse_coco(
    record: Dict[str, Any], run: str, source: Path, log_tables: Dict[str, Dict[str, Optional[float]]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary = {field: None for field in SUMMARY_FIELDS}
    summary.update(run=run, protocol="COCO-2D", step=record.get("step"), source=str(source))
    for raw_suffix, output_name in COCO_OVERALL_KEYS.items():
        summary[output_name] = _number(_find_key(record, raw_suffix))
    latency = _number(record.get("time"))
    if latency and latency > 0:
        summary["latency_ms"] = 1000.0 * latency
        summary["throughput_per_s"] = 1.0 / latency

    class_names: List[str] = []
    flat_class_ap: Dict[str, Optional[float]] = {}
    for key, value in record.items():
        match = re.search(r"(?:^|/)([^/]+)_precision$", key)
        if match:
            name = match.group(1)
            class_names.append(name)
            flat_class_ap[name] = _number(value)
    for name in log_tables:
        if name not in class_names:
            class_names.append(name)

    per_class: List[Dict[str, Any]] = []
    table_mapping = {
        "mAP": "AP",
        "mAP_50": "AP50",
        "mAP_75": "AP75",
        "mAP_s": "AP_small",
        "mAP_m": "AP_medium",
        "mAP_l": "AP_large",
    }
    for name in class_names:
        row = {field: None for field in CLASS_FIELDS}
        row.update(run=run, protocol="COCO-2D", **{"class": name}, AP=flat_class_ap.get(name))
        for table_name, output_name in table_mapping.items():
            table_value = log_tables.get(name, {}).get(table_name)
            if table_value is not None:
                row[output_name] = table_value
        per_class.append(row)
    return summary, per_class


def parse_run(path: Path, run_name: Optional[str] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    json_paths, log_paths = _candidate_sources(path)
    record, source = _select_record(json_paths)
    if not record or source is None:
        raise ValueError(f"No supported metric record found in {path}")

    run = run_name or (path.stem if path.is_file() else path.name)
    log_tables: Dict[str, Dict[str, Optional[float]]] = {}
    for log_path in log_paths:
        parsed = _parse_coco_tables(log_path)
        if parsed:
            log_tables = parsed

    if "mean_ap" in record or "nd_score" in record:
        return _parse_official_nuscenes(record, run, source)
    if any(key.endswith("/NDS") or "_AP_dist_" in key for key in record):
        return _parse_flat_nuscenes(record, run, source)
    return _parse_coco(record, run, source, log_tables)


def _format(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index == 0 else "---:" for index in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(_format(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path, summaries: List[Dict[str, Any]], class_rows: List[Dict[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    _write_csv(output_dir / "per_class.csv", CLASS_FIELDS, class_rows)
    (output_dir / "thesis_metrics.json").write_text(
        json.dumps({"summary": summaries, "per_class": class_rows}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    summary_columns = [
        "run",
        "protocol",
        "AP",
        "AP50",
        "AP75",
        "NDS",
        "mATE",
        "mASE",
        "mAOE",
        "mAVE",
        "mAAE",
        "latency_ms",
        "throughput_per_s",
    ]
    report = [
        "# LCF3D thesis evaluation metrics",
        "",
        _markdown_table(summary_columns, [[row.get(column) for column in summary_columns] for row in summaries]),
        "",
        "COCO-2D AP and nuScenes-3D AP use different protocols and must not be compared as the same metric. "
        "Higher AP, AR, NDS, and throughput are better; lower nuScenes TP errors and latency are better.",
    ]
    for summary in summaries:
        rows = [row for row in class_rows if row["run"] == summary["run"]]
        if not rows:
            continue
        if summary["protocol"] == "COCO-2D":
            columns = ["class", "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
        else:
            columns = ["class", "AP", "AP_0.5m", "AP_1.0m", "AP_2.0m", "AP_4.0m", "ATE", "ASE", "AOE", "AVE", "AAE"]
        report.extend(
            [
                "",
                f"## {summary['run']}: per-class metrics",
                "",
                _markdown_table(columns, [[row.get(column) for column in columns] for row in rows]),
            ]
        )
    report.extend(
        [
            "",
            "## Reporting notes",
            "",
            "- COCO AP is averaged over IoU thresholds 0.50:0.05:0.95. AP50 and AP75 are reported separately.",
            "- nuScenes AP is averaged over 0.5 m, 1 m, 2 m, and 4 m center-distance thresholds.",
            "- MMDetection names its per-category COCO AP field `*_precision`; this report correctly labels it AP.",
            "- Timing from MMEngine logs is mean model/data-loop time per image or sample, not an end-to-end fusion benchmark.",
            "",
        ]
    )
    (output_dir / "thesis_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Metric files or evaluation directories")
    parser.add_argument("--output-dir", type=Path, default=Path("thesis_metrics"))
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional run names; the number of labels must equal the number of inputs",
    )
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.inputs):
        parser.error("--labels must contain exactly one label per input")

    summaries: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    for index, input_path in enumerate(args.inputs):
        label = args.labels[index] if args.labels else None
        summary, rows = parse_run(input_path.expanduser().resolve(), label)
        summaries.append(summary)
        class_rows.extend(rows)
    write_outputs(args.output_dir.expanduser().resolve(), summaries, class_rows)
    print(f"Wrote thesis metrics to {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()

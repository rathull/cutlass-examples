from __future__ import annotations

import csv
import html
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class PtxasRecord:
    function: str
    arch: str | None = None
    registers: int | None = None
    smem_bytes: int = 0
    stack_frame_bytes: int = 0
    spill_stores_bytes: int = 0
    spill_loads_bytes: int = 0
    cmem_bytes: dict[str, int] = field(default_factory=dict)
    raw: list[str] = field(default_factory=list)

    @property
    def has_spills(self) -> bool:
        return self.spill_stores_bytes > 0 or self.spill_loads_bytes > 0

    @property
    def total_spill_bytes(self) -> int:
        return self.spill_stores_bytes + self.spill_loads_bytes

    @property
    def total_cmem_bytes(self) -> int:
        return sum(self.cmem_bytes.values())

    def row(self) -> dict[str, object]:
        return {
            "function": self.function,
            "arch": self.arch,
            "registers_per_thread": self.registers,
            "smem_bytes": self.smem_bytes,
            "stack_frame_bytes": self.stack_frame_bytes,
            "spill_stores_bytes": self.spill_stores_bytes,
            "spill_loads_bytes": self.spill_loads_bytes,
            "total_spill_bytes": self.total_spill_bytes,
            "cmem_bytes": self.total_cmem_bytes,
            "has_spills": self.has_spills,
        }


@dataclass(frozen=True)
class PtxasReport:
    records: list[PtxasRecord]
    warnings: list[str]
    raw_log: str

    def to_dict(self) -> dict[str, object]:
        return {
            "records": [asdict(record) for record in self.records],
            "warnings": self.warnings,
            "raw_log": self.raw_log,
        }


_COMPILE_RE = re.compile(r"ptxas info\s+:\s+Compiling entry function '([^']+)' for '([^']+)'")
_PROPS_RE = re.compile(r"ptxas info\s+:\s+Function properties for (.+)$")
_STACK_RE = re.compile(
    r"(?P<stack>\d+) bytes stack frame,\s+"
    r"(?P<stores>\d+) bytes spill stores,\s+"
    r"(?P<loads>\d+) bytes spill loads"
)
_USED_RE = re.compile(r"ptxas info\s+:\s+Used (?P<body>.+)$")
_REG_RE = re.compile(r"(\d+) registers?")
_SMEM_RE = re.compile(r"(\d+) bytes smem")
_CMEM_RE = re.compile(r"(\d+) bytes cmem\[(\d+)\]")


def parse_ptxas_log(raw_log: str) -> PtxasReport:
    records_by_function: dict[str, dict[str, Any]] = {}
    current_function: str | None = None
    warnings: list[str] = []

    for line in raw_log.splitlines():
        stripped = line.strip()
        if "ptxas warning" in stripped.lower() or "warning:" in stripped.lower():
            warnings.append(stripped)

        compile_match = _COMPILE_RE.search(stripped)
        if compile_match:
            current_function = compile_match.group(1)
            state = records_by_function.setdefault(
                current_function,
                _new_state(current_function),
            )
            state["arch"] = compile_match.group(2)
            cast(list[str], state["raw"]).append(stripped)
            continue

        props_match = _PROPS_RE.search(stripped)
        if props_match:
            current_function = props_match.group(1).strip()
            state = records_by_function.setdefault(
                current_function,
                _new_state(current_function),
            )
            cast(list[str], state["raw"]).append(stripped)
            continue

        if current_function is None:
            continue

        state = records_by_function[current_function]
        if stripped:
            cast(list[str], state["raw"]).append(stripped)
            if stripped.startswith("ptxas info") and "Compile time" in stripped:
                current_function = None
                continue

        stack_match = _STACK_RE.search(stripped)
        if stack_match:
            state["stack_frame_bytes"] = int(stack_match.group("stack"))
            state["spill_stores_bytes"] = int(stack_match.group("stores"))
            state["spill_loads_bytes"] = int(stack_match.group("loads"))
            continue

        used_match = _USED_RE.search(stripped)
        if used_match:
            body = used_match.group("body")
            reg_match = _REG_RE.search(body)
            if reg_match:
                state["registers"] = int(reg_match.group(1))
            smem_match = _SMEM_RE.search(body)
            if smem_match:
                state["smem_bytes"] = int(smem_match.group(1))
            cmem_bytes = {
                index: int(size)
                for size, index in _CMEM_RE.findall(body)
            }
            if cmem_bytes:
                state["cmem_bytes"] = cmem_bytes

    records = [
        PtxasRecord(
            function=str(state["function"]),
            arch=state["arch"] if isinstance(state["arch"], str) else None,
            registers=state["registers"] if isinstance(state["registers"], int) else None,
            smem_bytes=int(cast(int, state["smem_bytes"])),
            stack_frame_bytes=int(cast(int, state["stack_frame_bytes"])),
            spill_stores_bytes=int(cast(int, state["spill_stores_bytes"])),
            spill_loads_bytes=int(cast(int, state["spill_loads_bytes"])),
            cmem_bytes=dict(cast(dict[str, int], state["cmem_bytes"])),
            raw=list(cast(list[str], state["raw"])),
        )
        for state in records_by_function.values()
    ]
    return PtxasReport(records=records, warnings=warnings, raw_log=raw_log)


def _new_state(function: str) -> dict[str, Any]:
    return {
        "function": function,
        "arch": None,
        "registers": None,
        "smem_bytes": 0,
        "stack_frame_bytes": 0,
        "spill_stores_bytes": 0,
        "spill_loads_bytes": 0,
        "cmem_bytes": {},
        "raw": [],
    }


def print_ptxas_report(report: PtxasReport) -> None:
    if not report.records:
        print("No ptxas function records found.")
        return

    header = (
        f"{'function':64s} {'arch':>7s} {'regs':>6s} {'smem':>10s} "
        f"{'stack':>10s} {'spill':>10s} {'cmem':>10s}"
    )
    print("ptxas resource summary:")
    print(header)
    print("-" * len(header))
    for record in sorted(
        report.records,
        key=lambda item: (
            item.total_spill_bytes,
            item.registers or 0,
            item.smem_bytes,
        ),
        reverse=True,
    ):
        print(
            f"{_shorten(record.function, 64):64s} "
            f"{(record.arch or '-'):>7s} "
            f"{_fmt(record.registers):>6s} "
            f"{record.smem_bytes:10d} "
            f"{record.stack_frame_bytes:10d} "
            f"{record.total_spill_bytes:10d} "
            f"{record.total_cmem_bytes:10d}"
        )

    if report.warnings:
        print("\nptxas warnings:")
        for warning in report.warnings:
            print(f"  {warning}")


def write_ptxas_artifacts(report: PtxasReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ptxas.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    (output_dir / "ptxas.log").write_text(report.raw_log)

    rows = [record.row() for record in report.records]
    if rows:
        with (output_dir / "ptxas.csv").open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    (output_dir / "ptxas.html").write_text(render_ptxas_html(report))


def render_ptxas_html(report: PtxasReport) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td><code>{html.escape(record.function)}</code></td>"
        f"<td>{html.escape(record.arch or '-')}</td>"
        f"<td>{_fmt(record.registers)}</td>"
        f"<td>{record.smem_bytes}</td>"
        f"<td>{record.stack_frame_bytes}</td>"
        f"<td class=\"{'bad' if record.has_spills else 'ok'}\">{record.total_spill_bytes}</td>"
        f"<td>{record.total_cmem_bytes}</td>"
        "</tr>"
        for record in sorted(report.records, key=lambda item: item.function)
    )
    warnings = "\n".join(f"<li>{html.escape(warning)}</li>" for warning in report.warnings)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ptxas Resource Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 0.45rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    code {{ font-size: 0.85rem; }}
    .bad {{ color: #b00020; font-weight: 700; }}
    .ok {{ color: #166534; }}
    pre {{ background: #111827; color: #f9fafb; padding: 1rem; overflow: auto; }}
  </style>
</head>
<body>
  <h1>ptxas Resource Report</h1>
  <p>{len(report.records)} function records, {len(report.warnings)} warnings.</p>
  <table>
    <thead>
      <tr>
        <th>Function</th><th>Arch</th><th>Registers/thread</th>
        <th>SMEM bytes</th><th>Stack bytes</th><th>Spill bytes</th><th>CMEM bytes</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Warnings</h2>
  <ul>{warnings or "<li>None</li>"}</ul>
  <h2>Raw ptxas Log</h2>
  <pre>{html.escape(report.raw_log)}</pre>
</body>
</html>
"""


def _fmt(value: int | None) -> str:
    return "-" if value is None else str(value)


def _shorten(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return "..." + value[-(width - 3):]

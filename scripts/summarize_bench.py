#!/usr/bin/env python3
"""
Summarize a `cargo bench` run of `benches/throughput.rs` into a single,
self-contained HTML file with a click-to-sort table.

Usage:
    python3 scripts/summarize_bench.py                                # reads target/criterion/
    python3 scripts/summarize_bench.py --out report.html
    python3 scripts/summarize_bench.py --criterion path/to/criterion
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass
class Measurement:
    mean_ns: float   # nanoseconds per iteration (of the whole file parse)
    bytes_: int      # file size in bytes


def load_measurements(criterion_dir: Path) -> dict[str, dict[str, Measurement]]:
    """Walk target/criterion and collect (group_id, function_id) -> Measurement."""
    out: dict[str, dict[str, Measurement]] = {}
    for bench_json in criterion_dir.rglob("new/benchmark.json"):
        est_json = bench_json.parent / "estimates.json"
        if not est_json.exists():
            continue
        try:
            bench = json.loads(bench_json.read_text())
            est = json.loads(est_json.read_text())
        except json.JSONDecodeError:
            continue
        throughput = bench.get("throughput")
        if not isinstance(throughput, dict):
            continue
        raw_bytes = throughput.get("BytesDecimal", throughput.get("Bytes"))
        if raw_bytes is None:
            continue
        group = bench["group_id"]
        func = bench["function_id"]
        out.setdefault(group, {})[func] = Measurement(
            mean_ns=est["mean"]["point_estimate"],
            bytes_=int(raw_bytes),
        )
    return out


def mb_per_sec(m: Measurement) -> float:
    # bytes / ns * 1000 == MB/s
    return m.bytes_ / m.mean_ns * 1000.0


def csveee_repr(backends: dict[str, Measurement]) -> Measurement | None:
    """The representative csveee measurement for speedups/aggregates.

    The bench emits `csveee-simd` (only when the dialect is in SIMD's support
    window) and `csveee-dfa` (always); `ParserBackend::Auto` prefers SIMD, so
    the effective number is SIMD-if-present-else-DFA.
    """
    return backends.get("csveee-simd") or backends.get("csveee-dfa")


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def geomean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# Ratios within ±5% count as a tie: roughly where Criterion's noise swamps the
# signal on this suite.
TIE_THRESHOLD = 0.05


def compute_aggregates(measurements: dict[str, dict[str, Measurement]]) -> dict:
    def backend_stats(getter) -> dict:
        total_bytes = 0
        total_ns = 0.0
        mbs_list: list[float] = []
        for backends in measurements.values():
            m = getter(backends)
            if m is None:
                continue
            total_bytes += m.bytes_
            total_ns += m.mean_ns
            mbs_list.append(mb_per_sec(m))
        return {
            "count": len(mbs_list),
            "total_mb_s": (total_bytes / total_ns * 1000.0) if total_ns > 0 else None,
            "geomean_mb_s": geomean(mbs_list),
        }

    def speedup_stats(baseline: str) -> dict:
        speedups: list[float] = []
        wins = losses = ties = 0
        for backends in measurements.values():
            csveee = csveee_repr(backends)
            base = backends.get(baseline)
            if csveee is None or base is None:
                continue
            s = base.mean_ns / csveee.mean_ns
            speedups.append(s)
            if s >= 1.0 + TIE_THRESHOLD:
                wins += 1
            elif s <= 1.0 - TIE_THRESHOLD:
                losses += 1
            else:
                ties += 1
        return {
            "count": len(speedups),
            "geomean": geomean(speedups),
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }

    getters = {
        "csveee": csveee_repr,
        "csveee-dfa": lambda b: b.get("csveee-dfa"),
        "csveee-simd": lambda b: b.get("csveee-simd"),
        "rust-csv": lambda b: b.get("rust-csv"),
        "duckdb": lambda b: b.get("duckdb"),
    }
    return {
        "backends": {b: backend_stats(g) for b, g in getters.items()},
        "vs_rust_csv": speedup_stats("rust-csv"),
        "vs_duckdb": speedup_stats("duckdb"),
    }


def render_summary(agg: dict) -> str:
    b = agg["backends"]
    vr = agg["vs_rust_csv"]
    vd = agg["vs_duckdb"]

    def f(v: float | None, suffix: str = "", digits: int = 1) -> str:
        return f"{v:.{digits}f}{suffix}" if v is not None else "—"

    return f"""
<div class="summary">
  <table class="agg">
    <caption>Throughput</caption>
    <thead><tr>
      <th></th>
      <th>csveee-dfa ({b['csveee-dfa']['count']})</th>
      <th>csveee-simd ({b['csveee-simd']['count']})</th>
      <th>rust-csv ({b['rust-csv']['count']})</th>
      <th>duckdb ({b['duckdb']['count']})</th>
    </tr></thead>
    <tbody>
      <tr><th>Total MB/s</th>
        <td>{f(b['csveee-dfa']['total_mb_s'])}</td>
        <td>{f(b['csveee-simd']['total_mb_s'])}</td>
        <td>{f(b['rust-csv']['total_mb_s'])}</td>
        <td>{f(b['duckdb']['total_mb_s'])}</td></tr>
      <tr><th>Geomean MB/s</th>
        <td>{f(b['csveee-dfa']['geomean_mb_s'])}</td>
        <td>{f(b['csveee-simd']['geomean_mb_s'])}</td>
        <td>{f(b['rust-csv']['geomean_mb_s'])}</td>
        <td>{f(b['duckdb']['geomean_mb_s'])}</td></tr>
    </tbody>
  </table>
  <table class="agg">
    <caption>csveee relative to baseline</caption>
    <thead><tr>
      <th></th>
      <th>vs rust-csv ({vr['count']})</th>
      <th>vs duckdb ({vd['count']})</th>
    </tr></thead>
    <tbody>
      <tr><th>Geomean speedup</th>
        <td>{f(vr['geomean'], '×', 2)}</td>
        <td>{f(vd['geomean'], '×', 2)}</td></tr>
      <tr><th>Wins / Ties / Losses</th>
        <td>{vr['wins']} / {vr['ties']} / {vr['losses']}</td>
        <td>{vd['wins']} / {vd['ties']} / {vd['losses']}</td></tr>
    </tbody>
  </table>
</div>
"""


def build_rows(measurements: dict[str, dict[str, Measurement]]) -> list[dict]:
    rows = []
    for group, backends in measurements.items():
        suite, _, rel = group.partition("::")
        if not rel:
            suite, rel = "", group

        csveee_dfa = backends.get("csveee-dfa")
        csveee_simd = backends.get("csveee-simd")
        csveee = csveee_repr(backends)
        rust_csv = backends.get("rust-csv")
        duckdb = backends.get("duckdb")

        size = next(
            (
                m.bytes_
                for m in (csveee, csveee_dfa, csveee_simd, rust_csv, duckdb)
                if m is not None
            ),
            0,
        )

        def mbs(m: Measurement | None) -> float | None:
            return mb_per_sec(m) if m else None

        def speedup(baseline: Measurement | None) -> float | None:
            # Bytes match, so the MB/s ratio is the ns ratio inverted.
            if csveee is None or baseline is None:
                return None
            return baseline.mean_ns / csveee.mean_ns

        rows.append({
            "suite": suite,
            "file": rel,
            "size": size,
            "csveee": mbs(csveee),
            "csveee_dfa": mbs(csveee_dfa),
            "csveee_simd": mbs(csveee_simd),
            "rust_csv": mbs(rust_csv),
            "duckdb": mbs(duckdb),
            "vs_rust_csv": speedup(rust_csv),
            "vs_duckdb": speedup(duckdb),
        })
    return rows


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>csveee throughput summary</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 1.5rem; color: #222; }
  h1 { margin-top: 0; }
  .meta { color: #666; font-size: 0.9em; margin-bottom: 1rem; }
  table { border-collapse: collapse; width: 100%; font-size: 0.9em; }
  th, td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; text-align: right; }
  th:nth-child(1), th:nth-child(2), td:nth-child(1), td:nth-child(2) { text-align: left; }
  th { background: #f5f5f5; cursor: pointer; user-select: none; position: sticky; top: 0; }
  th.sorted-asc::after  { content: " \\25B2"; font-size: 0.8em; }
  th.sorted-desc::after { content: " \\25BC"; font-size: 0.8em; }
  tr:hover td { background: #fafafa; }
  .fast  { color: #067d17; font-weight: 600; }
  .slow  { color: #b00020; font-weight: 600; }
  .missing { color: #aaa; }
  code { font-family: ui-monospace, monospace; font-size: 0.95em; }
  .summary { display: flex; gap: 2rem; flex-wrap: wrap; margin-bottom: 1.25rem; }
  .agg { width: auto; font-size: 0.9em; }
  .agg caption { text-align: left; font-weight: 600; padding-bottom: 0.3rem; color: #555; }
  .agg th { cursor: default; position: static; }
  .agg th:first-child { text-align: left; font-weight: normal; color: #555; background: transparent; }
  .agg tbody th { background: transparent; }
</style>
</head>
<body>
<h1>csveee throughput summary</h1>
<p class="meta">{meta}</p>
{summary}
<table id="t">
  <thead><tr>
    <th data-key="suite">suite</th>
    <th data-key="file">file</th>
    <th data-key="size">size</th>
    <th data-key="csveee_dfa">csveee-dfa MB/s</th>
    <th data-key="csveee_simd">csveee-simd MB/s</th>
    <th data-key="rust_csv">rust-csv MB/s</th>
    <th data-key="duckdb">duckdb MB/s</th>
    <th data-key="vs_rust_csv">vs rust-csv</th>
    <th data-key="vs_duckdb">vs duckdb</th>
  </tr></thead>
  <tbody>
{rows}
  </tbody>
</table>
<script>
const tbl = document.getElementById('t');
const tbody = tbl.tBodies[0];
let sortKey = null, sortDir = 1;
tbl.tHead.addEventListener('click', (e) => {
  const th = e.target.closest('th');
  if (!th) return;
  const key = th.dataset.key;
  sortDir = (sortKey === key) ? -sortDir : 1;
  sortKey = key;
  for (const h of tbl.tHead.rows[0].cells) h.classList.remove('sorted-asc', 'sorted-desc');
  th.classList.add(sortDir === 1 ? 'sorted-asc' : 'sorted-desc');
  const rows = Array.from(tbody.rows);
  rows.sort((a, b) => {
    const av = a.dataset[key], bv = b.dataset[key];
    const an = parseFloat(av), bn = parseFloat(bv);
    const na = isNaN(an), nb = isNaN(bn);
    // Missing values always sort to the bottom regardless of direction.
    if (na && nb) return 0;
    if (na) return 1;
    if (nb) return -1;
    if (!isNaN(an) && !isNaN(bn)) return (an - bn) * sortDir;
    return av.localeCompare(bv) * sortDir;
  });
  for (const r of rows) tbody.appendChild(r);
});
</script>
</body>
</html>
"""


def fmt_num(v: float | None, digits: int = 1) -> str:
    return f"{v:.{digits}f}" if v is not None else "—"


def fmt_speedup(v: float | None) -> tuple[str, str]:
    """Return (text, css_class)."""
    if v is None:
        return "—", "missing"
    cls = "fast" if v >= 1.1 else "slow" if v <= 0.9 else ""
    return f"{v:.2f}×", cls


def render_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        vs_rc_text, vs_rc_cls = fmt_speedup(r["vs_rust_csv"])
        vs_dd_text, vs_dd_cls = fmt_speedup(r["vs_duckdb"])
        def data(v): return "" if v is None else f"{v}"
        out.append(
            f'    <tr '
            f'data-suite="{escape(r["suite"])}" '
            f'data-file="{escape(r["file"])}" '
            f'data-size="{r["size"]}" '
            f'data-csveee_dfa="{data(r["csveee_dfa"])}" '
            f'data-csveee_simd="{data(r["csveee_simd"])}" '
            f'data-rust_csv="{data(r["rust_csv"])}" '
            f'data-duckdb="{data(r["duckdb"])}" '
            f'data-vs_rust_csv="{data(r["vs_rust_csv"])}" '
            f'data-vs_duckdb="{data(r["vs_duckdb"])}">'
            f'<td>{escape(r["suite"])}</td>'
            f'<td><code>{escape(r["file"])}</code></td>'
            f'<td>{human_bytes(r["size"])}</td>'
            f'<td>{fmt_num(r["csveee_dfa"])}</td>'
            f'<td>{fmt_num(r["csveee_simd"])}</td>'
            f'<td>{fmt_num(r["rust_csv"])}</td>'
            f'<td>{fmt_num(r["duckdb"])}</td>'
            f'<td class="{vs_rc_cls}">{vs_rc_text}</td>'
            f'<td class="{vs_dd_cls}">{vs_dd_text}</td>'
            f'</tr>'
        )
    return "\n".join(out)


def collect_meta() -> dict:
    """Machine and git metadata for a snapshot."""

    def git(*args: str) -> str | None:
        try:
            res = subprocess.run(
                ["git", *args], capture_output=True, text=True, check=False
            )
            if res.returncode != 0:
                return None
            return res.stdout.strip()
        except FileNotFoundError:
            return None

    dirty_out = git("status", "--porcelain")
    return {
        "commit_sha": git("rev-parse", "--short", "HEAD"),
        "commit_date": git("log", "-1", "--format=%cI", "HEAD"),
        "dirty": bool(dirty_out) if dirty_out is not None else None,
        "run_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "arch": platform.machine(),
        "os": platform.system(),
    }


def save_snapshot(
    path: Path, measurements: dict[str, dict[str, Measurement]], meta: dict
) -> None:
    payload = {
        "meta": meta,
        "measurements": {
            group: {
                backend: {"mean_ns": m.mean_ns, "bytes": m.bytes_}
                for backend, m in backends.items()
            }
            for group, backends in measurements.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_snapshot(path: Path) -> tuple[dict, dict[str, dict[str, Measurement]]]:
    data = json.loads(path.read_text())
    measurements: dict[str, dict[str, Measurement]] = {}
    for group, backends in data["measurements"].items():
        measurements[group] = {
            backend: Measurement(mean_ns=m["mean_ns"], bytes_=m["bytes"])
            for backend, m in backends.items()
        }
    return data.get("meta", {}), measurements


def fmt_delta(cur: float | None, prev: float | None) -> tuple[str, str, str]:
    """Return (abs_text, pct_text, css_class) for a change from prev → cur."""
    if cur is None or prev is None:
        return "—", "—", "missing"
    delta = cur - prev
    pct = (delta / prev * 100.0) if prev else 0.0
    cls = "fast" if pct >= 2.0 else "slow" if pct <= -2.0 else ""
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}", f"{sign}{pct:.1f}%", cls


COMPARE_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>csveee throughput compare</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 1.5rem; color: #222; }
  h1 { margin-top: 0; }
  .meta { color: #666; font-size: 0.9em; margin-bottom: 1rem; }
  table { border-collapse: collapse; width: 100%; font-size: 0.9em; }
  th, td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; text-align: right; }
  th:nth-child(1), th:nth-child(2), td:nth-child(1), td:nth-child(2) { text-align: left; }
  th { background: #f5f5f5; cursor: pointer; user-select: none; position: sticky; top: 0; }
  th.sorted-asc::after  { content: " \\25B2"; font-size: 0.8em; }
  th.sorted-desc::after { content: " \\25BC"; font-size: 0.8em; }
  tr:hover td { background: #fafafa; }
  .fast  { color: #067d17; font-weight: 600; }
  .slow  { color: #b00020; font-weight: 600; }
  .missing { color: #aaa; }
  code { font-family: ui-monospace, monospace; font-size: 0.95em; }
</style>
</head>
<body>
<h1>csveee throughput: compare</h1>
<p class="meta">{meta}</p>
<table id="t">
  <thead><tr>
    <th data-key="suite">suite</th>
    <th data-key="file">file</th>
    <th data-key="size">size</th>
    <th data-key="csveee">csveee MB/s</th>
    <th data-key="csveee_prev">prev MB/s</th>
    <th data-key="csveee_delta_pct">Δ%</th>
    <th data-key="vs_rust_csv">vs rust-csv</th>
    <th data-key="vs_rust_csv_prev">prev</th>
    <th data-key="vs_duckdb">vs duckdb</th>
    <th data-key="vs_duckdb_prev">prev</th>
  </tr></thead>
  <tbody>
{rows}
  </tbody>
</table>
<script>
const tbl = document.getElementById('t');
const tbody = tbl.tBodies[0];
let sortKey = null, sortDir = 1;
tbl.tHead.addEventListener('click', (e) => {
  const th = e.target.closest('th');
  if (!th) return;
  const key = th.dataset.key;
  sortDir = (sortKey === key) ? -sortDir : 1;
  sortKey = key;
  for (const h of tbl.tHead.rows[0].cells) h.classList.remove('sorted-asc', 'sorted-desc');
  th.classList.add(sortDir === 1 ? 'sorted-asc' : 'sorted-desc');
  const rows = Array.from(tbody.rows);
  rows.sort((a, b) => {
    const av = a.dataset[key], bv = b.dataset[key];
    const an = parseFloat(av), bn = parseFloat(bv);
    const na = isNaN(an), nb = isNaN(bn);
    if (na && nb) return 0;
    if (na) return 1;
    if (nb) return -1;
    if (!isNaN(an) && !isNaN(bn)) return (an - bn) * sortDir;
    return av.localeCompare(bv) * sortDir;
  });
  for (const r of rows) tbody.appendChild(r);
});
</script>
</body>
</html>
"""


def render_compare_rows(
    cur_rows: list[dict], prev_rows: list[dict]
) -> str:
    prev_by_key = {(r["suite"], r["file"]): r for r in prev_rows}
    out = []
    for r in cur_rows:
        p = prev_by_key.get((r["suite"], r["file"]))
        p_csveee = p["csveee"] if p else None
        p_vs_rc = p["vs_rust_csv"] if p else None
        p_vs_dd = p["vs_duckdb"] if p else None

        _, delta_pct_text, delta_cls = fmt_delta(r["csveee"], p_csveee)
        vs_rc_text, vs_rc_cls = fmt_speedup(r["vs_rust_csv"])
        vs_dd_text, vs_dd_cls = fmt_speedup(r["vs_duckdb"])
        p_vs_rc_text, _ = fmt_speedup(p_vs_rc)
        p_vs_dd_text, _ = fmt_speedup(p_vs_dd)

        delta_pct = None
        if r["csveee"] is not None and p_csveee:
            delta_pct = (r["csveee"] - p_csveee) / p_csveee * 100.0

        def data(v): return "" if v is None else f"{v}"
        out.append(
            f'    <tr '
            f'data-suite="{escape(r["suite"])}" '
            f'data-file="{escape(r["file"])}" '
            f'data-size="{r["size"]}" '
            f'data-csveee="{data(r["csveee"])}" '
            f'data-csveee_prev="{data(p_csveee)}" '
            f'data-csveee_delta_pct="{data(delta_pct)}" '
            f'data-vs_rust_csv="{data(r["vs_rust_csv"])}" '
            f'data-vs_rust_csv_prev="{data(p_vs_rc)}" '
            f'data-vs_duckdb="{data(r["vs_duckdb"])}" '
            f'data-vs_duckdb_prev="{data(p_vs_dd)}">'
            f'<td>{escape(r["suite"])}</td>'
            f'<td><code>{escape(r["file"])}</code></td>'
            f'<td>{human_bytes(r["size"])}</td>'
            f'<td>{fmt_num(r["csveee"])}</td>'
            f'<td>{fmt_num(p_csveee)}</td>'
            f'<td class="{delta_cls}">{delta_pct_text}</td>'
            f'<td class="{vs_rc_cls}">{vs_rc_text}</td>'
            f'<td>{p_vs_rc_text}</td>'
            f'<td class="{vs_dd_cls}">{vs_dd_text}</td>'
            f'<td>{p_vs_dd_text}</td>'
            f'</tr>'
        )
    return "\n".join(out)


def render_compare(
    cur_rows: list[dict],
    prev_rows: list[dict],
    cur_meta: dict,
    prev_meta: dict,
    cur_agg: dict,
    prev_agg: dict,
) -> str:
    def desc(m: dict) -> str:
        sha = m.get("commit_sha") or "?"
        dirty = " (dirty)" if m.get("dirty") else ""
        date = m.get("commit_date") or m.get("run_timestamp") or "?"
        host = m.get("hostname") or "?"
        arch = m.get("arch") or "?"
        return f"{sha}{dirty} · {date} · {host}/{arch}"

    cur_total = cur_agg["backends"]["csveee"]["total_mb_s"]
    prev_total = prev_agg["backends"]["csveee"]["total_mb_s"]
    _, tot_pct, tot_cls = fmt_delta(cur_total, prev_total)

    cur_geo = cur_agg["backends"]["csveee"]["geomean_mb_s"]
    prev_geo = prev_agg["backends"]["csveee"]["geomean_mb_s"]
    _, geo_pct, geo_cls = fmt_delta(cur_geo, prev_geo)

    meta_line = (
        f"current: {escape(desc(cur_meta))}<br>"
        f"previous: {escape(desc(prev_meta))}<br>"
        f"csveee total MB/s: {fmt_num(cur_total)} vs {fmt_num(prev_total)} "
        f"(<span class='{tot_cls}'>{tot_pct}</span>) · "
        f"geomean MB/s: {fmt_num(cur_geo)} vs {fmt_num(prev_geo)} "
        f"(<span class='{geo_cls}'>{geo_pct}</span>)"
    )

    return (
        COMPARE_HTML_TEMPLATE
        .replace("{meta}", meta_line)
        .replace("{rows}", render_compare_rows(cur_rows, prev_rows))
    )


# ---------------------------------------------------------------------------
# History mode: aggregate many snapshots, plot key metrics over time per
# (hostname, arch) so runs from different machines don't get jumbled.
# ---------------------------------------------------------------------------

# Per-machine series colors, legible on white alongside the "fast"/"slow" cells.
_SERIES_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def load_history(history_dir: Path) -> list[dict]:
    """Load every *.json snapshot in a directory, sorted by commit/run date."""
    snapshots = []
    for p in sorted(history_dir.glob("*.json")):
        try:
            meta, measurements = load_snapshot(p)
        except (json.JSONDecodeError, KeyError):
            continue
        agg = compute_aggregates(measurements)
        snapshots.append({
            "path": p,
            "meta": meta,
            "agg": agg,
        })
    snapshots.sort(key=lambda s: s["meta"].get("commit_date")
                   or s["meta"].get("run_timestamp") or "")
    return snapshots


def render_svg_chart(
    title: str,
    series: dict[str, list[tuple[str, float | None, str]]],
    width: int = 700,
    height: int = 260,
) -> str:
    """Render a line chart as inline SVG. `series` maps label → list of
    (x_label, y_value, tooltip) points; x is categorical (commit index)."""
    pad_l, pad_r, pad_t, pad_b = 50, 140, 30, 40

    all_y = [y for pts in series.values() for (_, y, _) in pts if y is not None]
    if not all_y:
        return f'<figure><figcaption>{escape(title)}</figcaption><svg width="{width}" height="{height}"></svg></figure>'

    y_min = min(all_y)
    y_max = max(all_y)
    if y_max == y_min:
        y_max = y_min + 1.0
    y_pad = (y_max - y_min) * 0.1
    y_lo = y_min - y_pad
    y_hi = y_max + y_pad

    x_labels: list[str] = []
    for pts in series.values():
        for (xl, _, _) in pts:
            if xl not in x_labels:
                x_labels.append(xl)
    n = len(x_labels)
    if n == 0:
        return ""

    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def sx(i: int) -> float:
        if n == 1:
            return pad_l + plot_w / 2
        return pad_l + (i / (n - 1)) * plot_w

    def sy(v: float) -> float:
        return pad_t + plot_h - (v - y_lo) / (y_hi - y_lo) * plot_h

    parts = [f'<figure><figcaption>{escape(title)}</figcaption>',
             f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
             f'style="font-family:system-ui,sans-serif;font-size:11px">']

    # Axes
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#999"/>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="#999"/>')

    # Y-axis ticks, 4 evenly spaced.
    for i in range(5):
        v = y_lo + (y_hi - y_lo) * i / 4
        y = sy(v)
        parts.append(f'<line x1="{pad_l - 4}" y1="{y:.1f}" x2="{pad_l}" y2="{y:.1f}" stroke="#999"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 3:.1f}" text-anchor="end" fill="#555">{v:.1f}</text>')

    # Label a few X samples to avoid crowding.
    step = max(1, n // 6)
    for i, xl in enumerate(x_labels):
        if i % step != 0 and i != n - 1:
            continue
        x = sx(i)
        parts.append(f'<line x1="{x:.1f}" y1="{pad_t + plot_h}" x2="{x:.1f}" y2="{pad_t + plot_h + 4}" stroke="#999"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{pad_t + plot_h + 16:.1f}" text-anchor="middle" fill="#555">{escape(xl)}</text>'
        )

    # Series
    for idx, (label, pts) in enumerate(series.items()):
        color = _SERIES_COLORS[idx % len(_SERIES_COLORS)]
        coords = []
        for (xl, y, _tip) in pts:
            if y is None:
                continue
            i = x_labels.index(xl)
            coords.append((sx(i), sy(y)))
        if len(coords) >= 2:
            d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for (x, y) in coords)
            parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        for (xl, y, tip) in pts:
            if y is None:
                continue
            i = x_labels.index(xl)
            parts.append(
                f'<circle cx="{sx(i):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}">'
                f'<title>{escape(tip)}</title></circle>'
            )
        # Legend
        ly = pad_t + idx * 16
        parts.append(
            f'<line x1="{pad_l + plot_w + 10}" y1="{ly}" x2="{pad_l + plot_w + 30}" y2="{ly}" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{pad_l + plot_w + 34}" y="{ly + 4}" fill="#333">{escape(label)}</text>'
        )

    parts.append("</svg></figure>")
    return "\n".join(parts)


def render_history(snapshots: list[dict]) -> str:
    # Group snapshots by machine identity.
    machines: dict[tuple[str, str], list[int]] = {}
    for i, s in enumerate(snapshots):
        key = (s["meta"].get("hostname") or "?", s["meta"].get("arch") or "?")
        machines.setdefault(key, []).append(i)

    def label_for(i: int) -> str:
        m = snapshots[i]["meta"]
        sha = m.get("commit_sha") or f"#{i}"
        return sha

    def tooltip_for(i: int, metric: str, value: float | None) -> str:
        m = snapshots[i]["meta"]
        sha = m.get("commit_sha") or "?"
        date = (m.get("commit_date") or m.get("run_timestamp") or "?")[:19]
        host = m.get("hostname") or "?"
        dirty = " (dirty)" if m.get("dirty") else ""
        val = f"{value:.2f}" if value is not None else "—"
        return f"{sha}{dirty} · {date}\n{host} · {metric}={val}"

    def machine_label(key: tuple[str, str]) -> str:
        host, arch = key
        return f"{host} ({arch})"

    def build_series(extractor) -> dict[str, list[tuple[str, float | None, str]]]:
        out = {}
        for key, idxs in machines.items():
            pts = []
            for i in idxs:
                v = extractor(snapshots[i]["agg"])
                pts.append((label_for(i), v, tooltip_for(i, "", v)))
            out[machine_label(key)] = pts
        return out

    charts = []
    charts.append(render_svg_chart(
        "csveee total MB/s (size-weighted)",
        build_series(lambda a: a["backends"]["csveee"]["total_mb_s"]),
    ))
    charts.append(render_svg_chart(
        "csveee geomean MB/s",
        build_series(lambda a: a["backends"]["csveee"]["geomean_mb_s"]),
    ))
    charts.append(render_svg_chart(
        "geomean speedup vs rust-csv",
        build_series(lambda a: a["vs_rust_csv"]["geomean"]),
    ))
    charts.append(render_svg_chart(
        "geomean speedup vs duckdb",
        build_series(lambda a: a["vs_duckdb"]["geomean"]),
    ))

    rows = []
    for s in snapshots:
        m = s["meta"]
        agg = s["agg"]
        rows.append(
            "<tr>"
            f"<td><code>{escape(m.get('commit_sha') or '?')}</code>"
            f"{' <span class=\"slow\">dirty</span>' if m.get('dirty') else ''}</td>"
            f"<td>{escape((m.get('commit_date') or m.get('run_timestamp') or '')[:19])}</td>"
            f"<td>{escape(m.get('hostname') or '?')}</td>"
            f"<td>{escape((m.get('os') or '') + '/' + (m.get('arch') or ''))}</td>"
            f"<td>{fmt_num(agg['backends']['csveee']['total_mb_s'])}</td>"
            f"<td>{fmt_num(agg['backends']['csveee']['geomean_mb_s'])}</td>"
            f"<td>{fmt_num(agg['vs_rust_csv']['geomean'], 2)}×</td>"
            f"<td>{fmt_num(agg['vs_duckdb']['geomean'], 2)}×</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>csveee throughput history</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1.5rem; color: #222; }}
  h1 {{ margin-top: 0; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 1rem; }}
  figure {{ margin: 0 0 1.5rem 0; }}
  figcaption {{ font-weight: 600; color: #333; margin-bottom: 0.25rem; }}
  .charts {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
  table {{ border-collapse: collapse; font-size: 0.9em; margin-top: 1rem; }}
  th, td {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; text-align: right; }}
  th:nth-child(-n+4), td:nth-child(-n+4) {{ text-align: left; }}
  th {{ background: #f5f5f5; }}
  .slow {{ color: #b00020; font-weight: 600; }}
  code {{ font-family: ui-monospace, monospace; }}
</style>
</head>
<body>
<h1>csveee throughput history</h1>
<p class="meta">{len(snapshots)} snapshot(s) across {len(machines)} machine(s). One line per (hostname, arch).</p>
<div class="charts">
{''.join(charts)}
</div>
<table>
  <thead><tr>
    <th>commit</th><th>date</th><th>host</th><th>os/arch</th>
    <th>total MB/s</th><th>geomean MB/s</th>
    <th>vs rust-csv</th><th>vs duckdb</th>
  </tr></thead>
  <tbody>
{''.join(rows)}
  </tbody>
</table>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--criterion", type=Path, default=Path("target/criterion"),
                    help="Path to the criterion output directory")
    ap.add_argument("--out", type=Path, default=Path("bench_summary.html"),
                    help="Output HTML file")
    ap.add_argument("--save", type=Path, default=None,
                    help="Also write a JSON snapshot (with git/machine metadata) to this path")
    ap.add_argument("--compare", type=Path, default=None,
                    help="Previous snapshot JSON to diff against; writes a compare HTML to --out")
    ap.add_argument("--history", type=Path, default=None,
                    help="Directory of snapshot JSONs; renders a history HTML to --out instead")
    args = ap.parse_args()

    if args.history is not None:
        if not args.history.exists():
            print(f"not found: {args.history}", file=sys.stderr)
            return 1
        snapshots = load_history(args.history)
        if not snapshots:
            print(f"no snapshots found in {args.history}", file=sys.stderr)
            return 1
        args.out.write_text(render_history(snapshots))
        print(f"wrote {args.out} ({len(snapshots)} snapshots)")
        return 0

    if not args.criterion.exists():
        print(f"not found: {args.criterion}", file=sys.stderr)
        return 1

    measurements = load_measurements(args.criterion)
    if not measurements:
        print("no benchmark data found", file=sys.stderr)
        return 1

    rows = build_rows(measurements)
    rows.sort(key=lambda r: (r["suite"], r["file"]))
    aggregates = compute_aggregates(measurements)
    cur_meta = collect_meta()

    if args.save is not None:
        save_snapshot(args.save, measurements, cur_meta)
        print(f"saved snapshot {args.save}")

    if args.compare is not None:
        prev_meta, prev_measurements = load_snapshot(args.compare)
        prev_rows = build_rows(prev_measurements)
        prev_rows.sort(key=lambda r: (r["suite"], r["file"]))
        prev_agg = compute_aggregates(prev_measurements)
        args.out.write_text(render_compare(rows, prev_rows, cur_meta, prev_meta, aggregates, prev_agg))
        print(f"wrote {args.out} (compare: {len(rows)} current vs {len(prev_rows)} prev)")
        return 0

    n_files = len(rows)
    n_with_all = sum(1 for r in rows if r["vs_rust_csv"] is not None and r["vs_duckdb"] is not None)
    meta_bits = [
        f"{n_files} file(s), {n_with_all} with all three backends.",
        "Speedups are csveee relative to the baseline "
        "(<span class='fast'>≥1.1×</span> = csveee faster, "
        "<span class='slow'>≤0.9×</span> = csveee slower).",
        "Click any column header to sort.",
    ]
    if cur_meta.get("commit_sha"):
        dirty = " (dirty)" if cur_meta.get("dirty") else ""
        meta_bits.insert(
            0,
            f"<code>{escape(cur_meta['commit_sha'])}</code>{dirty} · "
            f"{escape((cur_meta.get('commit_date') or '')[:19])} · "
            f"{escape(cur_meta.get('hostname') or '?')}/"
            f"{escape(cur_meta.get('arch') or '?')}",
        )
    meta = " ".join(meta_bits)

    html = (
        HTML_TEMPLATE
        .replace("{meta}", meta)
        .replace("{summary}", render_summary(aggregates))
        .replace("{rows}", render_rows(rows))
    )
    args.out.write_text(html)
    print(f"wrote {args.out} ({n_files} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Render a ``bench.py`` JSON report as a print-ready document (HTML + PDF).

The layout is fixed so results from different machines are directly
comparable: cover block, system table, cross-GPU summary, one section per GPU
with achieved-vs-datasheet bars, methodology and sources.

    python3 report.py results/bench-*.json -o report.pdf
    python3 report.py results/bench-*.json -o report.pdf \
        --title "DGX Station Benchmark" --subtitle "Kunde / Projekt" \
        --operator "primeLine Solutions GmbH"

PDF conversion uses WeasyPrint (best CSS paged-media support, real page
numbers), then headless Chrome, then wkhtmltopdf.  The HTML is always written
next to the PDF so the document can be inspected or re-printed.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import Any

# --------------------------------------------------------------------------- #
# i18n
# --------------------------------------------------------------------------- #
STRINGS: dict[str, dict[str, str]] = {
    "de": {
        "kicker": "Hardware-Benchmark",
        "default_title": "GPU-Benchmark-Bericht",
        "system": "System",
        "date": "Datum",
        "driver": "Treiber",
        "stack": "PyTorch / CUDA",
        "operator": "Erstellt von",
        "host": "Host-System",
        "cpu": "CPU",
        "ram": "Arbeitsspeicher",
        "os": "Betriebssystem",
        "gpus": "Grafikprozessoren",
        "summary": "Zusammenfassung",
        "summary_note": "Alle Werte sind Dauerlastwerte über 3 s je Phase, "
                        "gemessen mit CUDA-Events. Prozentwerte beziehen sich "
                        "auf die Dense-Peak-Werte der Hersteller-Datenblätter.",
        "metric": "Metrik",
        "device": "Gerät",
        "memory": "Speicher",
        "compute_cap": "Compute Capability",
        "sms": "SMs",
        "power_limit": "Power-Limit",
        "bandwidth": "Speicherbandbreite",
        "compute": "Rechenleistung",
        "interconnect": "Anbindung Host ↔ GPU",
        "phase": "Phase",
        "achieved": "Erreicht",
        "share": "Anteil Peak",
        "power_point": "Betriebspunkt",
        "observations": "Beobachtungen",
        "method": "Messmethodik",
        "sources": "Referenzwerte",
        "p2p": "GPU-zu-GPU",
        "skipped": "nicht messbar",
        "read": "Lesen",
        "write": "Schreiben",
        "copy": "Kopieren",
        "triad": "Triad (2R+1W)",
        "h2d": "Host → GPU",
        "d2h": "GPU → Host",
        "page": "Seite",
        "of": "von",
        "no_data": "keine Daten",
        "appendix": "Anhang: Rohdaten",
    },
    "en": {
        "kicker": "Hardware benchmark",
        "default_title": "GPU Benchmark Report",
        "system": "System",
        "date": "Date",
        "driver": "Driver",
        "stack": "PyTorch / CUDA",
        "operator": "Prepared by",
        "host": "Host system",
        "cpu": "CPU",
        "ram": "Memory",
        "os": "Operating system",
        "gpus": "GPUs",
        "summary": "Summary",
        "summary_note": "All figures are sustained values over 3 s per phase, "
                        "timed with CUDA events. Percentages refer to the dense "
                        "peak figures published in the vendor datasheets.",
        "metric": "Metric",
        "device": "Device",
        "memory": "Memory",
        "compute_cap": "Compute capability",
        "sms": "SMs",
        "power_limit": "Power limit",
        "bandwidth": "Memory bandwidth",
        "compute": "Compute throughput",
        "interconnect": "Host ↔ GPU interconnect",
        "phase": "Phase",
        "achieved": "Achieved",
        "share": "Share of peak",
        "power_point": "Operating point",
        "observations": "Observations",
        "method": "Methodology",
        "sources": "Reference figures",
        "p2p": "GPU-to-GPU",
        "skipped": "not measurable",
        "read": "Read",
        "write": "Write",
        "copy": "Copy",
        "triad": "Triad (2R+1W)",
        "h2d": "Host → GPU",
        "d2h": "GPU → Host",
        "page": "Page",
        "of": "of",
        "no_data": "no data",
        "appendix": "Appendix: raw data",
    },
}

METHOD = {
    "de": [
        "Jede Messphase läuft über ein festes Zeitbudget von 3 s, aufgeteilt in "
        "drei Blöcke; der schnellste Block geht in den Bericht ein. Kurze "
        "Bursts würden Boost-Taktraten zeigen, die unter Dauerlast nicht "
        "gehalten werden.",
        "Speicherbandbreite: 8 GiB große Puffer je Strom, gemessen als Lesen "
        "(Reduktion), Schreiben (Fill), Kopieren (1R+1W) und Triad (2R+1W). "
        "Angegeben ist die tatsächlich bewegte Datenmenge pro Sekunde.",
        "Rechenleistung: Matrixmultiplikation 16384³ über cuBLAS, 2·N³ FLOPs "
        "pro Durchlauf. FP32 ohne TF32, TF32 separat, FP8 als e4m3 und FP4 als "
        "nvfp4 mit 1×16-Block-Skalierung.",
        "Anbindung: gepinnter Host-Speicher, 1 GiB je Richtung. Auf "
        "Grace-Blackwell-Systemen misst das die NVLink-C2C-Verbindung, sonst "
        "PCIe.",
        "Betriebspunkt: Leistungsaufnahme, Temperatur und SM-Takt werden über "
        "NVML alle 0,25 s abgefragt und je Phase gemittelt. Das zeigt, ob eine "
        "Karte im Power-Limit läuft oder thermisch drosselt.",
        "Jede GPU läuft in einem eigenen Prozess (CUDA_VISIBLE_DEVICES per "
        "UUID). Systeme, die GPUs in unterschiedlichen Adressmodi (HMM und "
        "ATS) mischen, erlauben keinen gemeinsamen CUDA-Kontext; GPU-zu-GPU-"
        "Messungen entfallen dort.",
    ],
    "en": [
        "Every phase runs for a fixed 3 s budget split into three blocks; the "
        "fastest block is reported. Short bursts would show boost clocks that "
        "cannot be sustained under load.",
        "Memory bandwidth: 8 GiB buffers per stream, measured as read "
        "(reduction), write (fill), copy (1R+1W) and triad (2R+1W). The figure "
        "is the amount of data actually moved per second.",
        "Compute: 16384³ matrix multiplication through cuBLAS, 2·N³ FLOPs per "
        "run. FP32 without TF32, TF32 separately, FP8 as e4m3 and FP4 as nvfp4 "
        "with 1×16 block scaling.",
        "Interconnect: pinned host memory, 1 GiB per direction. On "
        "Grace-Blackwell systems this measures NVLink-C2C, otherwise PCIe.",
        "Operating point: power draw, temperature and SM clock are sampled via "
        "NVML every 0.25 s and averaged per phase. This shows whether a card "
        "is power-capped or thermally throttled.",
        "Each GPU runs in its own process (CUDA_VISIBLE_DEVICES by UUID). "
        "Systems that mix GPUs in different addressing modes (HMM and ATS) do "
        "not allow a shared CUDA context; GPU-to-GPU measurements are omitted "
        "there.",
    ],
}

SOURCES = [
    ("NVIDIA Blackwell Ultra datasheet — Individual Blackwell Ultra GPU "
     "Specifications (HGX B300 column, dense figures)",
     "https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet"),
    ("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition datasheet",
     "https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/"
     "professional-desktop-gpus/rtx-pro-6000-max-q/"
     "workstation-datasheet-blackwell-rtx-pro-6000-max-q-nvidia-us-5349650-web.pdf"),
]

# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
CSS = """
@page {
  size: A4;
  margin: 18mm 15mm 16mm 15mm;
  @bottom-left  { content: string(doctitle); font-size: 7.5pt; color: #8a94a6; }
  @bottom-right { content: "PAGE_LABEL " counter(page) " OF_LABEL " counter(pages);
                  font-size: 7.5pt; color: #8a94a6; }
}
* { box-sizing: border-box; }
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 9.2pt; line-height: 1.45; color: #1d2433; margin: 0;
}
h1 { font-size: 21pt; margin: 2mm 0 1mm 0; letter-spacing: -0.3pt;
     string-set: doctitle content(); }
h2 { font-size: 12.5pt; margin: 7mm 0 2mm 0; padding-bottom: 1.2mm;
     border-bottom: 0.6pt solid #d4dae4; color: #14304f; }
h3 { font-size: 10pt; margin: 5mm 0 1.5mm 0; color: #14304f;
     page-break-after: avoid; }
p  { margin: 0 0 2mm 0; }
.kicker { font-size: 8pt; letter-spacing: 1.6pt; text-transform: uppercase;
          color: #2f6feb; font-weight: 700; }
.subtitle { font-size: 11pt; color: #4a5568; margin-bottom: 4mm; }
.rule { height: 2.2pt; background: #14304f; margin: 3mm 0 5mm 0; }
table { width: 100%; border-collapse: collapse; margin: 1mm 0 3mm 0;
        page-break-inside: avoid; }
th, td { text-align: left; padding: 1.5mm 2mm; vertical-align: middle; }
th { font-size: 7.6pt; text-transform: uppercase; letter-spacing: 0.5pt;
     color: #5a6478; border-bottom: 0.8pt solid #c9d1de; font-weight: 700; }
td { border-bottom: 0.4pt solid #e6eaf1; }
tr.total td { font-weight: 700; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.meta td:first-child { width: 34mm; color: #5a6478; font-size: 8.4pt; }
.meta td { border-bottom: 0.4pt solid #eef1f6; }
.card { border: 0.6pt solid #dde3ec; border-radius: 2mm; padding: 3mm 4mm;
        margin: 0 0 4mm 0; background: #fbfcfe; }
.gpu-head { background: #14304f; color: #ffffff; padding: 2.5mm 4mm;
            border-radius: 2mm 2mm 0 0; margin-top: 0;
            page-break-after: avoid; }
.gpu-head .name { font-size: 11.5pt; font-weight: 700; }
.gpu-head .facts { font-size: 8pt; color: #c3d0e2; margin-top: 0.8mm; }
.gpu-body { border: 0.6pt solid #dde3ec; border-top: none; padding: 3mm 4mm 1mm 4mm; }
.bar { display: inline-block; width: 34mm; height: 2.6mm; background: #e8ecf3;
       border-radius: 1.3mm; vertical-align: middle; }
.bar span { display: block; height: 2.6mm; border-radius: 1.3mm; background: #2f6feb; }
.bar span.warn { background: #d98324; }
.bar span.bad  { background: #c0392b; }
.pct { display: inline-block; width: 12mm; text-align: right; font-size: 8.2pt;
       color: #4a5568; }
.point { font-size: 7.8pt; color: #6b7688; }
ul { margin: 0 0 2mm 0; padding-left: 5mm; }
li { margin-bottom: 1.2mm; }
.note { font-size: 8.2pt; color: #5a6478; }
.pagebreak { page-break-before: always; }
.tag { display: inline-block; font-size: 7.4pt; padding: 0.4mm 1.6mm;
       border-radius: 1mm; background: #eaf0fb; color: #2f6feb; font-weight: 700; }
.tag.warn { background: #fdf1e3; color: #b06a12; }
.footer-note { margin-top: 6mm; font-size: 7.6pt; color: #8a94a6;
               border-top: 0.4pt solid #e6eaf1; padding-top: 2mm; }
"""

# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "–"
    try:
        return f"{float(value):,.{digits}f}".replace(",", " ")
    except (TypeError, ValueError):
        return esc(value)


def bar(pct: float | None) -> str:
    if pct is None:
        return ""
    width = max(0.0, min(100.0, pct))
    cls = "" if pct >= 70 else (" warn" if pct >= 45 else " bad")
    return (f'<span class="bar"><span class="{cls.strip()}" '
            f'style="width:{width:.1f}%"></span></span> '
            f'<span class="pct">{pct:.0f} %</span>')


def point(power: dict[str, float] | None) -> str:
    if not power:
        return ""
    return (f'<span class="point">{power["power_avg_w"]:.0f} W · '
            f'{power["clock_avg_mhz"] / 1000:.2f} GHz · '
            f'{power["temp_max_c"]:.0f} °C</span>')


def short_name(name: str) -> str:
    """'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition' -> 'RTX PRO 6000 ...'."""
    return name.replace("NVIDIA ", "").replace(" Workstation Edition", "").strip()


def pretty_date(value: str) -> str:
    if not value:
        return "–"
    text = value.strip()
    # Python < 3.11 rejects "+0100"; normalise to "+01:00".
    if len(text) >= 5 and text[-5] in "+-" and text[-3] != ":":
        text = text[:-2] + ":" + text[-2:]
    try:
        return datetime.fromisoformat(text).strftime("%d.%m.%Y, %H:%M")
    except ValueError:
        return value


def share(value: float | None, peak: float | None) -> float | None:
    if not value or not peak:
        return None
    return 100.0 * value / peak


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def cover(report: dict[str, Any], meta: dict[str, str], t: dict[str, str]) -> str:
    cpu = report.get("cpu", {})
    gpus = report.get("gpus", [])
    names = ", ".join(g["info"]["name"] for g in gpus) or t["no_data"]
    driver = next((g["info"].get("driver") for g in gpus if g["info"].get("driver")), "–")
    rows = [
        (t["system"], names),
        (t["date"], pretty_date(report.get("timestamp", ""))),
        (t["driver"], driver),
        (t["stack"], f"{cpu.get('torch', '–')} / CUDA {cpu.get('cuda', '–')}"),
    ]
    if meta.get("operator"):
        rows.append((t["operator"], meta["operator"]))
    meta_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in rows
    )
    subtitle = f'<div class="subtitle">{esc(meta["subtitle"])}</div>' if meta.get("subtitle") else ""
    return f"""
<div class="kicker">{esc(t['kicker'])}</div>
<h1>{esc(meta['title'])}</h1>
{subtitle}
<div class="rule"></div>
<table class="meta">{meta_rows}</table>
"""


def host_section(report: dict[str, Any], t: dict[str, str]) -> str:
    cpu = report.get("cpu", {})
    rows = [
        (t["cpu"], f"{cpu.get('cpu_model', '–')} ({cpu.get('cpu_cores', '–')} Cores, "
                   f"{cpu.get('machine', '–')})"),
        (t["ram"], f"{num(cpu.get('ram_gib'), 0)} GiB"),
        (t["os"], f"{cpu.get('os', '–')} · Kernel {cpu.get('kernel', '–')}"),
    ]
    if cpu.get("triad_gbs"):
        rows.append((t["bandwidth"], f"CPU-Triad {num(cpu['triad_gbs'])} GB/s · "
                                     f"BLAS {num(cpu.get('gemm_gflops'), 0)} GFLOP/s "
                                     f"({cpu.get('threads', '–')} Threads)"))
    body = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in rows)
    return f'<h2>{esc(t["host"])}</h2><table class="meta">{body}</table>'


def summary_section(report: dict[str, Any], t: dict[str, str]) -> str:
    gpus = report.get("gpus", [])
    if not gpus:
        return ""
    head = "".join(
        f'<th class="num">GPU {esc(g["info"]["index"])} · {esc(short_name(g["info"]["name"]))}</th>'
        for g in gpus
    )

    def row(label: str, getter) -> str:
        cells = "".join(f'<td class="num">{getter(g)}</td>' for g in gpus)
        return f"<tr><td>{esc(label)}</td>{cells}</tr>"

    def hbm(phase: str):
        return lambda g: num(g.get("hbm", {}).get("phases", {}).get(phase, {}).get("gbs_best"))

    def comp(dtype: str):
        return lambda g: num(g.get("compute", {}).get("phases", {}).get(dtype, {}).get("tflops_best"))

    def link(direction: str):
        return lambda g: num(g.get("host_device", {}).get(direction, {}).get("gbs_best"))

    rows = [
        row(f'{t["bandwidth"]} · {t["copy"]} (GB/s)', hbm("copy")),
        row(f'{t["bandwidth"]} · {t["triad"]} (GB/s)', hbm("triad")),
        row("FP32 (TFLOP/s)", comp("fp32")),
        row("BF16 (TFLOP/s)", comp("bf16")),
        row("FP8 (TFLOP/s)", comp("fp8")),
        row("FP4 (TFLOP/s)", comp("fp4")),
        row(f'{t["interconnect"]} · {t["h2d"]} (GB/s)', link("h2d")),
        row(f'{t["interconnect"]} · {t["d2h"]} (GB/s)', link("d2h")),
        row("Power max (W)", lambda g: num(g.get("power", {}).get("power_max_w"), 0)),
    ]
    return f"""
<h2>{esc(t['summary'])}</h2>
<table>
  <tr><th>{esc(t['metric'])}</th>{head}</tr>
  {''.join(rows)}
</table>
<p class="note">{esc(t['summary_note'])}</p>
"""


def gpu_section(gpu: dict[str, Any], t: dict[str, str]) -> str:
    info = gpu["info"]
    sheet = info.get("datasheet", {})
    facts = [
        f'{num(info.get("memory_total_gib"), 0)} GiB',
        f'CC {info.get("compute_capability", "–")}',
        f'{info.get("sm_count", "–")} SMs',
        f'{t["power_limit"]} {num(info.get("power_limit_w"), 0)} W',
    ]
    if info.get("clock_max_mhz"):
        facts.append(f'Max. SM-Takt {num(info["clock_max_mhz"], 0)} MHz')
    head = f"""
<div class="pagebreak"></div>
<div class="gpu-head">
  <div class="name">GPU {esc(info.get('index'))} · {esc(info['name'])}</div>
  <div class="facts">{esc(' · '.join(facts))}</div>
</div>
<div class="gpu-body">
"""

    parts = [head]

    hbm = gpu.get("hbm")
    if hbm:
        rows = []
        for phase in ("read", "write", "copy", "triad"):
            values = hbm["phases"].get(phase)
            if not values:
                continue
            pct = share(values["gbs_best"], sheet.get("hbm_gbs"))
            rows.append(
                f'<tr><td>{esc(t[phase])}</td>'
                f'<td class="num">{num(values["gbs_best"])} GB/s</td>'
                f'<td>{bar(pct)}</td>'
                f'<td>{point(values.get("power"))}</td></tr>'
            )
        peak = f' · Peak {num(sheet["hbm_gbs"], 0)} GB/s' if sheet.get("hbm_gbs") else ""
        parts.append(
            f'<h3>{esc(t["bandwidth"])} <span class="note">'
            f'({num(hbm["buffer_gib"], 0)} GiB Puffer{peak})</span></h3>'
            f'<table><tr><th>{esc(t["phase"])}</th><th class="num">{esc(t["achieved"])}</th>'
            f'<th>{esc(t["share"])}</th><th>{esc(t["power_point"])}</th></tr>'
            f'{"".join(rows)}</table>'
        )

    comp = gpu.get("compute")
    if comp:
        rows = []
        for dtype, values in comp["phases"].items():
            if "skipped" in values:
                rows.append(f'<tr><td>{esc(dtype.upper())}</td>'
                            f'<td colspan="3" class="note">{esc(t["skipped"])}</td></tr>')
                continue
            pct = values.get("pct_of_datasheet")
            peak = values.get("datasheet_tflops")
            peak_txt = (f'<span class="note">Peak {num(peak, 0)}</span>' if peak
                        else '<span class="note">kein Dense-Wert</span>')
            rows.append(
                f'<tr><td>{esc(dtype.upper())}</td>'
                f'<td class="num">{num(values["tflops_best"])} TFLOP/s</td>'
                f'<td>{bar(pct)} {peak_txt}</td>'
                f'<td>{point(values.get("power"))}</td></tr>'
            )
        parts.append(
            f'<h3>{esc(t["compute"])} <span class="note">'
            f'(GEMM {comp["matrix"]}³, cuBLAS)</span></h3>'
            f'<table><tr><th>Präzision</th><th class="num">{esc(t["achieved"])}</th>'
            f'<th>{esc(t["share"])}</th><th>{esc(t["power_point"])}</th></tr>'
            f'{"".join(rows)}</table>'
        )

    link = gpu.get("host_device")
    if link:
        rows = []
        for direction in ("h2d", "d2h"):
            values = link.get(direction)
            if not values:
                continue
            rows.append(f'<tr><td>{esc(t[direction])}</td>'
                        f'<td class="num">{num(values["gbs_best"])} GB/s</td>'
                        f'<td>{point(link.get("power"))}</td></tr>')
        parts.append(
            f'<h3>{esc(t["interconnect"])} <span class="note">'
            f'({num(link["buffer_mib"], 0)} MiB gepinnter Speicher)</span></h3>'
            f'<table><tr><th>{esc(t["phase"])}</th><th class="num">{esc(t["achieved"])}</th>'
            f'<th>{esc(t["power_point"])}</th></tr>{"".join(rows)}</table>'
        )

    observations = build_observations(gpu, t)
    if observations:
        parts.append(f'<h3>{esc(t["observations"])}</h3><ul>'
                     + "".join(f"<li>{o}</li>" for o in observations) + "</ul>")

    parts.append("</div>")
    return "".join(parts)


def build_observations(gpu: dict[str, Any], t: dict[str, str]) -> list[str]:
    """Derive plain-language findings from the numbers (no hand-written claims)."""
    info = gpu["info"]
    sheet = info.get("datasheet", {})
    out: list[str] = []

    hbm = gpu.get("hbm", {}).get("phases", {})
    if hbm and sheet.get("hbm_gbs"):
        best = max(v["gbs_best"] for v in hbm.values())
        pct = 100.0 * best / sheet["hbm_gbs"]
        verdict = ("im erwarteten Bereich" if pct >= 75 else
                   "unter den Erwartungen" if pct >= 60 else "auffällig niedrig")
        out.append(f"Speicherbandbreite erreicht {pct:.0f} % des Datenblatt-Peaks "
                   f"({num(best)} von {num(sheet['hbm_gbs'], 0)} GB/s) — {verdict}.")

    comp = gpu.get("compute", {}).get("phases", {})
    fp32 = comp.get("fp32", {})
    if fp32.get("pct_of_datasheet"):
        out.append(f"FP32 (CUDA-Cores) erreicht {fp32['pct_of_datasheet']:.0f} % des "
                   f"Peaks — der sauberste Indikator dafür, dass die Karte unter "
                   f"Last nicht drosselt.")

    tensor = [(k, v) for k, v in comp.items()
              if k in ("tf32", "bf16", "fp16", "fp8", "fp4") and v.get("pct_of_datasheet")]
    if len(tensor) > 1:
        low = min(v["pct_of_datasheet"] for _, v in tensor)
        high = max(v["pct_of_datasheet"] for _, v in tensor)
        out.append(f"Tensor-Core-Werte liegen bei {low:.0f}–{high:.0f} % des "
                   f"Dense-Peaks. Die Datenblatt-Peaks gelten für Boost-Takt; "
                   f"unter Dauerlast ist dieser Abstand normal.")
    elif tensor:
        name, values = tensor[0]
        out.append(f"{name.upper()} erreicht {values['pct_of_datasheet']:.0f} % des "
                   f"Dense-Peaks — der einzige von NVIDIA für diese SKU "
                   f"veröffentlichte Dense-Tensorwert.")

    power = gpu.get("power", {})
    limit = info.get("power_limit_w")
    if power and limit:
        ratio = 100.0 * power["power_max_w"] / limit
        if ratio >= 97:
            out.append(f"Die Karte läuft am Power-Limit ({num(power['power_max_w'], 0)} W "
                       f"bei {num(limit, 0)} W Limit) — die Leistungsaufnahme, nicht "
                       f"die Rechenleistung, begrenzt den Durchsatz.")
        elif ratio < 80:
            out.append(f"Die Karte bleibt mit {num(power['power_max_w'], 0)} W deutlich "
                       f"unter ihrem Power-Limit ({num(limit, 0)} W); der Durchsatz "
                       f"wird nicht durch die Leistungsaufnahme begrenzt.")

    clock_max = info.get("clock_max_mhz")
    if power and clock_max and power.get("clock_avg_mhz"):
        pct = 100.0 * power["clock_avg_mhz"] / clock_max
        out.append(f"Durchschnittlicher SM-Takt {num(power['clock_avg_mhz'], 0)} MHz "
                   f"von maximal {num(clock_max, 0)} MHz ({pct:.0f} %).")

    if power.get("temp_max_c"):
        out.append(f"Maximale Temperatur {num(power['temp_max_c'], 0)} °C.")
    return out


def p2p_section(report: dict[str, Any], t: dict[str, str]) -> str:
    links = report.get("p2p", [])
    if not links:
        return ""
    rows = []
    for link in links:
        if "skipped" in link:
            rows.append(f'<tr><td colspan="2" class="note">{esc(t["skipped"])}: '
                        f'{esc(link["skipped"])}</td></tr>')
            continue
        rows.append(f'<tr><td>GPU {esc(link["src"])} → GPU {esc(link["dst"])}</td>'
                    f'<td class="num">{num(link["gbs_best"])} GB/s</td></tr>')
    return (f'<h2>{esc(t["p2p"])}</h2><table>'
            f'<tr><th>{esc(t["phase"])}</th><th class="num">{esc(t["achieved"])}</th></tr>'
            f'{"".join(rows)}</table>')


def method_section(t: dict[str, str], lang: str) -> str:
    items = "".join(f"<li>{esc(x)}</li>" for x in METHOD[lang])
    return f'<h2>{esc(t["method"])}</h2><ul>{items}</ul>'


def sources_section(t: dict[str, str]) -> str:
    items = "".join(
        f'<li>{esc(label)}<br><span class="note">{esc(url)}</span></li>'
        for label, url in SOURCES
    )
    return f'<h2>{esc(t["sources"])}</h2><ul>{items}</ul>'


def render(report: dict[str, Any], meta: dict[str, str], lang: str) -> str:
    t = STRINGS[lang]
    body = [
        cover(report, meta, t),
        host_section(report, t),
        summary_section(report, t),
    ]
    for gpu in report.get("gpus", []):
        body.append(gpu_section(gpu, t))
    body.append('<div class="pagebreak"></div>')
    body.append(p2p_section(report, t))
    body.append(method_section(t, lang))
    body.append(sources_section(t))
    if meta.get("notes"):
        body.append(f'<h2>{esc(t["observations"])}</h2>'
                    f'<p>{esc(meta["notes"])}</p>')
    body.append(
        f'<div class="footer-note">sys-stress bench · Bericht erzeugt am '
        f'{datetime.now().strftime("%d.%m.%Y %H:%M")} · '
        f'Rohdaten: {esc(meta.get("source", "–"))}</div>'
    )
    css = CSS.replace("PAGE_LABEL", t["page"]).replace("OF_LABEL", t["of"])
    return (f"<!DOCTYPE html><html lang=\"{lang}\"><head><meta charset=\"utf-8\">"
            f"<title>{esc(meta['title'])}</title><style>{css}</style></head>"
            f"<body>{''.join(body)}</body></html>")


# --------------------------------------------------------------------------- #
# PDF backends
# --------------------------------------------------------------------------- #
def _chrome_binary() -> str | None:
    candidates = [
        os.environ.get("CHROME"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return shutil.which("google-chrome") or shutil.which("chromium")


def to_pdf(html_path: str, pdf_path: str) -> str:
    """Convert HTML to PDF, returning the backend that succeeded."""
    errors: list[str] = []

    if shutil.which("weasyprint"):
        proc = subprocess.run(["weasyprint", html_path, pdf_path],
                              capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(pdf_path):
            return "weasyprint"
        errors.append(f"weasyprint: {proc.stderr.strip().splitlines()[-1:] or 'failed'}")

    chrome = _chrome_binary()
    if chrome:
        proc = subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
             "--virtual-time-budget=2000",
             f"--print-to-pdf={pdf_path}", f"file://{os.path.abspath(html_path)}"],
            capture_output=True, text=True,
        )
        if proc.returncode == 0 and os.path.exists(pdf_path):
            return "chrome"
        errors.append(f"chrome: {proc.stderr.strip().splitlines()[-1:] or 'failed'}")

    if shutil.which("wkhtmltopdf"):
        proc = subprocess.run(
            ["wkhtmltopdf", "--enable-local-file-access", "--quiet",
             html_path, pdf_path], capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(pdf_path):
            return "wkhtmltopdf"
        errors.append(f"wkhtmltopdf: {proc.stderr.strip().splitlines()[-1:] or 'failed'}")

    raise RuntimeError("No PDF backend available (tried weasyprint, chrome, "
                       "wkhtmltopdf).\n" + "\n".join(errors))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a bench.py JSON report as PDF.")
    parser.add_argument("report", help="JSON file produced by bench.py --json")
    parser.add_argument("-o", "--out", default="report.pdf", help="output PDF path")
    parser.add_argument("--title", help="document title")
    parser.add_argument("--subtitle", help="subtitle, e.g. customer or project")
    parser.add_argument("--operator", help="who ran the benchmark")
    parser.add_argument("--notes", help="free-text paragraph appended to the report")
    parser.add_argument("--lang", choices=sorted(STRINGS), default="de")
    parser.add_argument("--html-only", action="store_true",
                        help="write the HTML but skip PDF conversion")
    args = parser.parse_args(argv)

    with open(args.report, encoding="utf-8") as fh:
        report = json.load(fh)

    t = STRINGS[args.lang]
    meta = {
        "title": args.title or t["default_title"],
        "subtitle": args.subtitle or "",
        "operator": args.operator or "",
        "notes": args.notes or "",
        "source": os.path.basename(args.report),
    }
    document = render(report, meta, args.lang)

    out_pdf = os.path.abspath(args.out)
    html_path = os.path.splitext(out_pdf)[0] + ".html"
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(document)
    print(f"HTML: {html_path}")

    if args.html_only:
        return 0
    backend = to_pdf(html_path, out_pdf)
    size_kb = os.path.getsize(out_pdf) / 1024
    print(f"PDF : {out_pdf} ({size_kb:.0f} KiB, via {backend})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

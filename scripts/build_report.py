"""reports/technical_report.md -> reports/technical_report.pdf (A4), enforcing the 6-page cap.

Uses Chromium via Playwright to print. Set CHROMIUM=/path/to/chrome if Playwright's bundled browser is absent.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
CSS = """
@page { size: A4; margin: 14mm 14mm 14mm 14mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 9.2pt; line-height: 1.32; color: #222; }
h1 { font-size: 15pt; margin: 0 0 4pt; } h2 { font-size: 11.5pt; margin: 9pt 0 3pt; border-bottom: 1px solid #ccc; }
h3 { font-size: 10pt; margin: 6pt 0 2pt; } p { margin: 3pt 0; } ul { margin: 2pt 0 2pt 14pt; padding: 0; }
table { border-collapse: collapse; margin: 4pt 0; font-size: 8.2pt; } td, th { border: 1px solid #bbb; padding: 1.5pt 4pt; }
th { background: #f0f0f0; } code { font-size: 8.2pt; } img { max-width: 100%; }
.fig { display: flex; gap: 6pt; } .fig img { width: 49%; }
"""


def main():
    src = ROOT / "reports" / "technical_report.md"
    out = ROOT / "reports" / "technical_report.pdf"
    lines, prev = [], ""
    for ln in src.read_text().splitlines():   # python-markdown needs a blank line before a list
        is_item = ln.lstrip().startswith(("* ", "- ")) or re.match(r"\s*\d+\. ", ln)
        prev_item = prev.lstrip().startswith(("* ", "- ")) or re.match(r"\s*\d+\. ", prev) or prev.startswith("  ")
        if is_item and prev.strip() and not prev_item:
            lines.append("")
        lines.append(ln)
        prev = ln
    html = markdown.markdown("\n".join(lines), extensions=["tables", "fenced_code", "md_in_html"])
    doc = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
    tmp = ROOT / "reports" / "_report.html"
    tmp.write_text(doc)
    from playwright.sync_api import sync_playwright
    exe = os.environ.get("CHROMIUM") or ("/opt/pw-browsers/chromium" if Path("/opt/pw-browsers/chromium").exists() else None)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        pg = b.new_page()
        pg.goto(tmp.resolve().as_uri())
        pg.pdf(path=str(out), format="A4", print_background=True)
        b.close()
    tmp.unlink()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", out.read_bytes()))
    print(f"{out} : {pages} pages")
    if pages > 6:
        sys.exit("technical report exceeds the 6-page cap")


if __name__ == "__main__":
    main()

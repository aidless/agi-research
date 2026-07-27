"""Convert thesis_draft_v1.0.md to PDF using fpdf2.

Simple but effective: parse markdown, render headings/paragraphs/code/tables.
"""
from fpdf import FPDF
import re
import sys

sys.stdout.reconfigure(line_buffering=True)

class ThesisPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 6, "Archimedes Project - Thesis Draft v1.0", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter(self, title):
        self.add_page()
        self.set_font("Helvetica", "B", 18)
        self.multi_cell(0, 10, title)
        self.ln(4)

    def section(self, title):
        self.set_font("Helvetica", "B", 14)
        self.multi_cell(0, 8, title)
        self.ln(2)
        self.set_font("Helvetica", "", 11)

    def subsection(self, title):
        self.set_font("Helvetica", "B", 12)
        self.multi_cell(0, 7, title)
        self.ln(1)
        self.set_font("Helvetica", "", 11)

    def body_text(self, text):
        self.set_font("Helvetica", "", 11)
        # Strip markdown
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def code_block(self, code):
        self.set_font("Courier", "", 9)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(0, 5, code, fill=True)
        self.ln(2)
        self.set_font("Helvetica", "", 11)

    def table_row(self, row, header=False):
        if header:
            self.set_font("Helvetica", "B", 10)
            self.set_fill_color(230, 230, 230)
        else:
            self.set_font("Helvetica", "", 10)
        col_width = (self.w - 20) / len(row)
        for cell in row:
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", str(cell))
            self.cell(col_width, 6, text[:50], border=1, fill=header)
        self.ln()
        self.set_font("Helvetica", "", 11)


def parse_markdown(text):
    """Parse markdown into structured sections."""
    lines = text.split("\n")
    blocks = []
    current_block = None

    for line in lines:
        if line.startswith("# "):
            if current_block:
                blocks.append(current_block)
            current_block = {"type": "chapter", "title": line[2:].strip(), "content": []}
        elif line.startswith("## "):
            if current_block:
                blocks.append(current_block)
            current_block = {"type": "section", "title": line[3:].strip(), "content": []}
        elif line.startswith("### "):
            if current_block:
                blocks.append(current_block)
            current_block = {"type": "subsection", "title": line[4:].strip(), "content": []}
        elif line.startswith("```"):
            if current_block and current_block.get("in_code"):
                current_block["content"].append("END_CODE")
                current_block["in_code"] = False
            elif current_block:
                current_block["content"].append("START_CODE")
                current_block["in_code"] = True
        elif current_block is not None:
            current_block["content"].append(line)

    if current_block:
        blocks.append(current_block)
    return blocks


def render_pdf(blocks, output_path):
    pdf = ThesisPDF()
    pdf.set_auto_page_break(auto=True, margin=18)

    for block in blocks:
        if block["type"] == "chapter":
            pdf.chapter(block["title"])
        elif block["type"] == "section":
            pdf.section(block["title"])
        elif block["type"] == "subsection":
            pdf.subsection(block["title"])

        # Process content
        content = "\n".join(block["content"])
        # Split into segments
        segments = re.split(r"(START_CODE[\s\S]*?END_CODE|\n\n)", content)
        for seg in segments:
            if seg.startswith("START_CODE"):
                code_text = seg.replace("START_CODE\n", "").replace("END_CODE", "").strip()
                # Indent code
                indented = "\n".join("    " + line for line in code_text.split("\n"))
                pdf.code_block(indented)
            elif seg == "\n\n":
                pass
            elif seg.strip():
                pdf.body_text(seg.strip())

    pdf.output(output_path)


if __name__ == "__main__":
    md_path = r"E:\agi-research\thesis_draft_v1.0.md"
    pdf_path = r"E:\agi-research\thesis_draft_v1.0.pdf"

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = parse_markdown(text)
    render_pdf(blocks, pdf_path)

    import os
    print(f"PDF written: {pdf_path}")
    print(f"Size: {os.path.getsize(pdf_path)} bytes")
    print(f"Blocks: {len(blocks)}")

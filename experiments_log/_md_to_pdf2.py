"""Convert thesis_draft_v1.0.md to PDF using fpdf2."""
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

    def section_title(self, level, title):
        if level == 0:  # chapter
            self.add_page()
            self.set_font("Helvetica", "B", 18)
            self.multi_cell(0, 10, title)
            self.ln(4)
            self.set_font("Helvetica", "", 11)
        elif level == 1:  # section
            self.set_font("Helvetica", "B", 14)
            self.multi_cell(0, 8, title)
            self.ln(2)
            self.set_font("Helvetica", "", 11)
        elif level == 2:  # subsection
            self.set_font("Helvetica", "B", 12)
            self.multi_cell(0, 7, title)
            self.ln(1)
            self.set_font("Helvetica", "", 11)

    def body_text(self, text):
        self.set_font("Helvetica", "", 11)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"`(.+?)`", r"\1", text)
        text = text.replace("\u2014", "--").replace("\u2013", "-")
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        self.multi_cell(0, 6, text)
        self.ln(2)

    def code_block(self, code):
        self.set_font("Courier", "", 9)
        self.set_fill_color(245, 245, 245)
        indented = "\n".join("    " + line for line in code.split("\n"))
        self.multi_cell(0, 5, indented, fill=True)
        self.ln(2)
        self.set_font("Helvetica", "", 11)


def parse_blocks(text):
    lines = text.split("\n")
    blocks = []
    current = None
    in_code = False

    for line in lines:
        if line.startswith("# "):
            if current:
                blocks.append(current)
            current = {"level": 0, "title": line[2:].strip(), "content": []}
        elif line.startswith("## "):
            if current:
                blocks.append(current)
            current = {"level": 1, "title": line[3:].strip(), "content": []}
        elif line.startswith("### "):
            if current:
                blocks.append(current)
            current = {"level": 2, "title": line[4:].strip(), "content": []}
        elif line.startswith("```"):
            if current is None:
                current = {"level": 1, "title": "Code", "content": []}
            if in_code:
                current["content"].append("END_CODE")
                in_code = False
            else:
                current["content"].append("START_CODE")
                in_code = True
        elif current is not None:
            current["content"].append(line)

    if current:
        blocks.append(current)
    return blocks


def render(blocks, output_path):
    pdf = ThesisPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    for block in blocks:
        pdf.section_title(block["level"], block["title"])
        content = "\n".join(block["content"])

        # Split on code blocks
        segments = re.split(r"(START_CODE.*?END_CODE)", content, flags=re.DOTALL)
        for seg in segments:
            if seg.startswith("START_CODE"):
                code = seg.replace("START_CODE\n", "").replace("END_CODE", "").strip()
                if code:
                    pdf.code_block(code)
            elif seg.strip():
                pdf.body_text(seg.strip())

    pdf.output(output_path)


if __name__ == "__main__":
    md_path = r"E:\agi-research\thesis_draft_v1.0.md"
    pdf_path = r"E:\agi-research\thesis_draft_v1.0.pdf"

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = parse_blocks(text)
    render(blocks, pdf_path)

    import os
    print(f"PDF written: {pdf_path}")
    print(f"Size: {os.path.getsize(pdf_path)} bytes")
    print(f"Blocks: {len(blocks)}")
    print(f"Pages: estimated ~{len(blocks) * 1.5:.0f}")

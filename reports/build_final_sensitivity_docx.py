"""Typeset the standalone sensitivity section from its paper text."""
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

root = Path(__file__).resolve().parents[1]
source = root / "reports/三问终稿模型敏感性分析.md"
doc = Document()
section = doc.sections[0]
section.page_height, section.page_width = Cm(29.7), Cm(21)
section.top_margin, section.bottom_margin = Cm(2.0), Cm(2.0)
section.left_margin = section.right_margin = Cm(2.35)
normal = doc.styles["Normal"]
for name, size in [("Normal", 11), ("Title", 17), ("Heading 2", 13)]:
    style = doc.styles[name]
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体" if name=="Normal" else "黑体")
    if name != "Normal":
        style.font.bold = True
        border = style.element.get_or_add_pPr().find(qn("w:pBdr"))
        if border is not None:
            border.getparent().remove(border)
normal.paragraph_format.line_spacing = 1.16
normal.paragraph_format.space_after = Pt(4)

def table(lines):
    rows = [[item.strip() for item in line.strip("|").split("|")] for line in lines if "---" not in line]
    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom"):
        b = OxmlElement("w:"+edge)
        for prop, value in [("val", "single"), ("sz", "8"), ("color", "444444")]:
            b.set(qn("w:"+prop), value)
        borders.append(b)
    t._tbl.tblPr.append(borders)
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        for cell, value in zip(cells, row):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.05
            r = p.add_run(value)
            r.bold, r.font.size = (i==0), Pt(10)
            if i == 0:
                props = OxmlElement("w:tcBorders")
                b = OxmlElement("w:bottom")
                b.set(qn("w:val"), "single")
                b.set(qn("w:sz"), "4")
                b.set(qn("w:color"), "777777")
                props.append(b)
                cell._tc.get_or_add_tcPr().append(props)

for block in source.read_text(encoding="utf8").strip().split("\n\n"):
    if block.startswith("## "):
        p = doc.add_paragraph(block[3:], "Title")
        p.paragraph_format.space_after = Pt(5)
    elif block.startswith("### "):
        p = doc.add_paragraph(block[4:], "Heading 2")
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(6), Pt(3)
        if block.startswith("### 服务"):
            p.paragraph_format.page_break_before = True
    elif block.startswith("|"):
        table(block.splitlines())
    elif block.startswith("!["):
        picture = re.search(r"\]\((.*)\)$", block)[1]
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(source.parent/picture), width=Cm(15))
    elif block.startswith("**") and block.endswith("**"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(block.strip("*"))
        r.bold, r.font.size = True, Pt(10)
    else:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Pt(22)
        for i, part in enumerate(block.split("**")):
            p.add_run(part).bold = bool(i % 2)
doc.core_properties.title = "敏感性分析"
doc.save(source.with_suffix(".docx"))

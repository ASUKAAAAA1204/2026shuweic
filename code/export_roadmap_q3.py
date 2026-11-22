"""Export the revised DrawIO text over the existing vector layout, with pixel checks."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("durationl1", "dispatchh", "dispatchl0", "dispatchl1", "dispatchl2",
          "q3result1", "q3result2", "q3result3")


def box(cells, key):
    cell = cells[key]
    geom = cell.find("mxGeometry")
    x, y, w, h = (float(geom.get(name, 0)) for name in ("x","y","width","height"))
    parent = cells.get(cell.get("parent"))
    while parent is not None and parent.find("mxGeometry") is not None:
        g = parent.find("mxGeometry")
        x += float(g.get("x",0))
        y += float(g.get("y",0))
        parent = cells.get(parent.get("parent"))
    return fitz.Rect(x,y,x+w,y+h)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"figures/fig_roadmap.drawio")
    parser.add_argument("--template",type=Path,default=ROOT/"figures/fig_roadmap.pdf")
    parser.add_argument("--output",type=Path,default=ROOT/"figures")
    parser.add_argument("--fonts",type=Path,default=Path("C:/Windows/Fonts"))
    opt = parser.parse_args()
    cells = {item.get("id"):item for item in ET.parse(opt.source).iter("mxCell")}
    fonts = {str(i):fitz.Font(fontfile=str(opt.fonts/name)) for i,name in enumerate(("msyh.ttc","msyhbd.ttc"))}
    areas = [box(cells,key) for key in LABELS[:5]]
    # The result panel now has three lines in the original two-line footprint.
    bounds = box(cells,"q3")
    areas.append(fitz.Rect(bounds.x0+15,bounds.y0+463,bounds.x1-10,bounds.y1-1))
    scale = fitz.Matrix(1.5,1.5)
    opt.output.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=opt.output) as tmp:
        staging = Path(tmp)
        with fitz.open(opt.template) as doc:
            page = doc[0]
            if len(doc)!=1 or page.rect!=fitz.Rect(0,0,1180,805):
                raise ValueError("Expected the original single-page roadmap layout")
            before = page.get_pixmap(matrix=scale,alpha=False)
            for area in areas:
                page.add_redact_annot(area,fill=False,cross_out=False)
            page.apply_redactions(images=0,graphics=0,text=0)
            for key in LABELS:
                node = cells[key]
                rect = box(cells,key)
                style = dict(s.split("=",1) for s in node.get("style").split(";") if "=" in s)
                font = fonts[style["fontStyle"]]
                size = float(style["fontSize"])
                rgb = tuple(b/255 for b in bytes.fromhex(style["fontColor"][1:]))
                text = node.get("value")
                if font.text_length(text,fontsize=size)>rect.width:
                    raise ValueError(f"Label exceeds its assigned width: {key}")
                baseline = (rect.y0+rect.y1+(font.ascender+font.descender)*size)/2
                writer = fitz.TextWriter(page.rect)
                writer.append((rect.x0,baseline),text,font=font,fontsize=size)
                writer.write_text(page,color=rgb)
            doc.subset_fonts()
            doc.save(staging/"fig_roadmap.pdf",deflate=True,garbage=4)
        with fitz.open(staging/"fig_roadmap.pdf") as doc:
            page = doc[0]
            pix = page.get_pixmap(matrix=scale,alpha=False)
            shape = (pix.height,pix.width,pix.n)
            changed = np.any(np.frombuffer(before.samples,np.uint8).reshape(shape) !=
                             np.frombuffer(pix.samples,np.uint8).reshape(shape),axis=2)
            for rect in areas:
                ir = (rect*scale).irect
                changed[ir.y0-1:ir.y1+1,ir.x0-1:ir.x1+1] = False
            if changed.any():
                raise ValueError(f"{changed.sum()} changed pixels outside the requested areas")
            normalized = "".join(page.get_text().split())
            for key in LABELS:
                if "".join(cells[key].get("value").split()) not in normalized:
                    raise ValueError(f"Missing PDF label: {key}")
            pix.save(staging/"fig_roadmap.png")
        for suffix in ("pdf","png"):
            name = f"fig_roadmap.{suffix}"
            (staging/name).replace(opt.output/name)
    print(f"{pix.width}x{pix.height}; all eight labels verified; zero changed pixels outside edited regions")


if __name__ == "__main__":
    main()

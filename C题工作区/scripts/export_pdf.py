from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from xml.sax.saxutils import escape
import io
root=Path(__file__).resolve().parents[1];out=root/'中文题面'
pdfmetrics.registerFont(TTFont('Chinese','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='CN',fontName='Chinese',fontSize=10,leading=16,spaceAfter=7,wordWrap='CJK'))
for name,size in [('Title',19),('Heading1',15),('Heading2',12),('Heading3',11)]:
 styles[name].fontName='Chinese';styles[name].fontSize=size;styles[name].leading=size*1.4;styles[name].wordWrap='CJK'
d=Document(out/'2026数维杯C题_中文翻译.docx');story=[]
for b in d.element.body:
 if b.tag==qn('w:p'):
  txt=''.join(b.itertext()) if False else ''.join(x.text or '' for x in b.iter(qn('w:t')))
  for blip in b.iter(qn('a:blip')):
   rid=blip.get(qn('r:embed'));blob=d.part.related_parts[rid].blob
   im=Image(io.BytesIO(blob));ratio=min(475/im.imageWidth,180/im.imageHeight);im.drawWidth=im.imageWidth*ratio;im.drawHeight=im.imageHeight*ratio;story.append(im);story.append(Spacer(1,5))
  if txt:
   pstyle=b.find(qn('w:pPr'));sid=''
   if pstyle is not None:
    el=pstyle.find(qn('w:pStyle'));sid=el.get(qn('w:val')) if el is not None else ''
   sn=sid if sid in ['Title','Heading1','Heading2','Heading3'] else 'CN'
   story.append(Paragraph(escape(txt),styles[sn]))
 elif b.tag==qn('w:tbl'):
  data=[]
  for row in b.findall(qn('w:tr')):
   data.append([Paragraph(escape(''.join(x.text or '' for x in c.iter(qn('w:t')))),styles['CN']) for c in row.findall(qn('w:tc'))])
  t=Table(data,colWidths=[90,130,255],repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DBEAF0')),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#B7C9D0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6)]));story.extend([t,Spacer(1,8)])
def footer(c,d):
 c.setFont('Chinese',8);c.setFillColor(colors.HexColor('#666666'));c.drawString(50,25,'数维杯C题 · 中文工作译本');c.drawRightString(545,25,str(d.page))
pdf=out/'2026数维杯C题_中文翻译.pdf';SimpleDocTemplate(str(pdf),pagesize=(595.28,841.89),leftMargin=55,rightMargin=55,topMargin=40,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
print(pdf)

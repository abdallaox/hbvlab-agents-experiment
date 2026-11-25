"""Assemble the EMS submission files in submission/:
  highlights.docx      separate editable highlights file (from the highlights block in paper/main.tex)
  Figure_1.pdf ...     each figure as a separate vector file, numbered in order of appearance (Figure 2, the example
                       tool reply, is a text graphic embedded in the LaTeX and is also exported as a page crop)
  cover_letter.docx    draft cover letter (placeholders in [brackets] for the authors to complete)
  graphical_abstract.* written by scripts/graphical_abstract.py
Run after building paper/main.pdf.
"""
import os
import re
import shutil

from docx import Document
from docx.shared import Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "submission")
os.makedirs(OUT, exist_ok=True)
tex = open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()


def plain(s):
    s = s.replace(r"\_", "_").replace(r"\$", "$").replace(r"\%", "%").replace(r"\,", " ")
    return re.sub(r"\s+", " ", s).strip()


# highlights
block = tex[tex.index(r"\begin{highlights}"):tex.index(r"\end{highlights}")]
items = [plain(x) for x in block.split(r"\item")[1:]]
doc = Document()
doc.add_heading("Highlights", level=1)
for it in items:
    assert len(it) <= 85, (len(it), it)
    doc.add_paragraph(it, style="List Bullet")
doc.save(os.path.join(OUT, "highlights.docx"))
print("highlights:", [len(i) for i in items])

# figures in order of appearance
order = re.findall(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}|\\input\{(listing_runmodel)\.tex\}", tex)
n = 0
for fig, listing in order:
    n += 1
    if fig:
        shutil.copy(os.path.join(ROOT, "paper", "figures", fig), os.path.join(OUT, f"Figure_{n}.pdf"))
        print(f"Figure_{n}.pdf <- {fig}")
    else:
        print(f"Figure_{n}: text graphic (listing), embedded in the LaTeX source")

# cover letter
doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(11)
for para in [
    "[Date]",
    "Editor-in-Chief, Environmental Modelling & Software",
    "Dear Editor,",
    "We submit the research article \"HBV_Lab: an open HBV model that AI agents can set up, calibrate and diagnose "
    "through the Model Context Protocol\" for consideration in Environmental Modelling & Software.",
    "Large language model agents can now call software tools, but whether they can calibrate a process-based model "
    "reliably, and whether their work can be trusted, has not been tested at scale. The paper presents HBV_Lab, an open "
    "Python implementation of the HBV model with a Model Context Protocol (MCP) server that exposes its functions to "
    "agents as typed tools. We first show that the package is credible: calibrated automatically on 661 CAMELS-GB "
    "catchments it reaches a median test NSE of 0.82, level with a published HBV calibration and above four FUSE "
    "models. We then let an AI agent calibrate HBV through the MCP server on 95 representative catchments, with only the "
    "tools, the training data and a fixed prompt, and audit every run. The agent matches automatic calibration while "
    "choosing its own set-up for each catchment and explaining its choices, at about US$0.11 per run.",
    "We believe the work fits the journal's interest in environmental software, model evaluation and artificial "
    "intelligence techniques. It reports model and software performance against data, documents the algorithms, and "
    "makes the software, the experiment code and every agent transcript openly available.",
    "The automatic calibration is compared with an HBV benchmark from our related study, [status and reference of the "
    "benchmark study], which is cited and not reused as a result of this paper. The manuscript has not been published "
    "and is not under consideration elsewhere. All authors have approved the submission. We used an AI assistant "
    "(Claude Code) in preparing the software and the manuscript, as declared in the manuscript.",
    "[Optional: suggested reviewers with affiliations and emails.]",
    "Yours sincerely,",
    "[Corresponding author name, affiliation, email]",
]:
    doc.add_paragraph(para)
doc.save(os.path.join(OUT, "cover_letter.docx"))

# manuscript: PDF for reading plus the editable LaTeX source the journal requires
shutil.copy(os.path.join(ROOT, "paper", "main.pdf"), os.path.join(OUT, "manuscript.pdf"))
src = os.path.join(OUT, "manuscript_source")
if os.path.exists(src):
    shutil.rmtree(src)
os.makedirs(os.path.join(src, "figures"))
for f in ("main.tex", "main.bbl", "references.bib", "appendix_prompt.tex", "listing_runmodel.tex"):
    shutil.copy(os.path.join(ROOT, "paper", f), os.path.join(src, f))
for f in os.listdir(os.path.join(ROOT, "paper", "figures")):
    if f.endswith(".pdf"):
        shutil.copy(os.path.join(ROOT, "paper", "figures", f), os.path.join(src, "figures", f))
print("wrote", OUT)

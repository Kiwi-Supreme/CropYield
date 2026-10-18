"""Convert a '# %%' percent-format script into a Jupyter notebook and (optionally) execute it.

usage: python tools/py2nb.py nb_src/01_x.py [--run]
Markdown cells are written as  '# %% [markdown]'  followed by '# ' prefixed lines.
"""
import sys, re, pathlib
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

src = pathlib.Path(sys.argv[1])
text = src.read_text(encoding="utf-8")
blocks = re.split(r"^# %%", text, flags=re.M)[1:]
cells = []
for b in blocks:
    head, _, body = b.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in head:
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
        cells.append(new_markdown_cell(md))
    else:
        cells.append(new_code_cell(body))
nb = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
out = pathlib.Path("notebooks") / (src.stem + ".ipynb")
out.parent.mkdir(exist_ok=True)
if "--run" in sys.argv:
    from nbconvert.preprocessors import ExecutePreprocessor
    ExecutePreprocessor(timeout=3600, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}).preprocess(nb, {"metadata": {"path": "notebooks"}})
nbformat.write(nb, out)
print("wrote", out)

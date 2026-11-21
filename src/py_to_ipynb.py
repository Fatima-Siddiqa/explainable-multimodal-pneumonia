"""Convert a '# %%' percent-format script into a Jupyter notebook. Usage: python py_to_ipynb.py in.py out.ipynb"""
import json, re, sys
src = open(sys.argv[1], encoding="utf-8").read()
parts = re.split(r"^# %%(.*)$", src, flags=re.M)
cells = []
for i in range(1, len(parts), 2):
    tag, body = parts[i], parts[i + 1].strip("\n")
    if "[markdown]" in tag:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body.splitlines(True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
print(len(cells), "cells ->", sys.argv[2])

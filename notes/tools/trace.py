#!/usr/bin/env python3
"""Lis le long d'une trace : l'ordre d'execution = l'ordre de lecture.

  uv run notes/tools/trace.py scratch.py     # trace un fichier
  uv run notes/tools/trace.py "expr"         # ou une expression inline
  (python tout court marche pareil)

DEV=PYTHON par defaut (surchargeable : DEV=CUDA uv run ...).
Trace le tinygrad DU REPO ou vit ce script — jamais celui d'un venv/PyPI.
"""
import sys, os, subprocess
from collections import OrderedDict

# self-location : notes/tools/trace.py -> racine du fork ; sinon cwd
HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.dirname(os.path.dirname(HERE)), os.getcwd()):
    if os.path.isdir(os.path.join(cand, "tinygrad")):
        REPO = cand; break
else:
    sys.exit("repo tinygrad introuvable : lance depuis le fork, ou place-moi dans notes/tools/")
sys.path.insert(0, REPO)                  # le code source du repo gagne sur tout venv
os.environ.setdefault("DEV", "PYTHON")    # AVANT l'import tinygrad

ROOT = os.path.join(REPO, "tinygrad")
seen, counts, n = OrderedDict(), {}, [0]

def prof(frame, event, arg):
    if event != "call": return
    fn = frame.f_code.co_filename
    if not fn.startswith(ROOT) or "autogen" in fn: return
    key = (os.path.relpath(fn, REPO), frame.f_code.co_name)
    counts[key] = counts.get(key, 0) + 1
    if key not in seen:
        n[0] += 1
        seen[key] = n[0]

target = sys.argv[1] if len(sys.argv) > 1 else "(Tensor.ones(4) + 2).sum().item()"

from tinygrad import Tensor, dtypes  # noqa
import tinygrad, tinygrad.nn as nn   # noqa
Tensor.ones(4).contiguous().tolist()  # prechauffe import/device, hors trace

if os.path.isfile(target):
    src = open(target).read()
    g = {"__name__": "__main__", "__file__": os.path.abspath(target)}
    sys.argv = sys.argv[1:]           # le fichier trace recoit ses propres args
    sys.setprofile(prof)
    exec(compile(src, target, "exec"), g)
    sys.setprofile(None)
    out = "(fichier execute)"
else:
    sys.setprofile(prof)
    out = eval(target)
    sys.setprofile(None)

try: pin = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
except Exception: pin = "?"

print(f"\ncible   : {target}")
print(f"resultat: {out}")
print(f"pin     : {pin}  ({REPO})\n")

by_file = OrderedDict()
for (f, name), rank in seen.items():
    by_file.setdefault(f, []).append((rank, name, counts[(f, name)]))

total = sum(counts.values())
print(f"{len(seen)} fonctions, {len(by_file)} fichiers, {total} appels\n")
print("  #  fichier                                appels   fonctions chaudes")
print("  -  -------                                ------   -----------------")
for f, fns in sorted(by_file.items(), key=lambda kv: min(r for r, _, _ in kv[1])):
    first = min(r for r, _, _ in fns)
    tot = sum(c for _, _, c in fns)
    hot = ", ".join(nm for _, nm, _ in sorted(fns, key=lambda x: -x[2])[:3])
    print(f"{first:3d}  {f:38s} {tot:6d} {100*tot/total:4.0f}%  {hot}")

print("\n^ ordre de premier appel = ordre de lecture de la seance.")

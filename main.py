from grok import train, SEED, NSEEDS, setfix
from plot import plot
import json, pathlib, random

def main(path:str|pathlib.Path):
    runs = []
    for _ in range(SEED, SEED+NSEEDS):
        runs.append(train(random.randint(1, int(1e10))))
    with open(pathlib.Path(path), "w") as f: json.dump(runs, f)

for f, p in zip([0, 1], (pathes:=["master.json", "fix.json"])): setfix(f); main(p)
plot(pathes)



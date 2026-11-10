import sys, json
import matplotlib.pyplot as plt

def plot(pathes: str | list[str]):
  pathes = list(pathes) if isinstance(pathes, str) else pathes
  fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
  for k, path in enumerate(pathes):
    with open(path) as f: d = json.load(f)
    for dic in d:
      for ax, key in zip(axes, list(dic.keys())[1:]): ax.plot(dic["step"], dic[key], color=f"C{1-k}", alpha=.6, lw=3)
  axes[1].set_yscale("log")
  for ax, title in zip(axes, ("mean logit margin (train)", "training loss", "test accuracy")): ax.set_title(title); ax.set_xlabel("epoch")
  axes[0].legend()
  plt.tight_layout()
  plt.show()
# plt.savefig(sys.argv[-1])

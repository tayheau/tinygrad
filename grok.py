from collections import defaultdict
import sys, math, json
from itertools import product
from tinygrad import Tensor, TinyJit, nn
from tinygrad.helpers import getenv, Context, tqdm

P, W = getenv("P", 31), getenv("W", 64)
LR, B1, B2, EPS, WD = getenv("LR", 1e-3),getenv("B1", .9),getenv("B2", .98),getenv("EPS", 1e-8),getenv("WD", 1.0),
SEED, TRPROP = getenv("SEED", 0), getenv("TRPROP", 0.5)
STEPS= getenv("STEPS", 10_000)
NSEEDS = getenv("NSEEDS", 1)
FIX = getenv("FIX", 0)
def setfix(v:int):
  global FIX; FIX = v

x = list(product(range(P), range(P)))
AB, Y = Tensor.stack(Tensor([a[0] for a in x]), Tensor([b[1] for b in x]), dim=1), Tensor([(a+b)%P for (a, b) in x])

def getstep(model, opt, X, Y):
  @TinyJit
  @Context(TRAINING=1)
  def step() -> Tensor:
    opt.zero_grad()
    loss = loss_fn(model(X), Y).backward()
    return loss.realize(*opt.schedule_step())
  return step

def getlogit(model, Xtr, Ytr, Xte, Yte):
  @TinyJit
  def logits() -> tuple[Tensor, Tensor, Tensor]:
    z = model(Xtr)
    return loss_fn(z, Ytr).realize(), logitmargin(z, Ytr).mean().realize(), (model(Xte).argmax(1)==Yte).float().mean().realize()
  return logits


# 1layer 4head width64 ReLU transformer without LayerNorm on (a+b)%31 (0.5 training) 10k epoch AdamW | theta 1e-3 | b_1 .9 | b_2 .98 | lambda 1 | epsilon 1e-8

class Transformer():
  def __init__(self):
    self.embedding, self.pos = nn.Embedding(P+2, W), nn.Embedding(4, W)
    self.q, self.k, self.v, self.o = (nn.Linear(W, W, bias=False ) for _ in range(4))
    self.m1, self.m2, self.out = nn.Linear(W, W*4), nn.Linear(W*4, W), nn.Linear(W, P, bias=False)
  def __call__(self, ab:Tensor) -> Tensor:
    a , b = ab[:, :1], ab[:, 1:]
    tok = a.cat(a.full_like(P), b, a.full_like(P+1), dim=1); ts = tok.shape[1]
    x = self.embedding(tok) + self.pos(Tensor.arange(ts))
    x = x + self.o(self.q(x).reshape(-1, ts, 4, W//4).transpose(1, 2).scaled_dot_product_attention(
          self.k(x).reshape(-1, ts, 4, W//4).transpose(1, 2), self.v(x).reshape(-1, ts, 4, W//4).transpose(1, 2)
        ).transpose(1, 2).reshape(-1, ts, W))
    x = x + self.m2(self.m1(x).relu())
    return self.out(x[:, -1])


def fixed_cross_entropy(z:Tensor, y:Tensor) -> Tensor:
  is_t = y.one_hot(z.shape[-1]).bool()
  z_t = is_t.where(z, 0).sum(-1, keepdim=True)
  return is_t.where(0, z - z_t).logsumexp(-1).mean()

def loss_fn(z:Tensor, y:Tensor) -> Tensor: return fixed_cross_entropy(z, y) if FIX else z.cross_entropy(y)

def logitmargin(logits:Tensor, Y:Tensor)->Tensor:
  is_t = Y.one_hot(logits.shape[1]).bool()
  z_t = is_t.where(logits, 0).sum(1, keepdim=True)
  return -is_t.where(-math.inf, logits - z_t).logsumexp(axis=1)

def train(seed:int)->defaultdict[str, list]:
  Tensor.manual_seed(seed)
  idx = Tensor.randperm(AB.size(0))
  tr, te = idx[:int(AB.size(0) * TRPROP)], idx[int(AB.size(0) * TRPROP):]
  ABtrain, Ytrain = AB[tr], Y[tr]
  ABtest, Ytest = AB[te], Y[te]
  model = Transformer()
  params = nn.state.get_parameters(model)
  opt = nn.optim.AdamW(params, lr=LR, b1=B1, b2=B2, eps=EPS, weight_decay=WD, fused=False)
  step = getstep(model, opt, ABtrain, Ytrain)
  logits = getlogit(model, ABtrain, Ytrain, ABtest, Ytest)

  log = defaultdict(list)
  for i in tqdm(range(STEPS+1), desc=f"{seed-SEED+1}/{NSEEDS}"):
    if i % 100 == 0:
      loss, m, acc = logits()
      log["step"].append(i); log["margin"].append(m.item())   # marge moyenne sur l'entraînement
      log["loss"].append(loss.item()); log["acc"].append(acc.item())
    if i < STEPS: step()
  return log


if __name__=="__main__":
  runs = []
  for seed in range(SEED, SEED+NSEEDS):
    runs.append(train(seed))
    with open(sys.argv[1], "w") as f: json.dump(runs, f)

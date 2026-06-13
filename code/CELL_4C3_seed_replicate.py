# CELL 4C-3 -- seed replicate of the w=2.5 dose (Front 4C), own-seed baseline.
# RUN ORDER on a fresh GPU runtime: 4C-1b (unchanged, W=2.5) -> FIX-EGO -> this.
# First run: SEED, W = 3, 2.5   Second run (other notebook): SEED, W = 4, 2.5
# Outputs are seed-keyed; nothing overwrites the seed-2 artifacts.
SEED, W = 3, 2.5

import os, sys, glob, json, random, importlib, numpy as np, pandas as pd, torch
import torch.nn as nn
from google.colab import drive
if not os.path.ismount("/content/drive"): drive.mount("/content/drive")
GD = "/content/drive/MyDrive/Kaggle/gnn_outputs"
CACHE = "/content/drive/MyDrive/Kaggle/aml_cache"
assert torch.cuda.is_available(), "no GPU attached"
device = torch.device("cuda")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
ANCHOR10S = {3: 2.5289, 4: 2.4104}   # locked own-seed baseline R@10%

os.chdir("/content/Multi-GNN")
if "/content/Multi-GNN" not in sys.path: sys.path.insert(0, "/content/Multi-GNN")
import train_util; importlib.reload(train_util)
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders
from data_loading import get_data
from torch_geometric.nn import to_hetero
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import degree
from sklearn.metrics import f1_score, average_precision_score
import models
try:
    from training import get_model
    print("[import] get_model from training.py")
except Exception as _e:
    _src = open("training.py").read()
    _i = _src.find("def get_model"); assert _i >= 0
    _j = _src.find("\ndef ", _i + 1)
    _g = {"torch": torch, "Data": Data, "HeteroData": HeteroData, "degree": degree}
    _g.update({k: getattr(models, k) for k in dir(models) if not k.startswith("_")})
    _ns = {}
    exec(compile(_src[_i:(_j if _j > 0 else len(_src))], "training.get_model", "exec"), _g, _ns)
    get_model = _ns["get_model"]
    print(f"[import] get_model exec'd from source ({type(_e).__name__})")

cands = [c for c in (glob.glob("/content/aml_data/*/formatted_transactions.csv")
                     or glob.glob("/content/**/formatted_transactions.csv", recursive=True))
         if "/content/drive/" not in c]
assert len(cands) == 1, f"expected one staged csv, found {cands}"
DATA_DIR = os.path.dirname(cands[0]); DATA_NAME = os.path.basename(DATA_DIR)
data_config = json.load(open("data_config.json"))
data_config["paths"]["aml_data"] = os.path.dirname(DATA_DIR)
print(f"[data] {cands[0]} | data={DATA_NAME}")

class A: pass
args = A()
args.model = "gin"; args.data = DATA_NAME
args.emlps = True; args.reverse_mp = True; args.ego = True; args.ports = True
args.tds = False; args.seed = SEED; args.tqdm = False
args.num_neighs = [100, 100]
args.batch_size = int(extract_param("batch_size", args) or 8192)
class C: pass
config = C()
for k in ("lr", "n_hidden", "n_gnn_layers", "dropout", "final_dropout", "w_ce1", "w_ce2"):
    setattr(config, k, extract_param(k, args))
config.epochs = 40; config.batch_size = args.batch_size
for k in ("lr", "n_hidden", "n_gnn_layers", "w_ce1", "w_ce2"):
    assert getattr(config, k) is not None, f"model_settings.json missing {k}"
print(f"[config] seed {SEED} | epochs 40 | batch {args.batch_size} | lr {config.lr:.6f}")

print("[get_data] building graphs (silent 15-20 min) ...")
tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
N_TR = tr_data["node", "to", "node"].edge_index.shape[1]
assert N_TR == 3_248_921 and int(val_inds.min()) == 3_248_921 and int(te_inds.min()) == 4_214_445
print(f"[gates] N_TR {N_TR:,} | splits OK")
HI_VAL = (3_248_921, 4_214_444); HI_TEST = (4_214_445, 5_078_344)

W_EDGE = np.load(f"{CACHE}/w_edge_4c_w{W}.npy")
assert len(W_EDGE) == N_TR
W_EDGE = torch.tensor(W_EDGE, dtype=torch.float32, device=device)
nw = int((W_EDGE > 1).sum())
print(f"[loss] edge w={W} on {nw:,} train edges ({100*nw/N_TR:.2f}%)")
assert nw == 1_133_570, "weighted-edge count mismatch"

y_hi = pd.read_csv(f"{CACHE}/Small_HI_formatted_transactions.csv",
                   usecols=["Is Laundering"])["Is Laundering"].values.astype("int8")
assert len(y_hi) == 5_078_345
def load_scores(path, lo, hi):
    d = pd.read_csv(path)
    ic = next((c for c in ("gid","EdgeID","te_global_idx","va_global_idx","global_idx","idx")
               if c in d.columns), None)
    assert ic is not None, f"{path}: no index col in {list(d.columns)}"
    v = d[ic].values.astype(np.int64)
    gid = v if (v.min() >= lo and v.max() <= hi) else v + lo
    pc = next(c for c in ("prob","score","p","pred") if c in d.columns)
    yc = next((c for c in ("y","label","Is Laundering") if c in d.columns), None)
    y = d[yc].values.astype("int8") if yc else y_hi[gid]
    out = pd.DataFrame({"gid": gid, "prob": d[pc].values, "y": y}).sort_values("gid")
    out = out.reset_index(drop=True)
    assert out.gid.min() >= lo and out.gid.max() <= hi and out.gid.is_unique, path
    return out

g_hi = np.load(f"{CACHE}/g_sorted_hismall.npy")
def Rpoint(al, y, g):
    inn = y == 0
    return ((al[inn & (g == 0)].mean() / al[inn & (g == 2)].mean())
            / (y[g == 0].mean() / y[g == 2].mean()))

# own-seed baseline from Drive
va_b = load_scores(f"{GD}/amlworld_gnn_val_scores_seed{SEED}.csv", *HI_VAL)
te_b = load_scores(f"{GD}/amlworld_gnn_test_scores_seed{SEED}.csv", *HI_TEST)
gb = g_hi[te_b["gid"].values]; yb = te_b["y"].values; pb = te_b["prob"].values
BASE = {}
for b in (0.01, 0.05, 0.10, 0.20):
    BASE[b] = Rpoint(pb >= np.quantile(va_b["prob"].values, 1 - b), yb, gb)
aup_base = average_precision_score(yb, pb)
anc = ANCHOR10S[SEED]
ok = abs(BASE[0.10] - anc) / anc < 1e-3
print(f"[baseline seed{SEED}] R@10% {BASE[0.10]:.4f} vs locked {anc:.4f} "
      f"{'[GATE OK]' if ok else '[GATE FAIL]'} | AUPRC {aup_base:.4f}")
assert ok

transform = AddEgoIds() if args.ego else None
add_arange_ids([tr_data, val_data, te_data])
tr_loader, val_loader, te_loader = get_loaders(
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)
sample_batch = next(iter(tr_loader))
model = get_model(sample_batch, config, args)
if args.reverse_mp: model = to_hetero(model, te_data.metadata(), aggr="mean")
model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=config.lr)
CLASS_W = torch.tensor([config.w_ce1, config.w_ce2], dtype=torch.float32, device=device)
ce_none = nn.CrossEntropyLoss(weight=CLASS_W, reduction="none")
CKPT = f"{GD}/checkpoint_4c_w{W}_seed{SEED}.tar"

def batch_mask_ids(batch, loader, inds):
    ii = inds.detach().cpu()
    bei = ii[batch["node", "to", "node"].input_id.detach().cpu()]
    bids = loader.data["node", "to", "node"].edge_attr.detach().cpu()[bei, 0]
    mask = torch.isin(batch["node", "to", "node"].edge_attr[:, 0].detach().cpu(), bids)
    ids = batch["node", "to", "node"].edge_attr[mask, 0].detach().cpu().long()
    batch["node", "to", "node"].edge_attr = batch["node", "to", "node"].edge_attr[:, 1:]
    batch["node", "rev_to", "node"].edge_attr = batch["node", "rev_to", "node"].edge_attr[:, 1:]
    return mask, ids

@torch.no_grad()
def eval_f1(loader, inds):
    model.eval(); P, Y = [], []
    for batch in loader:
        mask, _ = batch_mask_ids(batch, loader, inds)
        batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[("node","to","node")]
        P.append(out[mask].argmax(-1).cpu()); Y.append(batch["node","to","node"].y[mask].cpu())
    return f1_score(torch.cat(Y).numpy(), torch.cat(P).numpy())

best_val = 0.0
for epoch in range(config.epochs):
    model.train()
    for batch in tr_loader:
        opt.zero_grad()
        mask, ids = batch_mask_ids(batch, tr_loader, tr_inds)
        batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[("node","to","node")]
        pred = out[mask]; gt = batch["node","to","node"].y[mask]
        ew = W_EDGE[ids.to(device)]
        loss = (ew * ce_none(pred, gt)).sum() / (ew * CLASS_W[gt]).sum()
        loss.backward(); opt.step()
    vf1 = eval_f1(val_loader, val_inds)
    line = f"[epoch {epoch:02d}] val f1 {vf1:.4f}"
    if epoch >= 1 and vf1 > best_val:
        best_val = vf1
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, CKPT)
        line += "  [checkpoint saved]"
    print(line, flush=True)

print(f"[train done] best val f1 {best_val:.4f}")
model.load_state_dict(torch.load(CKPT, map_location=device)["model_state_dict"])

@torch.no_grad()
def capture(loader, inds):
    model.eval(); G, P, Y = [], [], []
    for batch in loader:
        mask, ids = batch_mask_ids(batch, loader, inds)
        batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[("node","to","node")]
        G.append(ids); P.append(out[mask].softmax(-1)[:, 1].cpu())
        Y.append(batch["node","to","node"].y[mask].cpu())
    return (pd.DataFrame({"gid": torch.cat(G).numpy(), "prob": torch.cat(P).numpy(),
                          "y": torch.cat(Y).numpy().astype("int8")})
            .sort_values("gid").reset_index(drop=True))

print("[capture] val ..."); va = capture(val_loader, val_inds)
print("[capture] test ..."); te = capture(te_loader, te_inds)
va.to_csv(f"{GD}/amlworld_gnn_val_scores_4c_w{W}_seed{SEED}.csv", index=False)
te.to_csv(f"{GD}/amlworld_gnn_test_scores_4c_w{W}_seed{SEED}.csv", index=False)
print(f"[capture] val n {len(va):,} | test n {len(te):,}  saved")

y = te["y"].values; g = g_hi[te["gid"].values]; p = te["prob"].values
aup_4c = average_precision_score(y, p)
rows = []
for b in (0.01, 0.05, 0.10, 0.20):
    R = Rpoint(p >= np.quantile(va["prob"].values, 1 - b), y, g)
    rows.append(dict(seed=SEED, w=W, budget=b, R=R, base_R=BASE[b],
                     auprc_4c=aup_4c, auprc_base=aup_base))
    print(f"[ladder] R@{int(b*100)}% = {R:.4f}   (seed{SEED} baseline {BASE[b]:.4f})")
pd.DataFrame(rows).to_csv(f"{GD}/amlworld_gnn_4c_w{W}_seed{SEED}_ladder.csv", index=False)
print(f"[auprc] 4c {aup_4c:.4f} vs seed{SEED} baseline {aup_base:.4f} "
      f"({100*aup_4c/aup_base:.1f}%)")
print(f"[delta] R@10%: {rows[2]['R']:.3f} vs own-seed baseline {BASE[0.10]:.3f} "
      f"({rows[2]['R']-BASE[0.10]:+.3f})")
print("[saved] seed-keyed checkpoint + scores + ladder -> gnn_outputs")

#!/usr/bin/env python3
"""
End-to-end inference + monitoring pipeline for 3-stage Fractional Knapsack AINN.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monitor_helper import main as clrs_main
    from monitor_helper import run_inference as clrs_run_inference
except Exception as e:
    clrs_main = None
    clrs_run_inference = None
    _IMPORT_ERR = e

# --- MUST BE AT TOP, before any heavy imports ---
import os
import warnings
import logging

warnings.filterwarnings("ignore")  # or target specific categories below

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=all, 1=filter INFO, 2=+WARNING, 3=+ERROR
os.environ["GLOG_minloglevel"] = "3"       # for glog-based logs
os.environ["ABSL_LOGGING_VERBOSITY"] = "3" # absl verbosity

os.environ["JAX_LOG_LEVEL"] = "error"

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass

logging.getLogger().setLevel(logging.ERROR)
for name in ["absl", "jax", "tensorflow", "clrs"]:
    logging.getLogger(name).setLevel(logging.ERROR)


def build_fractional_mask(k_mask: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    B, L = k_mask.shape
    one_index = torch.argmax(k_mask, dim=1)  # [B]
    mask = torch.arange(L, device=k_mask.device).expand(B, L)
    full_mask = (mask <= one_index.unsqueeze(1)).float()
    return full_mask


class FractionalKnapsackTwoStage(nn.Module):
    def __init__(self, d_model: int = 32 * 4):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=1),
            num_layers=2
        )
        self.k_head = nn.Linear(d_model, 1)
        self.r_head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = x.shape
        x_input = x.unsqueeze(-1)            # [B, L, 1]
        h = self.embed(x_input)              # [B, L, D]
        h = h.permute(1, 0, 2)               # [L, B, D]
        h = self.encoder(h)                  # [L, B, D]
        h = F.relu(h.permute(1, 0, 2))       # [B, L, D]

        k_logits = self.k_head(h).squeeze(-1)                 # [B, L]
        k_idx = torch.argmax(k_logits, dim=1)                 # [B]
        k_mask = F.one_hot(k_idx, num_classes=L).float()      # [B, L]

        h_pooled = h.mean(dim=1)                              # [B, D]
        r = self.r_head(h_pooled).squeeze(-1)                 # [B]
        return k_mask, k_logits, r


class SimpleNN(nn.Module):
    # Used for stage1 ratio prediction
    def __init__(self, input_size: int = 10, hidden_size: int = 16 * 4, output_size: int = 5):
        super().__init__()
        assert input_size == 10 and output_size == 5

        self.transforms = [lambda x: x, torch.log, torch.exp, torch.sin]
        self.num_transforms = len(self.transforms)
        self.shared_gate_logits = nn.Parameter(torch.randn(self.num_transforms))

        self.shared_mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        outputs = []

        gate_idx = torch.argmax(self.shared_gate_logits).view(1, 1)
        gate_idx = gate_idx.expand(batch_size, 1)  # [B, 1]

        for i in range(5):
            wi = x[:, i].unsqueeze(1)
            pi = x[:, i + 5].unsqueeze(1)

            wi_trans = torch.cat([tf(wi) for tf in self.transforms], dim=1)
            pi_trans = torch.cat([tf(pi) for tf in self.transforms], dim=1)

            wi_selected = torch.gather(wi_trans, dim=1, index=gate_idx)
            pi_selected = torch.gather(pi_trans, dim=1, index=gate_idx)

            inp = torch.cat([wi_selected, pi_selected], dim=1)
            ri = self.shared_mlp(inp)
            outputs.append(ri)

        return torch.cat(outputs, dim=1)  # [B, 5]


class Stage2Encoder(nn.Module):
    def __init__(self, in_dim: int = 10, embed_dim: int = 16 * 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64 * 3),
            nn.LeakyReLU(),
            nn.Linear(64 * 3, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        logits = self.classifier(z).squeeze(-1)
        return z, logits


def keep_leftmost_one(k: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(k, dim=1)
    mask_first_one = (cumsum == 1) & (k == 1)
    return mask_first_one.to(k.dtype)


def shift_one_right_batchwise(k: torch.Tensor) -> torch.Tensor:
    B, N = k.shape
    device = k.device
    indices = torch.argmax(k, dim=1)
    not_at_end = indices < (N - 1)
    not_at_start = indices != 0
    shift_mask = not_at_end & not_at_start

    k_new = torch.zeros_like(k)
    row_idx = torch.arange(B, device=device)[shift_mask]
    col_idx = indices[shift_mask] + 1
    k_new[row_idx, col_idx] = 1

    row_idx_static = torch.arange(B, device=device)[~shift_mask]
    col_idx_static = indices[~shift_mask]
    k_new[row_idx_static, col_idx_static] = 1
    return k_new


class Stage3Encoder(nn.Module):
    def __init__(self, in_dim: int = 15, embed_dim: int = 16 * 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64 * 3),
            nn.LeakyReLU(),
            nn.Linear(64 * 3, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_half = x[:, :5]
        x_half = keep_leftmost_one(x_half)
        x_half2 = x[:, 5:10]
        x_half2 = shift_one_right_batchwise(x_half2)
        new_input = torch.cat((x_half, x_half2, x[:, 10:]), dim=1)
        z = self.encoder(new_input)
        logits = self.classifier(z).squeeze(-1)
        return z, logits


# -----------------------------
#  Stage1 curve monitor (per-item absolute difference)
# -----------------------------


import numpy as np

def comparative_consistency_pairwise(
    pred_ratio,
    curve_ratio
):
    pred = np.asarray(pred_ratio, dtype=np.float64)
    curve = np.asarray(curve_ratio, dtype=np.float64)
    n = pred.shape[0]
    
    for i in range(n):
        for j in range(i + 1, n):
            cd = curve[i] - curve[j]
            pd = pred[i] - pred[j]
            if cd * pd < 0:   # opposite signs -> comparative inconsistency
                return True
                
    return False



@dataclass
class Stage1Curve:
    a: float
    b: float
    d: float
    threshold: Optional[float] = None  # per-item abs diff threshold
    mse_threshold: Optional[float] = None  # optional legacy field

    def curve_pred(self, weights: np.ndarray, profits: np.ndarray) -> np.ndarray:
        """Curve prediction per item: a*log(|b*(p/w)|)+d"""
        w = np.asarray(weights, dtype=np.float64)
        p = np.asarray(profits, dtype=np.float64)
        ratio_gt = p / w #(w + 1e-12)
        return self.a * np.log(np.abs(self.b * ratio_gt)) + self.d

    def score_mse(self, weights: np.ndarray, profits: np.ndarray, pred_ratio: np.ndarray) -> float:
        """Mean squared residual between curve prediction and stage1 predicted ratio (legacy)."""
        f_pred = self.curve_pred(weights, profits)
        z = np.asarray(pred_ratio, dtype=np.float64)
        return float(np.mean((f_pred - z) ** 2))

    def per_item_flags(self, weights: np.ndarray, profits: np.ndarray, pred_ratio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (abs_diff[5], flags[5]) comparing curve_pred vs stage1 outputs."""
        f_pred = self.curve_pred(weights, profits)
        z = np.asarray(pred_ratio, dtype=np.float64)
        abs_diff = np.abs(f_pred - z)/np.abs(f_pred)
        if self.threshold is None:
            flags = np.zeros_like(abs_diff, dtype=bool)
        else:
            flags = abs_diff > float(self.threshold)
        global_flag = comparative_consistency_pairwise(f_pred, pred_ratio)
        return abs_diff, flags, global_flag


# -----------------------------
#  Helpers
# -----------------------------
def _parse_list(s: str, name: str, n: int = 5) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) != n:
        raise ValueError(f"{name} must have length {n}, got {len(parts)}: {parts}")
    return [float(x) for x in parts]


def _load_yaml_num_samples(config_name: str) -> Optional[int]:
    # Mirrors the training scripts: ./config/{config}.yaml
    try:
        import yaml  # lazy import
        cfg_path = os.path.join("config", f"{config_name}.yaml")
        if not os.path.exists(cfg_path):
            return None
        cfg = yaml.safe_load(open(cfg_path, "r"))
        return int(cfg["data"]["num_samples"])
    except Exception:
        return None


def _load_stage1_curve_from_json(path: str) -> Stage1Curve:
    with open(path, "r") as f:
        obj = json.load(f)
    # accept a few key names to be robust:
    # - preferred: threshold
    # - legacy: mse_threshold
    threshold = obj.get("threshold", obj.get("abs_threshold", None))
    mse_threshold = obj.get("mse_threshold", None)
    return Stage1Curve(
        a=float(obj["a"]),
        b=float(obj["b"]),
        d=float(obj["d"]),
        threshold=float(threshold) if threshold is not None else None,
        mse_threshold=float(mse_threshold) if mse_threshold is not None else None,
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _load_torch_ckpt(path: str, map_location: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


def _stable_argsort_desc(values: np.ndarray) -> np.ndarray:
    """Deterministic descending sort with tie-break by index (stable)."""
    values = np.asarray(values)
    idx = np.arange(values.shape[0])
    # lexsort uses last key as primary; so provide (idx, -values) => primary -values, tiebreak idx
    return np.lexsort((idx, -values))


def fractional_knapsack_gt(weights: np.ndarray, profits: np.ndarray, capacity: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Greedy optimal fractional knapsack:
      - sort by p/w descending
      - take full until capacity, then take fraction of next
    Returns (fractions[5] in original order, sort_indices, total_profit).
    """
    w = np.asarray(weights, dtype=np.float64)
    p = np.asarray(profits, dtype=np.float64)
    ratio = p / w #(w + 1e-12)
    order = _stable_argsort_desc(ratio)  # indices in original space

    remaining = float(capacity)
    frac = np.zeros_like(w, dtype=np.float64)
    for i in order:
        if remaining <= 1e-12:
            break
        if w[i] <= remaining + 1e-12:
            frac[i] = 1.0
            remaining -= float(w[i])
        else:
            frac[i] = remaining / float(w[i])
            remaining = 0.0
            break
    total = float(np.sum(p * frac))
    return frac.astype(np.float32), order.astype(np.int64), total


# -----------------------------
#  Core pipeline
# -----------------------------
@dataclass
class PipelineOutputs:
    # inputs
    ratio_gt: np.ndarray

    # stage1 (model)
    pred_ratio: np.ndarray

    # stage1 curve comparison
    curve_ratio: Optional[np.ndarray]
    stage1_abs_diff: Optional[np.ndarray]
    stage1_flags: Optional[np.ndarray]
    stage1_seq_flag: Optional[bool]
    stage1_ok: Optional[bool]            # True if no flags when threshold provided
    stage1_curve_mse: Optional[float]    # optional, for debugging/legacy

    # stage2 (model) + monitor
    sort_indices: np.ndarray
    stage2_logit: float
    stage2_prob: float
    stage2_ok: bool
    stage2_embed: np.ndarray

    # stage2 ground truth
    stage2_gt_indices: np.ndarray

    # stage3 (model) + monitor
    cumsum_order: np.ndarray
    k_mask: np.ndarray
    k_logits: np.ndarray
    stage3_logit: float
    stage3_prob: float
    stage3_ok: bool
    stage3_embed: np.ndarray

    # final (model)
    fractions: np.ndarray
    total_profit: float

    # stage3 ground truth (final selection)
    fractions_gt: np.ndarray
    total_profit_gt: float




def run_pipeline(
    weights: List[float],
    profits: List[float],
    capacity: float,
    stage1_model: SimpleNN,
    sorter_model,
    rng_key,
    stage3_model: FractionalKnapsackTwoStage,
    enc2: Stage2Encoder,
    enc3: Stage3Encoder,
    stage1_curve: Optional[Stage1Curve],
    device: torch.device,
) -> PipelineOutputs:
    # ---------- Inputs / ground truth for stage1 ----------
    w = np.asarray(weights, dtype=np.float32)
    v = np.asarray(profits, dtype=np.float32)
    ratio_gt = v / w

    # ---------- Stage 1 model ----------
    x1 = torch.tensor(np.concatenate([w, v])[None, :], dtype=torch.float, device=device)
    stage1_model.eval()
    with torch.no_grad():
        pred_ratio_t = stage1_model(x1)
    pred_ratio = pred_ratio_t.detach().cpu().numpy()[0]  # [5]

    # ---------- Stage 1 curve per-item comparison ----------
    curve_ratio = None
    abs_diff = None
    flags = None
    stage1_ok = None
    curve_mse = None
    if stage1_curve is not None:
        curve_ratio = stage1_curve.curve_pred(w, v).astype(np.float32)
        abs_diff, flags, gflag = stage1_curve.per_item_flags(w, v, pred_ratio)
        abs_diff = abs_diff.astype(np.float32)
        stage1_ok = ((np.sum(flags) + float(gflag)) == 0) #bool(np.all(~flags)) if stage1_curve.threshold is not None else None
        curve_mse = stage1_curve.score_mse(w, v, pred_ratio) if stage1_curve.mse_threshold is not None else None

    # ---------- Stage 2 GT (sorted result of input) ----------
    _frac_gt, stage2_gt_indices, _tot_gt = fractional_knapsack_gt(w, v, capacity)

    # ---------- Stage 2 model inference (CLRS sorting) ----------
    if clrs_run_inference is None:
        raise RuntimeError(
            f"monitor_helper import failed; cannot run stage2 sorting.\nOriginal error: {_IMPORT_ERR}"
        )
    ret_ind, _preds_sorting = clrs_run_inference(sorter_model, np.array(pred_ratio)[None, :], rng_key)
    ret_ind = np.asarray(ret_ind[0], dtype=np.int64)  # [5]

    # Stage2 monitor features: concat([pinf, argsort(-pred_ratio)])
    stage2_feat = np.concatenate([ret_ind, np.argsort(-1.0 * pred_ratio)], axis=0)[None, :]


    x2 = torch.tensor(stage2_feat, dtype=torch.float32, device=device)
    #x2_list.append(x2)
    enc2.eval()
    with torch.no_grad():
        z2, logit2_t = enc2(x2)
    logit2 = logit2_t.detach().cpu().numpy()[0]
    stage2_prob = _sigmoid(logit2)
    stage2_ok = logit2 >= 0.0
    stage2_embed = z2.detach().cpu().numpy()[0]

    # ---------- Stage 3 model (K/r and final fractions) ----------
    ret_ind_t = torch.tensor(ret_ind[None, :], dtype=torch.long, device=device)
    w_t = torch.tensor(w[None, :], dtype=torch.float32, device=device)

    reordered = torch.gather(torch.tensor(w)[None, :] / capacity, dim=1, index=torch.tensor(ret_ind[None, :]))
    cumsum_order_t = torch.cumsum(reordered, dim=1)                          # [1,5]

    stage3_model.eval()
    with torch.no_grad():
        k_mask_t, k_logits_t, _ = stage3_model(cumsum_order_t.to(device))
    # --- final fraction construction ---
    k_mask_cpu = k_mask_t.detach().cpu()
    B, L = k_mask_cpu.shape
    k1_index = torch.argmax(k_mask_cpu, dim=1)  # [B]
    kp1_index = k1_index + 1
    valid_k1 = kp1_index < L
    batch_indices = torch.arange(B, device=k_mask_cpu.device)

    cumsum_k1 = cumsum_order_t.detach().cpu()[batch_indices[valid_k1], k1_index[valid_k1]]
    reordered_k1 = reordered.detach().cpu()[batch_indices[valid_k1], kp1_index[valid_k1]]

    r = torch.zeros(B, device=k_mask_cpu.device)
    final_value = build_fractional_mask(k_mask_cpu, r)

    over_mask = cumsum_k1 > 1.0
    under_mask = ~over_mask

    if over_mask.any():
        r_over = 1.0 / cumsum_k1[over_mask]
        final_value[batch_indices[valid_k1][over_mask], k1_index[valid_k1][over_mask]] = r_over

    if under_mask.any():
        r_under = (1.0 - cumsum_k1[under_mask]) / (reordered_k1[under_mask]) # + 1e-12)
        final_value[batch_indices[valid_k1][under_mask], kp1_index[valid_k1][under_mask]] = r_under

    inverse_indices = torch.zeros_like(ret_ind_t.detach().cpu())
    inverse_indices.scatter_(1, ret_ind_t.detach().cpu(), torch.arange(ret_ind_t.size(1)).expand_as(ret_ind_t.detach().cpu()))
    final_value2 = torch.gather(final_value, dim=1, index=inverse_indices).numpy()[0]  # [5]
    fractions = final_value2.astype(np.float32)
    total_profit = float(np.sum(v * fractions))

    # Stage3 monitor features: concat([c>1, k_mask, k_logits])
    c_np = cumsum_order_t.detach().cpu().numpy()[0]
    k_np = k_mask_t.detach().cpu().numpy()[0]
    kl_np = k_logits_t.detach().cpu().numpy()[0]
    stage3_feat = np.concatenate([(c_np > 1.0).astype(np.float32), k_np.astype(np.float32), kl_np.astype(np.float32)], axis=0)[None, :]
    x3 = torch.tensor(stage3_feat, dtype=torch.float32, device=device)
    enc3.eval()
    with torch.no_grad():
        z3, logit3_t = enc3(x3)
    logit3 = float(logit3_t.detach().cpu().numpy()[0])
    stage3_prob = _sigmoid(logit3)
    stage3_ok = logit3 >= 0.0
    stage3_embed = z3.detach().cpu().numpy()[0]

    # Stage3 GT final selection (fractions)
    fractions_gt, _order_gt, total_profit_gt = fractional_knapsack_gt(w, v, capacity)

    return PipelineOutputs(
        ratio_gt=ratio_gt.astype(np.float32),
        pred_ratio=pred_ratio.astype(np.float32),

        curve_ratio=curve_ratio,
        stage1_abs_diff=abs_diff,
        stage1_flags=flags,
        stage1_seq_flag=gflag,
        stage1_ok=stage1_ok,
        stage1_curve_mse=curve_mse,

        sort_indices=ret_ind,
        stage2_logit=logit2,
        stage2_prob=stage2_prob,
        stage2_ok=stage2_ok,
        stage2_embed=stage2_embed,

        stage2_gt_indices=stage2_gt_indices,

        cumsum_order=c_np,
        k_mask=k_np,
        k_logits=kl_np,
        stage3_logit=logit3,
        stage3_prob=stage3_prob,
        stage3_ok=stage3_ok,
        stage3_embed=stage3_embed,

        fractions=fractions,
        total_profit=total_profit,

        fractions_gt=fractions_gt,
        total_profit_gt=total_profit_gt,
    )




def print_failure_banner(failures):
    if not failures:
        print("\n\033[92m" + "NO ANOMALY DETECTED".center(70) + "\033[0m\n")
        return

    msg = f"DETECTED POTENTIAL ANOMALY: {', '.join(failures).upper()}"
    line = "!" * max(70, len(msg) + 10)
    print("\n\033[91m" + line + "\033[0m")
    print("\033[91m" + msg.center(len(line)) + "\033[0m")




def main():
    parser = argparse.ArgumentParser("AINN end-to-end inference + monitoring")
    parser.add_argument("--weights", type=str, help="Comma-separated list of 5 weights (e.g., 2,5,4,7,1)")
    parser.add_argument("--profits", type=str, help="Comma-separated list of 5 profits (e.g., 10,8,7,15,3)")
    parser.add_argument("--capacity", type=float, default=15.0, help="Knapsack capacity (default 15)")
    parser.add_argument("--config", type=str, default="ainn_less_iter", help="Config name (used to locate checkpoints)")
    parser.add_argument("--num_samples", type=int, default=None, help="Override cfg['data']['num_samples'] (optional)")

    # checkpoints (default paths mimic training scripts)
    parser.add_argument("--stage1_ckpt", type=str, default=None, help="Path to stage1 model .pth (optional)")
    parser.add_argument("--stage2_ckpt", type=str, default=None, help="Path to stage2 sorter .pkl (optional)")
    parser.add_argument("--stage3_ckpt", type=str, default=None, help="Path to stage3 model .pth (optional)")
    parser.add_argument("--enc2_ckpt", type=str, default="ainn_model/encoder_stage2.pth", help="Path to stage2 encoder .pth")
    parser.add_argument("--enc3_ckpt", type=str, default="ainn_model/encoder_stage3.pth", help="Path to stage3 encoder .pth")
    parser.add_argument("--print_gt", action="store_true", help="If set, print ground-truth outputs for each stage.")

    # stage1 curve params JSON:
    #   {"a":..., "b":..., "d":..., "threshold": ...}
    parser.add_argument("--stage1_curve_json", type=str, default="ainn_model/stage1_curve.json",
                        help='Path to stage1 curve JSON. If present and includes "threshold", per-item flags are enabled. '
                             'If missing, stage1 curve monitoring is disabled.')

    args = parser.parse_args()

    if args.weights is None or args.profits is None:
        print("Enter weights (5 comma-separated): ", end="", flush=True)
        args.weights = sys.stdin.readline().strip()
        print("Enter profits (5 comma-separated): ", end="", flush=True)
        args.profits = sys.stdin.readline().strip()

    weights = _parse_list(args.weights, "weights", n=5)
    profits = _parse_list(args.profits, "profits", n=5)

    # resolve num_samples from yaml if not provided
    num_samples = args.num_samples
    if num_samples is None:
        num_samples = _load_yaml_num_samples(args.config)
    if num_samples is None:
        raise RuntimeError(
            "Could not determine num_samples. Provide --num_samples, or ensure ./config/{config}.yaml exists."
        )

    # default ckpt paths (mirrors training scripts)
    stage1_ckpt = args.stage1_ckpt or os.path.join("ainn_model", f"model_{args.config}_stage1_{num_samples}.pth")
    stage2_ckpt = args.stage2_ckpt or f"model_{args.config}_stage2_{num_samples}.pkl"
    stage3_ckpt = args.stage3_ckpt or os.path.join("ainn_model", f"model_{args.config}_stage3_{num_samples}.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_loc = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- load stage1 model -----
    stage1_model = SimpleNN().to(device)
    sd1 = _load_torch_ckpt(stage1_ckpt, map_location=map_loc)
    stage1_model.load_state_dict(sd1["net"])
    stage1_model.eval()

    # ----- load stage2 sorter (CLRS) -----
    if clrs_main is None:
        raise RuntimeError(
            f"monitor_helper import failed; cannot run stage2 sorting.\nOriginal error: {_IMPORT_ERR}"
        )
    sorter_model, _feedback, rng_key = clrs_main()
    sorter_model.restore_model(stage2_ckpt)

    # ----- load stage3 model -----
    stage3_model = FractionalKnapsackTwoStage().to(device)
    sd3 = _load_torch_ckpt(stage3_ckpt, map_location=map_loc)
    stage3_model.load_state_dict(sd3["net"])
    stage3_model.eval()

    # ----- load encoders -----
    enc2 = Stage2Encoder().to(device)
    enc2_sd = _load_torch_ckpt(args.enc2_ckpt, map_location=map_loc)
    enc2.load_state_dict(enc2_sd["net"])
    enc2.eval()

    enc3 = Stage3Encoder().to(device)
    enc3_sd = _load_torch_ckpt(args.enc3_ckpt, map_location=map_loc)
    enc3.load_state_dict(enc3_sd["net"])
    enc3.eval()

    # ----- load stage1 curve JSON -----
    stage1_curve = None
    if args.stage1_curve_json and os.path.exists(args.stage1_curve_json):
        stage1_curve = _load_stage1_curve_from_json(args.stage1_curve_json)

    out = run_pipeline(
        weights=weights,
        profits=profits,
        capacity=args.capacity,
        stage1_model=stage1_model,
        sorter_model=sorter_model,
        rng_key=rng_key,
        stage3_model=stage3_model,
        enc2=enc2,
        enc3=enc3,
        stage1_curve=stage1_curve,
        device=device,
    )

    # ---------------- reporting ----------------
    print("\n==================== AINN End-to-End Inference ====================")
    print(f"Weights:  {weights}")
    print(f"Profits:  {profits}")
    print(f"Capacity: {args.capacity}\n")

    print("\n[Stage 1] Scoring Block")
    print(f"  pred_ratio: {np.array2string(out.pred_ratio, precision=4)}")

    if out.curve_ratio is not None:
        print(f"  curve_pred: {np.array2string(out.curve_ratio, precision=4)}")
        print(f"  relative_diff: {np.array2string(out.stage1_abs_diff, precision=4)}")
        if stage1_curve is not None and stage1_curve.threshold is not None:
            flag_list = [bool(x) for x in out.stage1_flags.tolist()]
            gflag = out.stage1_seq_flag
            print(f"  flags(diff>{stage1_curve.threshold}): {flag_list}")
            print(f"  sequence flag: {gflag}")
            if out.stage1_ok:
                print(f"  anomaly_monitor: \033[92m no anomaly detected \033[0m")
            else:
                print(f"  anomaly_monitor: \033[91m anomaly detected \033[0m")
        else:
            print("  flags: (no threshold in json)")
        if out.stage1_curve_mse is not None and stage1_curve is not None and stage1_curve.mse_threshold is not None:
            print(f"  curve_mse (legacy): {out.stage1_curve_mse:.6f}  (mse_thr={stage1_curve.mse_threshold})")
    else:
        print("  curve_monitor: (disabled; stage1_curve_json missing)")

    print("\n[Stage 2] Sorting Block")
    print(f"  sort_indices (model ret_ind): {out.sort_indices.tolist()}")
    if out.stage2_ok:
        print(f"  anomaly_monitor: \033[92m no anomaly detected \033[0m")
    else:
        print(f"  anomaly_monitor: \033[91m anomaly detected \033[0m")

    print("\n[Stage 3] Selection Block")
    print(f"  cumsum_order (sorted): {np.array2string(out.cumsum_order, precision=4)}")
    print(f"  k_idx: {int(np.argmax(out.k_mask))}  k_mask: {out.k_mask.astype(int).tolist()}")
    if out.stage3_ok:
        print(f"  anomaly_monitor: \033[92m no anomaly detected \033[0m")
    else:
        print(f"  anomaly_monitor: \033[91m anomaly detected \033[0m")

    print("\n[Final Output]")
    print(f"  fractions (model, original order): {np.array2string(out.fractions, precision=4)}")
    print(f"  total_profit (model): {out.total_profit:.6f}")
    diff_sel = np.abs(out.fractions - out.fractions_gt)
    print(f"  |fractions-model - fractions-gt|: {np.array2string(diff_sel, precision=4)}  (max={float(diff_sel.max()):.6f})")
    if float(diff_sel.max()) < 1e-5:
        print('Correct Final Output')
    else:
        print('Incorrect Final Output')

    # locate earliest failing stage
    failures = []
    if not out.stage1_ok:
        failures.append("stage1")
    if not out.stage2_ok:
        failures.append("stage2")
    if not out.stage3_ok:
        failures.append("stage3")
    '''
    if failures:
        print(f"\nDetected potential failure(s): {', '.join(failures)}")
    else:
        print("\nNo monitor flagged failures (given current thresholds).")
    '''
    print_failure_banner(failures)
    if args.print_gt:
        print("[Ground Truth]")
        print(f"  Stage1 ratio_gt (p/w):          {np.array2string(out.ratio_gt, precision=4)}")
        print(f"  Stage2 sorted_idx_gt (by p/w):  {out.stage2_gt_indices.tolist()}")
        print(f"  Stage3 fractions_gt (optimal):  {np.array2string(out.fractions_gt, precision=4)}")
        print(f"  Stage3 total_profit_gt:         {out.total_profit_gt:.6f}")




if __name__ == "__main__":
    main()


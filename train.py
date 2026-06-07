"""
train.py  (v3)
Graph-Coupled Flow Matching — 主训练循环

v3 相对 v2 的修正：
  - 移除 OT Coupling：训练时 OT 配对的 Z 与推理时 i.i.d. 高斯 Z 存在
    分布不一致（train-test mismatch），导致 FM loss ↓ 但 MAE ↑。
  - 移除 Encoder GCN（ENC_GCN_LAYERS=0）：2 层 GCN 对节点特征过度
    平滑（over-smoothing），使条件向量 C 失去空间区分能力。
  - 恢复 LR=1e-3 + AdamW + StepLR：在此数据集上更稳定。
  - evaluate() 加入随机集成采样（N_SAMPLES=5）：多条 Euler 路径取均值，
    抵消积分误差，对 MAE 有稳定提升且无额外训练开销。

保留的架构改进（v2 中无副作用的部分）：
  - Fourier 正弦步长嵌入（替代 MLP s-embed）
  - VectorField 残差块中的 LayerNorm
  - 更大的模型容量（D_ENC=128, VF_HIDDEN=256, GCN_LAYERS=3）
"""

import argparse
import os
import pickle
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_data
from model   import GCFlowTeacher, LMTC_ABLATE_MODES, VF_ABLATE_MODES
from sampler import euler_sample


# ─── 超参数 ────────────────────────────────────────────────────
# 数据集路径（通过命令行参数 --data_dir 指定）
DATA_DIR       = 'metr-la'  # 默认数据集目录
INPUT_STEPS    = 12
OUTPUT_STEPS   = 12
BATCH_SIZE     = 64
D_ENC          = 128      # GRU Encoder hidden dim
VF_HIDDEN      = 256      # VectorField hidden dim
GCN_LAYERS     = 3        # GCN layers in VectorField
ENC_GCN_LAYERS = 1        # Encoder 空间 GCN 层数（0 = 禁用，避免过平滑）
USE_LMTC       = True     # Encoder 是否启用 LMTC（用于消融）
EULER_K        = 20       # Euler steps at inference
N_SAMPLES      = 30        # 推理时随机集成采样次数（多路径取均值）
USE_RESIDUAL_FM = True     # 论文默认：学习残差 ΔY=Y-X_last；False: 学习绝对值 Y
TRAIN_MODE     = "fm" # fm / sup / hybrid
SUP_LOSS       = "l1"     # l1 / l2
LAMBDA_FM      = 1.0
LAMBDA_SUP     = 1.0
VAL_EVERY      = 5        # validate every N epochs
# 两阶段验证：训练中快验证（用于选ckpt）+ 训练后高精度评估（用于报告最终指标）
FAST_VAL_EULER_K = 12
FAST_VAL_N_SAMPLES = 1
FAST_VAL_PROB_METRICS = False
LOG_FILE       = None  # 由 main() 根据数据集和参数动态生成
LR             = 1e-3
EPOCHS         = 50
# StepLR 步长自动推导为 EPOCHS 的 1/3（每衰减一次 gamma=0.5）
LR_STEP        = max(1, EPOCHS // 3)
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_INTERVAL   = 500
# ───────────────────────────────────────────────────────────────


def load_adj(path: str, device: str) -> torch.Tensor:
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    adj_np = obj[2]
    adj = torch.tensor(adj_np, dtype=torch.float32).to(device)
    print(f"[Adj] shape={adj.shape}, min={adj.min():.4f}, max={adj.max():.4f}")
    return adj


def find_adj_path(data_dir: str) -> str:
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    preferred_names = [
        "adj.pkl",
        "adj_baymat.pkl",
        "adj_lamat.pkl",
        "metr-la_adj.pkl",
        f"{dataset_name}_adj.pkl",
    ]

    for name in preferred_names:
        cand = os.path.join(data_dir, name)
        if os.path.exists(cand):
            return cand

    for name in sorted(os.listdir(data_dir)):
        if name.endswith("_adj.pkl"):
            return os.path.join(data_dir, name)

    raise FileNotFoundError(
        f"Could not find adjacency matrix in {data_dir}. "
        "Expected adj.pkl or *_adj.pkl."
    )


def inverse_norm(x_norm: torch.Tensor, mean, std) -> torch.Tensor:
    if not torch.is_tensor(mean):
        mean = torch.as_tensor(mean, device=x_norm.device, dtype=x_norm.dtype)
    if not torch.is_tensor(std):
        std = torch.as_tensor(std, device=x_norm.device, dtype=x_norm.dtype)
    if mean.ndim == 1:
        mean = mean.view(1, 1, -1)
        std = std.view(1, 1, -1)
    return x_norm * (std + 1e-8) + mean


def to_device_batch(batch, device: str):
    x, y, time_feat, y_mask, y_baseline = batch
    return (
        x.to(device),
        y.to(device),
        time_feat.to(device),
        y_mask.to(device),
        y_baseline.to(device),
    )


def build_targets(x: torch.Tensor, y: torch.Tensor, y_baseline: torch.Tensor,
                  baseline_mode: str):
    """
    Build normalized FM target and the baseline used to recover absolute Y.

    baseline_mode:
      - last:     Y_target = Y - X_last
      - seasonal: Y_target = Y - hour_of_week_mean
      - none:     Y_target = Y
    """
    T_out = y.shape[1]
    Y = y.permute(0, 2, 1)  # (B, N, T_out)

    if baseline_mode == "seasonal":
        base = y_baseline.permute(0, 2, 1)
        Y_target = Y - base
    elif baseline_mode == "last":
        base = x[:, -1:, :].permute(0, 2, 1).expand(-1, -1, T_out)
        Y_target = Y - base
    elif baseline_mode == "none":
        base = None
        Y_target = Y
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    return Y, Y_target, base


def restore_absolute_prediction(Y_hat_target: torch.Tensor, base: torch.Tensor = None):
    if base is None:
        return Y_hat_target
    return Y_hat_target + base


def calc_point_metrics(pred: torch.Tensor, true: torch.Tensor,
                       valid_mask: torch.Tensor = None,
                       eps: float = 1e-5, mape_mask_threshold: float = 1e-3):
    err_abs = (pred - true).abs()
    err_sq = (pred - true) ** 2

    if valid_mask is not None:
        m = valid_mask > 0.5
        if m.any():
            mae = err_abs[m].mean().item()
            rmse = err_sq[m].mean().sqrt().item()
            denom = true.abs()
            mape_mask = m & (denom > mape_mask_threshold)
            mape = (err_abs[mape_mask] / (denom[mape_mask] + eps)).mean().item() * 100.0 if mape_mask.any() else float("nan")
            return mae, rmse, mape
        return float("nan"), float("nan"), float("nan")

    mae = err_abs.mean().item()
    rmse = err_sq.mean().sqrt().item()
    denom = true.abs()
    mape_mask = denom > mape_mask_threshold
    mape = (err_abs[mape_mask] / (denom[mape_mask] + eps)).mean().item() * 100.0 if mape_mask.any() else float("nan")
    return mae, rmse, mape


def calc_crps_ensemble(samples: torch.Tensor, true: torch.Tensor,
                       valid_mask: torch.Tensor = None) -> float:
    """
    Ensemble CRPS for scalar targets, optionally only over valid positions (masked CRPS).
    samples: (S, B, T, N), true: (B, T, N), valid_mask: (B, T, N) optional, >0.5 为有效.
    """
    S, B, T, N = samples.shape
    samples_flat = samples.reshape(S, B, T * N)   # (S, B, D)
    true_flat = true.reshape(B, T * N)            # (B, D)

    term1 = (samples_flat - true_flat.unsqueeze(0)).abs().mean(dim=0)       # (B, D)
    pairwise = (
        samples_flat.unsqueeze(0) - samples_flat.unsqueeze(1)
    ).abs().mean(dim=(0, 1))                                               # (B, D)
    crps_per_cell = term1 - 0.5 * pairwise

    if valid_mask is not None:
        mask_flat = valid_mask.reshape(B, T * N) > 0.5
        if mask_flat.any():
            return crps_per_cell[mask_flat].mean().item()
        return float("nan")
    return crps_per_cell.mean().item()


def calc_target_scale(true: torch.Tensor, valid_mask: torch.Tensor = None,
                      eps: float = 1e-8) -> float:
    """
    Scale for CRPS normalization: mean(|y_true|) on valid positions.
    """
    if valid_mask is not None:
        m = valid_mask > 0.5
        if m.any():
            return true.abs()[m].mean().item() + eps
        return float("nan")
    return true.abs().mean().item() + eps


def calc_mis95(samples: torch.Tensor, true: torch.Tensor,
               valid_mask: torch.Tensor = None, alpha: float = 0.05) -> float:
    """
    Mean Interval Score for 95% prediction interval (MIS95).

    samples: (S, B, T, N), true: (B, T, N), valid_mask: (B, T, N) optional.
    Uses ensemble quantiles:
      L = Q_{alpha/2}, U = Q_{1-alpha/2}
      MIS = (U-L) + 2/alpha * (L-y) * I(y<L) + 2/alpha * (y-U) * I(y>U)
    """
    lower = torch.quantile(samples, q=alpha / 2.0, dim=0)         # (B,T,N)
    upper = torch.quantile(samples, q=1.0 - alpha / 2.0, dim=0)   # (B,T,N)

    width = upper - lower
    under_pen = (lower - true).clamp_min(0.0)
    over_pen = (true - upper).clamp_min(0.0)
    mis = width + (2.0 / alpha) * (under_pen + over_pen)

    if valid_mask is not None:
        m = valid_mask > 0.5
        if m.any():
            return mis[m].mean().item()
        return float("nan")
    return mis.mean().item()


def naive_baseline(loader, mean, std, device: str):
    all_mae, all_rmse, all_mape = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y, _tf, y_mask, _yb = to_device_batch(batch, device)
            last = x[:, -1, :].unsqueeze(1).expand_as(y)
            pred_real = inverse_norm(last, mean, std)
            true_real = inverse_norm(y, mean, std)
            mae, rmse, mape = calc_point_metrics(pred_real, true_real, valid_mask=y_mask)
            all_mae.append(mae)
            all_rmse.append(rmse)
            all_mape.append(mape)
    mae_arr = np.asarray(all_mae, dtype=np.float64)
    rmse_arr = np.asarray(all_rmse, dtype=np.float64)
    mape_arr = np.asarray(all_mape, dtype=np.float64)
    mae_mean = np.nanmean(mae_arr) if np.any(~np.isnan(mae_arr)) else float("nan")
    rmse_mean = np.nanmean(rmse_arr) if np.any(~np.isnan(rmse_arr)) else float("nan")
    mape_mean = np.nanmean(mape_arr) if np.any(~np.isnan(mape_arr)) else float("nan")
    return mae_mean, rmse_mean, mape_mean


def seasonal_naive_baseline(loader, mean, std, device: str):
    all_mae, all_rmse, all_mape = [], [], []
    with torch.no_grad():
        for batch in loader:
            _x, y, _tf, y_mask, y_baseline = to_device_batch(batch, device)
            pred_real = inverse_norm(y_baseline, mean, std)
            true_real = inverse_norm(y, mean, std)
            mae, rmse, mape = calc_point_metrics(pred_real, true_real, valid_mask=y_mask)
            all_mae.append(mae)
            all_rmse.append(rmse)
            all_mape.append(mape)
    mae_arr = np.asarray(all_mae, dtype=np.float64)
    rmse_arr = np.asarray(all_rmse, dtype=np.float64)
    mape_arr = np.asarray(all_mape, dtype=np.float64)
    mae_mean = np.nanmean(mae_arr) if np.any(~np.isnan(mae_arr)) else float("nan")
    rmse_mean = np.nanmean(rmse_arr) if np.any(~np.isnan(rmse_arr)) else float("nan")
    mape_mean = np.nanmean(mape_arr) if np.any(~np.isnan(mape_arr)) else float("nan")
    return mae_mean, rmse_mean, mape_mean


def calc_supervised_loss(pred: torch.Tensor, target: torch.Tensor,
                         valid_mask: torch.Tensor = None,
                         loss_type: str = "l1") -> torch.Tensor:
    if valid_mask is not None:
        m = valid_mask > 0.5
        if m.any():
            if loss_type == "l1":
                return (pred - target).abs()[m].mean()
            if loss_type == "l2":
                return ((pred - target) ** 2)[m].mean()
            raise ValueError(f"Unknown sup loss: {loss_type}")
        # 极端情况下整批都无效，返回 0 防止 NaN
        return torch.zeros((), dtype=pred.dtype, device=pred.device)
    if loss_type == "l1":
        return nn.functional.l1_loss(pred, target)
    if loss_type == "l2":
        return nn.functional.mse_loss(pred, target)
    raise ValueError(f"Unknown sup loss: {loss_type}")


def train_one_epoch(model, loader, optimizer, device, baseline_mode: str,
                    train_mode: str = "hybrid", lambda_fm: float = 1.0,
                    lambda_sup: float = 1.0, sup_loss_type: str = "l1"):
    model.train()
    total_loss, total_fm, total_sup = 0.0, 0.0, 0.0
    for i, batch in enumerate(loader):
        x, y, time_feat, y_mask, y_baseline = to_device_batch(batch, device)

        B, T_in, N = x.shape
        T_out = y.shape[1]
        Y, Y_target, _base = build_targets(x, y, y_baseline, baseline_mode)
        Y_mask = y_mask.permute(0, 2, 1)  # (B, N, T_out)

        if i == 0 and not hasattr(train_one_epoch, '_shape_checked'):
            train_one_epoch._shape_checked = True
            print(f"\n[Shape Check]")
            print(f"  x         : {x.shape}  (B, T_in, N)")
            print(f"  y         : {y.shape}  (B, T_out, N)")
            print(f"  y_baseline: {y_baseline.shape}  (B, T_out, N)")
            print(f"  time_feat : {time_feat.shape}  (B, T_in, 4)")
            print(f"  baseline  : {baseline_mode}")
            print(f"  Y(permute): {Y.shape}  (B, N, T_out)")

        C = model.encode(x, time_feat)
        loss_fm = torch.tensor(0.0, device=device)
        loss_sup = torch.tensor(0.0, device=device)

        if train_mode in ("fm", "hybrid"):
            Z      = torch.randn_like(Y_target)               # (B, N, T_out)
            s      = torch.rand(B, 1, 1, device=device)
            Y_s    = s * Y_target + (1 - s) * Z
            U_star = Y_target - Z
            v      = model(Y_s, s, C)
            # Masked FM loss：仅在有效位置（y_mask>0.5）计算 MSE，缺失值不参与训练
            m = Y_mask > 0.5
            if m.any():
                loss_fm = ((v - U_star) ** 2)[m].mean()
            else:
                loss_fm = torch.zeros((), device=device)

        if train_mode in ("sup", "hybrid"):
            pred_target = model.predict_target(C)
            if pred_target.shape[-1] != T_out:
                pred_target = pred_target[:, :, :T_out]
            loss_sup = calc_supervised_loss(
                pred_target, Y_target, valid_mask=Y_mask, loss_type=sup_loss_type
            )

        loss = 0.0
        if train_mode in ("fm", "hybrid"):
            loss = loss + lambda_fm * loss_fm
        if train_mode in ("sup", "hybrid"):
            loss = loss + lambda_sup * loss_sup

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_fm += loss_fm.item()
        total_sup += loss_sup.item()
        if (i + 1) % LOG_INTERVAL == 0:
            print(
                f"  batch [{i+1}/{len(loader)}]  loss={loss.item():.6f}  "
                f"fm={loss_fm.item():.6f}  sup={loss_sup.item():.6f}"
            )

    n = len(loader)
    return {
        "loss": total_loss / n,
        "fm": total_fm / n,
        "sup": total_sup / n,
    }


@torch.no_grad()
def evaluate(model, loader, mean, std, device: str,
             K: int = 20, n_samples: int = 1, baseline_mode: str = "last",
             use_residual_fm: bool = None,
             compute_prob_metrics: bool = True,
             infer_mode: str = "fm"):
    """
    Euler 采样后反归一化计算点预测 + 概率指标：
      - 点预测：MAE / RMSE / MAPE（按 infer_mode 用 sup 或 FM 的样本均值）
      - 概率指标：NCRPS / MIS95 始终由 FM 分支采样计算，且使用 masked 口径（与 MAE 等一致，公平）。
    n_samples > 1 时进行随机集成：对同一条件 C 独立采样 n_samples 条
    Euler 路径并取均值，平均掉积分噪声，MAE 通常可稳定下降。
    compute_prob_metrics=False 时跳过 NCRPS/MIS95，适合训练中快验证。
    use_residual_fm 为旧接口兼容参数；若提供则覆盖 baseline_mode。
    """
    if use_residual_fm is not None:
        baseline_mode = "last" if use_residual_fm else "none"
    model.eval()
    all_mae, all_rmse, all_mape = [], [], []
    all_ncrps, all_crps_raw, all_mis95 = [], [], []

    for batch in loader:
        x, y, time_feat, y_mask, y_baseline = to_device_batch(batch, device)

        C = model.encode(x, time_feat)   # (B, N, D)
        _Y, _Y_target, base = build_targets(x, y, y_baseline, baseline_mode)
        preds_fm = None   # FM 分支采样结果，仅当需要 NCRPS/MIS95 且 n_samples>1 时使用
        if infer_mode == "sup":
            Y_hat_target = model.predict_target(C)             # (B, N, T)
        else:
            if n_samples == 1:
                Y_hat_target = euler_sample(model, C, K=K)         # (B, N, T)
            else:
                preds_fm = torch.stack([euler_sample(model, C, K=K)
                                         for _ in range(n_samples)])   # (S, B, N, T)
                Y_hat_target = preds_fm.mean(0)                       # (B, N, T)

        Y_hat = restore_absolute_prediction(Y_hat_target, base)
        if preds_fm is not None:
            preds_fm = restore_absolute_prediction(preds_fm, base.unsqueeze(0))

        Y_hat = Y_hat.permute(0, 2, 1)           # (B, T_out, N)
        if preds_fm is not None:
            preds_fm = preds_fm.permute(0, 1, 3, 2)   # (S, B, T_out, N)

        pred_real = inverse_norm(Y_hat, mean, std)
        true_real = inverse_norm(y, mean, std)

        mae, rmse, mape = calc_point_metrics(pred_real, true_real, valid_mask=y_mask)
        all_mae.append(mae)
        all_rmse.append(rmse)
        all_mape.append(mape)

        # 概率指标：始终用 FM 分支采样，masked NCRPS/MIS95（与点预测的 masked MAE 口径一致）
        if compute_prob_metrics and n_samples > 1:
            if preds_fm is None:
                preds_fm = torch.stack([euler_sample(model, C, K=K) for _ in range(n_samples)])
                preds_fm = restore_absolute_prediction(preds_fm, base.unsqueeze(0) if base is not None else None)
                preds_fm = preds_fm.permute(0, 1, 3, 2)
            preds_real = inverse_norm(preds_fm, mean, std)
            crps_raw = calc_crps_ensemble(preds_real, true_real, valid_mask=y_mask)
            denom = calc_target_scale(true_real, valid_mask=y_mask)
            ncrps = crps_raw / denom if np.isfinite(denom) else float("nan")
            mis95 = calc_mis95(preds_real, true_real, valid_mask=y_mask, alpha=0.05)
            all_ncrps.append(ncrps)
            all_crps_raw.append(crps_raw)
            all_mis95.append(mis95)

    return {
        "mae": np.nanmean(all_mae) if np.any(~np.isnan(all_mae)) else float("nan"),
        "rmse": np.nanmean(all_rmse) if np.any(~np.isnan(all_rmse)) else float("nan"),
        "mape": np.nanmean(all_mape) if np.any(~np.isnan(all_mape)) else float("nan"),
        "crps": np.mean(all_ncrps) if all_ncrps else float("nan"),
        "crps_raw": np.mean(all_crps_raw) if all_crps_raw else float("nan"),
        "mis95": np.mean(all_mis95) if all_mis95 else float("nan"),
    }


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
        self.log.write("\n" + "=" * 60 + "\n")
        self.log.write(f"[New Run] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write("=" * 60 + "\n")
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_lmtc_variant(use_lmtc: bool, lmtc_ablate: str) -> str:
    if not use_lmtc:
        return "no_lmtc"
    if lmtc_ablate and lmtc_ablate != "none":
        return lmtc_ablate
    return "full"


def infer_vf_variant(vf_ablate: str) -> str:
    if vf_ablate and vf_ablate != "none":
        return vf_ablate
    return "full"


def resolve_exp_group(exp_group: str, use_lmtc: bool, lmtc_ablate: str,
                      vf_ablate: str = "none") -> str:
    if exp_group == "auto":
        if vf_ablate and vf_ablate != "none":
            return "vf_ablation"
        if (not use_lmtc) or (lmtc_ablate != "none"):
            return "lmtc_ablation"
        return ""
    if exp_group.lower() in ("", "main", "none"):
        return ""
    return exp_group


def setup_run_dirs(data_dir: str, input_steps: int, output_steps: int,
                   use_lmtc: bool, lmtc_ablate: str,
                   exp_group: str = "auto", seed: int = None,
                   vf_ablate: str = "none"):
    """
    主实验: result/<dataset>/logs/ + result/<dataset>/<dataset>_in*_out*.pt
    LMTC 消融: result/<dataset>/lmtc_ablation/...
    VF 消融:   result/<dataset>/vf_ablation/...
    """
    dataset_tag = data_dir.replace("/", "_").strip("_")
    exp_group = resolve_exp_group(exp_group, use_lmtc, lmtc_ablate, vf_ablate)
    if exp_group == "vf_ablation":
        result_dir = os.path.join("result", dataset_tag, exp_group)
        variant = infer_vf_variant(vf_ablate)
        ckpt_base = f"{variant}_in{input_steps}_out{output_steps}"
    elif exp_group:
        result_dir = os.path.join("result", dataset_tag, exp_group)
        variant = infer_lmtc_variant(use_lmtc, lmtc_ablate)
        ckpt_base = f"{variant}_in{input_steps}_out{output_steps}"
    else:
        result_dir = os.path.join("result", dataset_tag)
        ckpt_base = f"{dataset_tag}_in{input_steps}_out{output_steps}"
    if seed is not None and exp_group:
        ckpt_base = f"{ckpt_base}_seed{seed}"
    log_dir = os.path.join(result_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return result_dir, log_dir, ckpt_base, exp_group


def make_log_file(log_dir: str, input_steps: int, output_steps: int,
                  use_lmtc: bool, lmtc_ablate: str, exp_group: str,
                  vf_ablate: str = "none") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if exp_group == "vf_ablation":
        variant_tag = f"_{infer_vf_variant(vf_ablate)}"
    elif exp_group:
        variant_tag = f"_{infer_lmtc_variant(use_lmtc, lmtc_ablate)}"
    else:
        variant_tag = ""
    return os.path.join(
        log_dir,
        f"train_log_in{input_steps}_out{output_steps}{variant_tag}_{timestamp}.txt",
    )


def run_experiment(exp_name: str, use_residual_fm: bool, use_lmtc: bool,
                   train_loader, val_loader, test_loader,
                   mean, std, adj: torch.Tensor, save_path: str,
                   input_steps: int = 12, output_steps: int = 12,
                   train_mode: str = "hybrid", sup_loss: str = "l1",
                   lambda_fm: float = 1.0, lambda_sup: float = 1.0,
                   infer_mode: str = "sup",
                   baseline_mode: str = None,
                   lmtc_ablate: str = "none",
                   vf_ablate: str = "none",
                   epochs: int = EPOCHS, lr: float = LR, lr_step: int = LR_STEP,
                   euler_k: int = EULER_K, n_samples: int = N_SAMPLES,
                   fast_val_k: int = FAST_VAL_EULER_K,
                   fast_val_n_samples: int = FAST_VAL_N_SAMPLES,
                   fast_val_prob_metrics: bool = FAST_VAL_PROB_METRICS,
                   val_every: int = VAL_EVERY):
    if baseline_mode is None:
        baseline_mode = "last" if use_residual_fm else "none"
    # Reset one-time shape print for each experiment
    if hasattr(train_one_epoch, "_shape_checked"):
        delattr(train_one_epoch, "_shape_checked")

    print("\n" + "=" * 60)
    print(f"[Experiment] {exp_name}")
    print(f"  USE_RESIDUAL_FM = {use_residual_fm}")
    print(f"  BASELINE_MODE   = {baseline_mode}")
    print(f"  USE_LMTC        = {use_lmtc}")
    print(f"  LMTC_ABLATE     = {lmtc_ablate if use_lmtc else 'disabled'}")
    print(f"  VF_ABLATE       = {vf_ablate if vf_ablate != 'none' else 'full'}")
    print(f"  TRAIN_MODE      = {train_mode}")
    print(f"  SUP_LOSS        = {sup_loss}")
    print(f"  LAMBDA_FM       = {lambda_fm}")
    print(f"  LAMBDA_SUP      = {lambda_sup}")
    print(f"  INFER_MODE      = {infer_mode}")
    print(f"  EPOCHS          = {epochs}")
    print(f"  LR              = {lr}")
    print(f"  LR_STEP         = {lr_step}")
    print(f"  EULER_K         = {euler_k}")
    print(f"  N_SAMPLES       = {n_samples}")
    print(f"  FAST_VAL_K      = {fast_val_k}")
    print(f"  FAST_VAL_S      = {fast_val_n_samples}")
    print(f"  FAST_VAL_PROB   = {fast_val_prob_metrics}")
    print(f"  VAL_EVERY       = {val_every}")
    print(f"  INPUT_STEPS     = {input_steps}")
    print(f"  OUTPUT_STEPS    = {output_steps}")
    print(f"  SAVE_PATH       = {save_path}")
    print("=" * 60)

    model = GCFlowTeacher(
        adj=adj,
        T_in=input_steps, T_out=output_steps,
        enc_hidden=D_ENC,
        vf_hidden=VF_HIDDEN,
        gcn_layers=GCN_LAYERS,
        enc_gcn_layers=ENC_GCN_LAYERS,
        use_lmtc=use_lmtc,
        lmtc_ablate=lmtc_ablate,
        vf_ablate=vf_ablate,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] 参数量: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.5)

    print("\n[Naive Baseline] 计算中...")
    val_nb_mae, val_nb_rmse, val_nb_mape = naive_baseline(val_loader, mean, std, DEVICE)
    test_nb_mae, test_nb_rmse, test_nb_mape = naive_baseline(test_loader, mean, std, DEVICE)
    print(f"  Val  Last-step -- MAE: {val_nb_mae:.4f}  RMSE: {val_nb_rmse:.4f}  MAPE: {val_nb_mape:.2f}%")
    print(f"  Test Last-step -- MAE: {test_nb_mae:.4f}  RMSE: {test_nb_rmse:.4f}  MAPE: {test_nb_mape:.2f}%")
    val_sb_mae, val_sb_rmse, val_sb_mape = seasonal_naive_baseline(val_loader, mean, std, DEVICE)
    test_sb_mae, test_sb_rmse, test_sb_mape = seasonal_naive_baseline(test_loader, mean, std, DEVICE)
    print(
        f"  Val  Seasonal  -- MAE: {val_sb_mae:.4f}  RMSE: {val_sb_rmse:.4f}  MAPE: {val_sb_mape:.2f}%"
    )
    print(
        f"  Test Seasonal  -- MAE: {test_sb_mae:.4f}  RMSE: {test_sb_rmse:.4f}  MAPE: {test_sb_mape:.2f}%"
    )
    print("=" * 60)

    best_mae = float('inf')

    for epoch in range(1, epochs + 1):
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch}/{epochs}]  lr={cur_lr:.2e}")
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, DEVICE, baseline_mode,
            train_mode=train_mode, lambda_fm=lambda_fm,
            lambda_sup=lambda_sup, sup_loss_type=sup_loss,
        )
        scheduler.step()

        if epoch % val_every == 0 or epoch == epochs:
            val_metrics = evaluate(
                model, val_loader, mean, std, DEVICE,
                K=fast_val_k,
                n_samples=fast_val_n_samples,
                baseline_mode=baseline_mode,
                compute_prob_metrics=fast_val_prob_metrics,
                infer_mode=infer_mode,
            )
            val_mae = val_metrics["mae"]
            print(
                "  >> train loss={:.6f} (fm={:.6f}, sup={:.6f})"
                "  |  fast-val MAE={:.4f}  RMSE={:.4f}  MAPE={:.2f}%".format(
                    train_metrics["loss"], train_metrics["fm"], train_metrics["sup"],
                    val_metrics["mae"], val_metrics["rmse"], val_metrics["mape"],
                )
            )
            print(
                f"     vs Val Naive: MAE={val_nb_mae:.4f} (diff={val_mae - val_nb_mae:+.4f}) | "
                f"MAPE={val_nb_mape:.2f}% (diff={val_metrics['mape'] - val_nb_mape:+.2f}%)"
            )

            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), save_path)
                print(f"     >>> saved best model (MAE={best_mae:.4f})")
        else:
            next_val = ((epoch // val_every) + 1) * val_every
            print(
                "  >> train loss={:.6f} (fm={:.6f}, sup={:.6f})  "
                "(skip val, next at epoch {})".format(
                    train_metrics["loss"], train_metrics["fm"], train_metrics["sup"], next_val
                )
            )

    print("\n" + "=" * 60)
    print(f"[Checkpoint Select] {exp_name} | 最优 val MAE: {best_mae:.4f}（Val Naive: {val_nb_mae:.4f}）")

    # 训练结束后：仅对 best checkpoint 在 test 上做一次最终评估
    best_state = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(best_state)
    final_metrics = evaluate(
        model, test_loader, mean, std, DEVICE,
        K=euler_k, n_samples=n_samples, baseline_mode=baseline_mode,
        compute_prob_metrics=True,
        infer_mode=infer_mode,
    )
    print(
        "[Final Test Eval] MAE={:.4f}  RMSE={:.4f}  MAPE={:.2f}%  CRPS={:.4f}  CRPS_raw={:.4f}  MIS95={:.4f}".format(
            final_metrics["mae"], final_metrics["rmse"], final_metrics["mape"],
            final_metrics["crps"], final_metrics["crps_raw"], final_metrics["mis95"],
        )
    )
    ref_name = "Last-step" if baseline_mode != "seasonal" else "Seasonal"
    ref_mae = test_nb_mae if baseline_mode != "seasonal" else test_sb_mae
    print(
        f"               vs Test {ref_name}: MAE={ref_mae:.4f} "
        f"(diff={final_metrics['mae'] - ref_mae:+.4f})"
    )
    if baseline_mode == "seasonal":
        print(
            f"               vs Test Last-step: MAE={test_nb_mae:.4f} "
            f"(diff={final_metrics['mae'] - test_nb_mae:+.4f})"
        )
    return final_metrics["mae"], test_nb_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab_compare", action="store_true",
                        help="顺序跑 absolute 与 residual 两个实验并输出对照结果")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_lmtc", action="store_true",
                        help="关闭 LMTC（消融：w/o LMTC）")
    parser.add_argument("--lmtc_ablate", type=str, default="none",
                        choices=list(LMTC_ABLATE_MODES),
                        help="LMTC 结构消融：none/full, wo_periodic, wo_multiscale, "
                             "fixed_smooth, wo_gate")
    parser.add_argument("--vf_ablate", type=str, default="none",
                        choices=list(VF_ABLATE_MODES),
                        help="VectorField 消融：none/full, reaction_only, "
                             "diffusion_only, wo_gate")
    parser.add_argument("--exp_group", type=str, default="auto",
                        help="实验输出分组：auto / vf_ablation / lmtc_ablation / main")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help="数据集目录（如 metr-la、PEMS04、PEMS08），需包含原始 npz 和 *_adj.pkl")
    parser.add_argument("--input_steps", type=int, default=INPUT_STEPS,
                        help="输入步长，默认12，可改为如36")
    parser.add_argument("--output_steps", "--t_out", dest="output_steps", type=int, default=OUTPUT_STEPS,
                        help="输出步长，默认12，可改为如12")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="batch size")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=LR,
                        help="学习率")
    parser.add_argument("--lr_step", type=int, default=0,
                        help="StepLR step_size，<=0 表示自动设为 epochs//3")
    parser.add_argument("--train_mode", type=str, default=TRAIN_MODE, choices=["fm", "sup", "hybrid"],
                        help="训练模式：fm(仅FM) / sup(仅监督) / hybrid(混合训练)")
    parser.add_argument("--sup_loss", type=str, default=SUP_LOSS, choices=["l1", "l2"],
                        help="监督损失类型")
    parser.add_argument("--lambda_fm", type=float, default=LAMBDA_FM,
                        help="FM loss 权重")
    parser.add_argument("--lambda_sup", type=float, default=LAMBDA_SUP,
                        help="监督 loss 权重")
    parser.add_argument("--infer_mode", type=str, default="auto", choices=["auto", "fm", "sup"],
                        help="验证/测试时点预测方式")
    parser.add_argument("--euler_k", type=int, default=EULER_K,
                        help="最终测试评估时 Euler 积分步数")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES,
                        help="最终测试评估时集成采样次数")
    parser.add_argument("--fast_val_k", type=int, default=FAST_VAL_EULER_K,
                        help="训练中快验证时 Euler 积分步数")
    parser.add_argument("--fast_val_s", type=int, default=FAST_VAL_N_SAMPLES,
                        help="训练中快验证时采样次数")
    parser.add_argument("--fast_val_prob", action="store_true",
                        help="训练中快验证是否计算 NCRPS/MIS95（默认关闭以节省开销）")
    parser.add_argument("--val_every", type=int, default=VAL_EVERY,
                        help="每多少个 epoch 做一次验证")
    parser.add_argument("--steps_per_day", type=str, default="auto",
                        help="steps per day for time features: auto, PEMS=288, Seattle=24")
    parser.add_argument("--node_norm", action="store_true",
                        help="按节点独立计算 mean/std 做归一化（适合 Seattle 等节点尺度差异大的数据集）")
    parser.add_argument("--baseline_mode", type=str, default="last",
                        choices=["last", "seasonal", "none"],
                        help="FM 目标基线：last=Y-X_last, seasonal=Y-hour_of_week_mean, none=绝对值 Y")
    args = parser.parse_args()
    use_lmtc = not args.no_lmtc
    lmtc_ablate = args.lmtc_ablate
    vf_ablate = args.vf_ablate
    if args.no_lmtc and args.lmtc_ablate != "none":
        print("[Warn] --no_lmtc 已启用，忽略 --lmtc_ablate")
        lmtc_ablate = "none"
    infer_mode = args.infer_mode
    lr_step = args.lr_step if args.lr_step > 0 else max(1, args.epochs // 3)
    steps_per_day = None if args.steps_per_day.lower() == "auto" else int(args.steps_per_day)
    if steps_per_day is not None and steps_per_day <= 0:
        raise ValueError(f"--steps_per_day must be positive or auto, got {args.steps_per_day}")
    if infer_mode == "auto":
        infer_mode = "sup" if args.train_mode in ("sup", "hybrid") else "fm"
    norm_mode = "node" if args.node_norm else "global"
    baseline_mode = args.baseline_mode

    # 主实验 -> result/<dataset>/；LMTC 消融 -> result/<dataset>/lmtc_ablation/
    result_dir, log_dir, ckpt_base, exp_group = setup_run_dirs(
        args.data_dir,
        args.input_steps,
        args.output_steps,
        use_lmtc=use_lmtc,
        lmtc_ablate=lmtc_ablate,
        exp_group=args.exp_group,
        seed=args.seed,
        vf_ablate=vf_ablate,
    )
    log_file = make_log_file(
        log_dir, args.input_steps, args.output_steps,
        use_lmtc, lmtc_ablate, exp_group, vf_ablate=vf_ablate,
    )
    sys.stdout = Logger(log_file)

    adj_path = find_adj_path(args.data_dir)
    set_seed(args.seed)

    print(f"Device: {DEVICE}")
    print("=" * 60)
    print("Hyperparameters (v3):")
    print(f"  DATA_DIR       = {args.data_dir}")
    print(f"  INPUT_STEPS    = {args.input_steps}")
    print(f"  OUTPUT_STEPS   = {args.output_steps}")
    print(f"  STEPS_PER_DAY  = {args.steps_per_day}")
    print(f"  NORM_MODE      = {norm_mode}")
    print(f"  BASELINE_MODE  = {baseline_mode}")
    print(f"  TRAIN_MODE     = {args.train_mode}")
    print(f"  SUP_LOSS       = {args.sup_loss}")
    print(f"  LAMBDA_FM      = {args.lambda_fm}")
    print(f"  LAMBDA_SUP     = {args.lambda_sup}")
    print(f"  INFER_MODE     = {infer_mode}")
    print(f"  BATCH_SIZE     = {args.batch_size}")
    print(f"  D_ENC          = {D_ENC}     (Encoder hidden dim)")
    print(f"  VF_HIDDEN      = {VF_HIDDEN}   (VectorField hidden dim)")
    print(f"  GCN_LAYERS     = {GCN_LAYERS}         (VectorField GCN layers)")
    print(f"  ENC_GCN_LAYERS = {ENC_GCN_LAYERS}         (Encoder GCN，0=禁用)")
    print(f"  USE_LMTC       = {use_lmtc}      (是否启用多尺度时间条件模块)")
    print(f"  LMTC_ABLATE    = {lmtc_ablate if use_lmtc else 'disabled'}")
    print(f"  VF_ABLATE      = {vf_ablate if vf_ablate != 'none' else 'full'}")
    print(f"  EULER_K        = {args.euler_k}        (Euler steps)")
    print(f"  N_SAMPLES      = {args.n_samples}         (集成采样次数)")
    print(f"  FAST_VAL_K     = {args.fast_val_k}        (训练中快验证 Euler steps)")
    print(f"  FAST_VAL_S     = {args.fast_val_s}         (训练中快验证采样数)")
    print(f"  FAST_VAL_PROB  = {args.fast_val_prob}     (训练中是否算NCRPS/MIS95)")
    print(f"  USE_RESIDUAL_FM= {USE_RESIDUAL_FM}      (论文默认单跑模式)")
    print(f"  VAL_EVERY      = {args.val_every}         (validate every N epochs)")
    print(f"  LR             = {args.lr}")
    print(f"  LR_STEP        = {lr_step}        (StepLR step_size, gamma=0.5)")
    print(f"  EPOCHS         = {args.epochs}")
    print(f"  SEED           = {args.seed}")
    print(f"  AB_COMPARE     = {args.ab_compare}")
    print(f"  EXP_GROUP      = {exp_group or 'main'}")
    print(f"  RESULT_DIR     = {result_dir}")
    print(f"  LOG_FILE       = {log_file}")
    print("=" * 60)

    train_loader, val_loader, test_loader, mean, std = load_data(
        args.data_dir,
        batch_size=args.batch_size,
        input_steps=args.input_steps,
        output_steps=args.output_steps,
        steps_per_day=steps_per_day,
        norm_mode=norm_mode,
        compute_seasonal=True,
    )
    adj = load_adj(adj_path, DEVICE)

    if args.ab_compare:
        # 公平对比：两次实验都用同一随机种子初始化
        set_seed(args.seed)
        mae_abs, nb_mae = run_experiment(
            exp_name="Absolute FM",
            use_residual_fm=False,
            use_lmtc=use_lmtc,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            mean=mean, std=std, adj=adj,
            save_path=os.path.join(result_dir, f"{ckpt_base}_absolute.pt"),
            input_steps=args.input_steps,
            output_steps=args.output_steps,
            train_mode=args.train_mode,
            sup_loss=args.sup_loss,
            lambda_fm=args.lambda_fm,
            lambda_sup=args.lambda_sup,
            infer_mode=infer_mode,
            baseline_mode="none",
            epochs=args.epochs,
            lr=args.lr,
            lr_step=lr_step,
            euler_k=args.euler_k,
            n_samples=args.n_samples,
            fast_val_k=args.fast_val_k,
            fast_val_n_samples=args.fast_val_s,
            fast_val_prob_metrics=args.fast_val_prob,
            val_every=args.val_every,
        )
        set_seed(args.seed)
        mae_res, _ = run_experiment(
            exp_name="Residual FM",
            use_residual_fm=True,
            use_lmtc=use_lmtc,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            mean=mean, std=std, adj=adj,
            save_path=os.path.join(result_dir, f"{ckpt_base}_residual.pt"),
            input_steps=args.input_steps,
            output_steps=args.output_steps,
            train_mode=args.train_mode,
            sup_loss=args.sup_loss,
            lambda_fm=args.lambda_fm,
            lambda_sup=args.lambda_sup,
            infer_mode=infer_mode,
            baseline_mode="last",
            epochs=args.epochs,
            lr=args.lr,
            lr_step=lr_step,
            euler_k=args.euler_k,
            n_samples=args.n_samples,
            fast_val_k=args.fast_val_k,
            fast_val_n_samples=args.fast_val_s,
            fast_val_prob_metrics=args.fast_val_prob,
            val_every=args.val_every,
        )

        print("\n" + "=" * 60)
        print("[AB Result]")
        print(f"  Test Naive MAE  : {nb_mae:.4f}")
        print(f"  Absolute FM MAE : {mae_abs:.4f}")
        print(f"  Residual FM MAE : {mae_res:.4f}")
        print(f"  Delta (Res-Abs) : {mae_res - mae_abs:+.4f}")
        better = "Residual FM" if mae_res < mae_abs else "Absolute FM"
        print(f"  Better          : {better}")
    else:
        best_mae, nb_mae = run_experiment(
            exp_name="Single Run",
            use_residual_fm=USE_RESIDUAL_FM,
            use_lmtc=use_lmtc,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            mean=mean, std=std, adj=adj,
            save_path=os.path.join(result_dir, f"{ckpt_base}.pt"),
            input_steps=args.input_steps,
            output_steps=args.output_steps,
            train_mode=args.train_mode,
            sup_loss=args.sup_loss,
            lambda_fm=args.lambda_fm,
            lambda_sup=args.lambda_sup,
            infer_mode=infer_mode,
            baseline_mode=baseline_mode,
            lmtc_ablate=lmtc_ablate,
            vf_ablate=vf_ablate,
            epochs=args.epochs,
            lr=args.lr,
            lr_step=lr_step,
            euler_k=args.euler_k,
            n_samples=args.n_samples,
            fast_val_k=args.fast_val_k,
            fast_val_n_samples=args.fast_val_s,
            fast_val_prob_metrics=args.fast_val_prob,
            val_every=args.val_every,
        )
        print("\n" + "=" * 60)
        print(f"训练完成！Test MAE: {best_mae:.4f}（Test Naive: {nb_mae:.4f}）")


if __name__ == '__main__':
    main()

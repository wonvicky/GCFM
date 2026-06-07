"""
dataset.py

Raw-series loading + chronological 8:1:1 split + sliding-window datasets.

Supported raw files:
  - METR-LA style: <data_dir>/<name>.npz with key "df/block0_values" -> (T, N)
  - PEMS style:    <data_dir>/<name>.npz with key "data" -> (T, N, C)

All loaders return batches of:
  x:         (B, input_steps, N)
  y:         (B, output_steps, N)
  time_feat: (B, input_steps, 4)
  y_mask:    (B, output_steps, N)
  y_baseline:(B, output_steps, N)  hour-of-week mean in normalized space
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

STEPS_PER_DAY = 288
DAYS_PER_WEEK = 7


def infer_steps_per_day(data_dir: str) -> int:
    """Infer temporal resolution from the dataset name/file names."""
    dataset_name = os.path.basename(os.path.normpath(data_dir)).lower()
    if "seattle" in dataset_name:
        return 24

    if os.path.isdir(data_dir):
        for name in os.listdir(data_dir):
            lower = name.lower()
            if lower.endswith(".npz") and ("1hour" in lower or "1h" in lower):
                return 24
    return STEPS_PER_DAY


def make_time_features(num_steps: int, offset: int = 0,
                       steps_per_day: int = STEPS_PER_DAY) -> np.ndarray:
    """
    Returns sinusoidal TOD/DOW features for each absolute time step.

    Shape: (num_steps, 4)
    """
    if steps_per_day <= 0:
        raise ValueError(f"steps_per_day must be positive, got {steps_per_day}")

    idx = np.arange(num_steps, dtype=np.int64) + int(offset)
    tod = (idx % steps_per_day).astype(np.float32)
    dow = ((idx // steps_per_day) % DAYS_PER_WEEK).astype(np.float32)

    tod_sin = np.sin(2 * np.pi * tod / steps_per_day)
    tod_cos = np.cos(2 * np.pi * tod / steps_per_day)
    dow_sin = np.sin(2 * np.pi * dow / DAYS_PER_WEEK)
    dow_cos = np.cos(2 * np.pi * dow / DAYS_PER_WEEK)
    return np.stack([tod_sin, tod_cos, dow_sin, dow_cos], axis=-1).astype(np.float32)


def _find_raw_npz(data_dir: str) -> str:
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    preferred = os.path.join(data_dir, f"{dataset_name}.npz")
    if os.path.exists(preferred):
        return preferred

    candidates = []
    for name in os.listdir(data_dir):
        if not name.endswith(".npz"):
            continue
        lower = name.lower()
        if lower in {"train.npz", "val.npz", "valid.npz", "test.npz"}:
            continue
        candidates.append(os.path.join(data_dir, name))

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find a raw dataset npz in {data_dir}. "
        "Expected something like <data_dir>/<dataset_name>.npz."
    )


def _to_float_series(series: np.ndarray, feature_idx: int = None) -> np.ndarray:
    """Coerce raw arrays (including object dtype) to (T, N) float32."""
    if series.ndim == 3:
        # PEMS / Seattle-style datasets may store multiple channels.
        if feature_idx is not None:
            if feature_idx < 0 or feature_idx >= series.shape[-1]:
                raise ValueError(
                    f"feature_idx={feature_idx} is out of range for raw series shape {series.shape}"
                )
            try:
                series = np.asarray(series[..., feature_idx], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"feature_idx={feature_idx} cannot be converted to numeric values"
                ) from exc
        else:
            # Pick the first channel that can be safely converted to float.
            picked = None
            for c in range(series.shape[-1]):
                cand = series[..., c]
                try:
                    cand = np.asarray(cand, dtype=np.float64)
                    picked = cand
                    break
                except (TypeError, ValueError):
                    continue
            if picked is None:
                raise ValueError(
                    "Could not convert any channel in 3D raw series to numeric values."
                )
            series = picked
    elif series.dtype == object:
        if feature_idx is not None:
            raise ValueError(
                f"feature_idx={feature_idx} was provided, but raw series is not 3D: {series.shape}"
            )
        series = np.asarray(series, dtype=np.float64)
    elif feature_idx is not None:
        if series.ndim != 3:
            raise ValueError(
                f"feature_idx={feature_idx} was provided, but raw series is not 3D: {series.shape}"
            )
        try:
            series = np.asarray(series[..., feature_idx], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"feature_idx={feature_idx} cannot be converted to numeric values"
            ) from exc
    if series.ndim != 2:
        raise ValueError(f"Expected raw series shape (T, N), got {series.shape}")
    return series.astype(np.float32)


def load_raw_series(data_dir: str, feature_idx: int = None) -> np.ndarray:
    """
    Load raw time series with shape (T, N), float32.
    """
    raw_path = _find_raw_npz(data_dir)
    raw_npz = np.load(raw_path, allow_pickle=True)
    keys = set(raw_npz.keys())

    if "df/block0_values" in keys:
        series = raw_npz["df/block0_values"]
    elif "data" in keys:
        series = raw_npz["data"]
        if series.ndim == 0:
            # Some datasets store a pandas DataFrame in a 0-d object array.
            series = series.item()
            if hasattr(series, "values"):
                series = series.values
    else:
        raise KeyError(
            f"Unsupported raw dataset format in {raw_path}. Found keys: {list(raw_npz.keys())}"
        )

    series = _to_float_series(np.asarray(series), feature_idx=feature_idx)
    if series.ndim != 2:
        raise ValueError(f"Expected raw series shape (T, N), got {series.shape} from {raw_path}")
    return series


def compute_seasonal_baseline(series: np.ndarray, train_time_end: int,
                              steps_per_day: int,
                              y_mask: np.ndarray = None) -> np.ndarray:
    """
    Hour-of-week mean per node from the training timeline only.

    Returns seasonal_mean with shape (steps_per_day * 7, N) in raw value space.
    """
    slots_per_week = steps_per_day * DAYS_PER_WEEK
    train_series = series[:train_time_end]
    if y_mask is None:
        mask = (train_series > 1e-6).astype(np.float64)
    else:
        mask = y_mask[:train_time_end].astype(np.float64)

    seasonal_sum = np.zeros((slots_per_week, series.shape[1]), dtype=np.float64)
    seasonal_count = np.zeros_like(seasonal_sum)
    how = np.arange(train_time_end, dtype=np.int64) % slots_per_week
    weighted = train_series.astype(np.float64) * mask
    np.add.at(seasonal_sum, how, weighted)
    np.add.at(seasonal_count, how, mask)

    seasonal_mean = seasonal_sum / np.maximum(seasonal_count, 1.0)
    node_mean = weighted.sum(axis=0) / np.maximum(mask.sum(axis=0), 1.0)
    missing = seasonal_count == 0
    seasonal_mean = np.where(missing, node_mean, seasonal_mean)
    return seasonal_mean.astype(np.float32)


def normalize_series(series: np.ndarray, train_time_end: int,
                     norm_mode: str = "global"):
    """Normalize using training timeline statistics only."""
    train_obs = series[:train_time_end]
    if norm_mode == "node":
        mean = train_obs.mean(axis=0).astype(np.float32)
        std = train_obs.std(axis=0).astype(np.float32)
    elif norm_mode == "global":
        mean = np.float32(train_obs.mean())
        std = np.float32(train_obs.std())
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    series_norm = (series - mean) / (std + 1e-8)
    return series_norm.astype(np.float32), mean, std


def split_sample_starts(total_steps: int, input_steps: int, output_steps: int,
                        ratios=(0.8, 0.1, 0.1)):
    if len(ratios) != 3:
        raise ValueError(f"Expected 3 split ratios, got {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    num_samples = total_steps - input_steps - output_steps + 1
    if num_samples <= 0:
        raise ValueError(
            f"Not enough time steps ({total_steps}) for input_steps={input_steps}, "
            f"output_steps={output_steps}"
        )

    # Time-based boundaries (chronological 8:1:1 on timeline).
    train_time_end = int(total_steps * ratios[0])
    val_time_end = int(total_steps * (ratios[0] + ratios[1]))
    train_time_end = max(train_time_end, input_steps + output_steps)
    val_time_end = max(val_time_end, train_time_end + 1)
    val_time_end = min(val_time_end, total_steps - 1)

    starts = np.arange(num_samples, dtype=np.int64)
    y_start = starts + input_steps
    y_end = y_start + output_steps  # exclusive

    train_starts = starts[y_end <= train_time_end]
    val_starts = starts[(y_start >= train_time_end) & (y_end <= val_time_end)]
    test_starts = starts[(y_start >= val_time_end) & (y_end <= total_steps)]

    if min(len(train_starts), len(val_starts), len(test_starts)) <= 0:
        raise ValueError(
            "Split produced an empty subset after chronological assignment: "
            f"train={len(train_starts)}, val={len(val_starts)}, test={len(test_starts)}; "
            f"total_steps={total_steps}, input_steps={input_steps}, output_steps={output_steps}"
        )
    return train_starts, val_starts, test_starts, train_time_end, val_time_end


class TrafficDataset(Dataset):
    def __init__(self, series: np.ndarray, y_mask: np.ndarray,
                 time_feat: np.ndarray, starts: np.ndarray,
                 input_steps: int, output_steps: int,
                 seasonal_baseline_norm: np.ndarray = None,
                 steps_per_day: int = STEPS_PER_DAY):
        """
        series:     (T, N) normalized float32
        y_mask:     (T, N) float32, 1 for valid target observations
        time_feat:  (T, 4) float32
        starts:     (num_samples,) int64
        seasonal_baseline_norm: optional (H, N) normalized hour-of-week means
        """
        self.series = torch.tensor(series, dtype=torch.float32)
        self.y_mask = torch.tensor(y_mask, dtype=torch.float32)
        self.time_feat = torch.tensor(time_feat, dtype=torch.float32)
        self.starts = torch.tensor(starts, dtype=torch.long)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.steps_per_day = steps_per_day
        self.slots_per_week = steps_per_day * DAYS_PER_WEEK
        self.seasonal_baseline_norm = (
            seasonal_baseline_norm.astype(np.float32)
            if seasonal_baseline_norm is not None else None
        )

    def __len__(self):
        return int(self.starts.numel())

    def __getitem__(self, idx):
        start = int(self.starts[idx].item())
        input_end = start + self.input_steps
        output_end = input_end + self.output_steps

        x = self.series[start:input_end]
        y = self.series[input_end:output_end]
        time_feat_x = self.time_feat[start:input_end]
        y_mask = self.y_mask[input_end:output_end]
        if self.seasonal_baseline_norm is not None:
            out_idx = np.arange(input_end, output_end, dtype=np.int64)
            how = out_idx % self.slots_per_week
            y_baseline = torch.tensor(
                self.seasonal_baseline_norm[how], dtype=torch.float32
            )
        else:
            y_baseline = torch.zeros_like(y)
        return x, y, time_feat_x, y_mask, y_baseline


def load_data(data_dir: str, batch_size: int = 32,
              input_steps: int = 12, output_steps: int = 12,
              split_ratio=(0.8, 0.1, 0.1), feature_idx: int = None,
              steps_per_day: int = None, norm_mode: str = "global",
              compute_seasonal: bool = True):
    """
    Returns:
      train_loader, val_loader, test_loader, mean, std
    """
    if steps_per_day is None:
        steps_per_day = infer_steps_per_day(data_dir)

    series = load_raw_series(data_dir, feature_idx=feature_idx)  # (T, N)
    total_steps, num_nodes = series.shape
    train_starts, val_starts, test_starts, train_time_end, val_time_end = split_sample_starts(
        total_steps=total_steps,
        input_steps=input_steps,
        output_steps=output_steps,
        ratios=split_ratio,
    )

    y_mask = (series > 1e-6).astype(np.float32)
    series_norm, mean, std = normalize_series(series, train_time_end, norm_mode=norm_mode)
    time_feat = make_time_features(total_steps, offset=0, steps_per_day=steps_per_day)

    seasonal_baseline_norm = None
    if compute_seasonal:
        seasonal_raw = compute_seasonal_baseline(
            series, train_time_end, steps_per_day, y_mask=y_mask
        )
        seasonal_baseline_norm = (seasonal_raw - mean) / (std + 1e-8)

    if norm_mode == "node":
        mean_msg = (
            f"mean(min/median/max)="
            f"{mean.min():.2f}/{np.median(mean):.2f}/{mean.max():.2f}, "
            f"std(min/median/max)="
            f"{std.min():.2f}/{np.median(std):.2f}/{std.max():.2f}"
        )
    else:
        mean_msg = f"mean={float(mean):.4f}, std={float(std):.4f}"

    print(f"[Dataset] raw series  -- shape={series.shape}, {mean_msg}")
    print(f"[Dataset] norm mode     -- {norm_mode}")
    print(f"[Dataset] input/output -- input_steps={input_steps}, output_steps={output_steps}")
    print(f"[Dataset] time feature -- steps_per_day={steps_per_day}")
    if seasonal_baseline_norm is not None:
        print(
            f"[Dataset] seasonal baseline -- shape={seasonal_baseline_norm.shape}, "
            f"slots_per_week={steps_per_day * DAYS_PER_WEEK}"
        )
    print(
        f"[Dataset] timeline cut  -- train:[0,{train_time_end}) "
        f"val:[{train_time_end},{val_time_end}) test:[{val_time_end},{total_steps})"
    )
    print(
        f"[Dataset] split windows -- train={len(train_starts)}, "
        f"val={len(val_starts)}, test={len(test_starts)}"
    )
    print(
        f"[Dataset] split ratio   -- "
        f"{len(train_starts)/(len(train_starts)+len(val_starts)+len(test_starts)):.4f} / "
        f"{len(val_starts)/(len(train_starts)+len(val_starts)+len(test_starts)):.4f} / "
        f"{len(test_starts)/(len(train_starts)+len(val_starts)+len(test_starts)):.4f}"
    )
    print(f"[Dataset] valid ratio   -- {(y_mask.mean()):.4f}")
    print(f"[Dataset] nodes         -- N={num_nodes}, total_steps={total_steps}")

    ds_kwargs = dict(
        seasonal_baseline_norm=seasonal_baseline_norm,
        steps_per_day=steps_per_day,
    )
    train_ds = TrafficDataset(
        series_norm, y_mask, time_feat, train_starts, input_steps, output_steps, **ds_kwargs
    )
    val_ds = TrafficDataset(
        series_norm, y_mask, time_feat, val_starts, input_steps, output_steps, **ds_kwargs
    )
    test_ds = TrafficDataset(
        series_norm, y_mask, time_feat, test_starts, input_steps, output_steps, **ds_kwargs
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    return train_loader, val_loader, test_loader, mean, std


if __name__ == "__main__":
    tr, va, te, mean, std = load_data("metr-la", batch_size=4, input_steps=12, output_steps=12)
    x, y, tf, y_mask, y_baseline = next(iter(tr))
    print(
        f"x shape: {x.shape}, y shape: {y.shape}, time_feat: {tf.shape}, "
        f"y_mask: {y_mask.shape}, y_baseline: {y_baseline.shape}"
    )
    print(f"mean={mean:.4f}, std={std:.4f}")

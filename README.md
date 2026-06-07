# GCFM: Graph-Coupled Flow Matching for Probabilistic Spatio-Temporal Forecasting

Official implementation of **GCFM** (ICDM 2026).

## Files

```
GCFM/
├── train.py      # Training & evaluation
├── model.py      # GCFlowTeacher / LMTC / VectorField
├── dataset.py    # Data loading and sliding windows
├── sampler.py    # Euler sampler
└── paths.py      # Path config (overridable via env vars)
```

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.8+
- PyTorch 2.0+ (CUDA recommended)

## Data

Each dataset directory needs `<name>.npz` and `<name>_adj.pkl`.

Supported: PEMS04, PEMS08, METR-LA (5 min), Seattle (1 hour).

Set dataset root via environment variable:

```bash
export ICDM_DATA_ROOT=/path/to/your/data
```

## Training

```bash
mkdir -p logs result

# PEMS08 — default FM setting
nohup python train.py \
  --data_dir ${ICDM_DATA_ROOT}/PEMS08 \
  --input_steps 12 --output_steps 12 \
  --train_mode fm --infer_mode fm \
  --batch_size 64 --epochs 50 --seed 42 \
  > logs/nohup_pems08_train.log 2>&1 &
```

Key arguments:

- `--train_mode`: `fm` (default) / `sup` / `hybrid`
- `--infer_mode`: `fm` / `sup`
- `--baseline_mode`: `last` / `seasonal` / `none`
- `--lmtc_ablate`: `none` / `wo_periodic` / `wo_multiscale` / `fixed_smooth` / `wo_gate`
- `--vf_ablate`: `none` / `reaction_only` / `diffusion_only` / `wo_gate`

Checkpoints and logs are saved under `result/` and `logs/` (git-ignored).

## Citation

```bibtex
@inproceedings{gcfm2026,
  title={GCFM: Graph-Coupled Flow Matching for Probabilistic Spatio-Temporal Forecasting},
  author={Anonymous},
  booktitle={IEEE International Conference on Data Mining (ICDM)},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).

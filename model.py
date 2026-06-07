"""
model.py  (v4)

Core modules:
  1. FourierEmbedding: sinusoidal embedding for flow time s
  2. GCNLayer: symmetric normalized graph convolution
  3. LMTC: lightweight multi-scale temporal conditioning
  4. GRUEncoder: temporal encoder + spatial refinement + LMTC fusion
  5. VectorField: graph-coupled reaction-diffusion velocity field
  6. GCFlowTeacher: encoder + vector field + supervised point head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEmbedding(nn.Module):
    """Sinusoidal Fourier embedding for scalar flow time s in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32)
            / max(half - 1, 1)
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = s.reshape(-1, 1) * self.freqs.unsqueeze(0)
        emb = torch.cat([s.sin(), s.cos()], dim=-1)
        return self.proj(emb)


class GCNLayer(nn.Module):
    """
    Symmetric normalized graph convolution:
      H' = ReLU(A_hat @ H @ W)
      A_hat = D^{-1/2}(A+I)D^{-1/2}
    """

    def __init__(self, in_dim: int, out_dim: int,
                 adj: torch.Tensor, add_self_loop: bool = True):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        if add_self_loop:
            adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        A_hat = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        self.register_buffer("A_hat", A_hat)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        ah = torch.einsum("nm,bnf->bmf", self.A_hat, h)
        return F.relu(self.W(ah))


LMTC_ABLATE_MODES = (
    "none",           # Full LMTC
    "wo_periodic",    # w/o periodic branch (calendar MLP)
    "wo_multiscale",  # w/o multi-scale flow decomposition
    "fixed_smooth",   # fixed moving-average low-pass instead of learnable
    "wo_gate",        # uniform band aggregation instead of scale gate
)

VF_ABLATE_MODES = (
    "none",             # Full: h_react + lambda * h_diff
    "reaction_only",    # node-wise reaction only
    "diffusion_only",   # graph diffusion only (h_react still computed internally)
    "wo_gate",          # fixed fusion: h_react + h_diff
)


class LMTC(nn.Module):
    """
    Learnable Multi-resolution Temporal Conditioning.

    Instead of hand-crafted trend (fixed-decay EMA) and volatility statistics
    (finite-difference moments), LMTC performs a *learnable causal
    multi-resolution decomposition* of the per-node history and conditions the
    flow-matching velocity field on the resulting band representations.

    Decomposition. Let x in R^T be a node series. We build a smoothing pyramid
    with K learnable causal low-pass filters {LP_k}:
        m_0 = x,   m_k = LP_k(m_{k-1}),   k = 1..K
    where each LP_k is a depthwise causal convolution whose kernel is
    softmax-normalized (non-negative, sums to one), i.e. a learnable convex
    smoothing operator. The detail (band-pass) components and the residual
    trend are
        u_k = m_{k-1} - m_k  (k = 1..K),   tau = m_K,
    which give an exact reconstruction  x = tau + sum_k u_k  (perfect-
    reconstruction multi-resolution analysis, akin to a learnable à-trous /
    Laplacian-pyramid wavelet). Coarse bands capture trend/periodic structure,
    fine bands capture volatility -- no magic decay constant, no manual moments.

    Fusion. Each band is embedded over time, then aggregated by a content-
    adaptive cross-scale gate (a normalized attention over scales), so the
    contribution of each resolution is data-dependent and interpretable. A
    calendar (time-feature) embedding injects periodic phase information.
    """

    def __init__(self, cond_dim: int, input_len: int,
                 time_feat_dim: int = 4, n_scales: int = 3,
                 branch_dim: int = 64, base_kernel: int = 3,
                 ablate_mode: str = "none"):
        super().__init__()
        if ablate_mode not in LMTC_ABLATE_MODES:
            raise ValueError(
                f"Unknown LMTC ablate_mode={ablate_mode!r}, "
                f"expected one of {LMTC_ABLATE_MODES}"
            )
        self.ablate_mode = ablate_mode
        self.input_len = input_len
        self.time_feat_dim = time_feat_dim
        self.n_scales = int(n_scales)
        self.n_bands = self.n_scales + 1  # K detail bands + 1 trend

        # Learnable causal low-pass filters (depthwise, single channel) with
        # growing receptive field across scales.
        self.smooth_convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=base_kernel + 2 * k, bias=False)
            for k in range(self.n_scales)
        ])

        # Shared temporal embedding applied to every band (B*N, n_bands, T) -> cond_dim.
        self.band_proj = nn.Linear(input_len, cond_dim)
        # Content-adaptive cross-scale gate over band energies.
        self.scale_gate = nn.Linear(self.n_bands, self.n_bands)
        # Calendar/periodic embedding from time features.
        self.periodic_mlp = nn.Sequential(
            nn.Linear(input_len * time_feat_dim, branch_dim),
            nn.SiLU(),
            nn.Linear(branch_dim, cond_dim),
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.single_branch_out = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def _causal_lowpass(self, s: torch.Tensor, k: int) -> torch.Tensor:
        """
        Apply the k-th learnable causal low-pass filter.

        s: (B*N, 1, T). The kernel is softmax-normalized so the filter is a
        convex smoothing operator (a proper learnable moving average).
        """
        conv = self.smooth_convs[k]
        w = torch.softmax(conv.weight, dim=-1)  # (1, 1, ks), non-negative, sums to 1
        ks = w.shape[-1]
        s_pad = F.pad(s, (ks - 1, 0))  # causal left padding
        return F.conv1d(s_pad, w)

    def _fixed_causal_lowpass(self, s: torch.Tensor, k: int) -> torch.Tensor:
        """Fixed uniform moving average with the same kernel size as scale k."""
        ks = self.smooth_convs[k].weight.shape[-1]
        w = torch.full(
            (1, 1, ks), 1.0 / ks, device=s.device, dtype=s.dtype
        )
        s_pad = F.pad(s, (ks - 1, 0))
        return F.conv1d(s_pad, w)

    def _compute_multiscale(self, s: torch.Tensor):
        """Return band components (B*N, n_bands, T) and gated embedding c_ms (B, N, D)."""
        b_n, _, t = s.shape
        n_bands = self.n_bands

        lowpass = (
            self._fixed_causal_lowpass
            if self.ablate_mode == "fixed_smooth"
            else self._causal_lowpass
        )

        bands = []
        m_prev = s
        for k in range(self.n_scales):
            m_k = lowpass(m_prev, k)
            bands.append(m_prev - m_k)
            m_prev = m_k
        bands.append(m_prev)
        comps = torch.cat(bands, dim=1)

        band_emb = self.band_proj(comps)
        if self.ablate_mode == "wo_gate":
            gate = torch.full(
                (b_n, n_bands), 1.0 / n_bands, device=s.device, dtype=s.dtype
            )
        else:
            energy = comps.abs().mean(dim=-1)
            gate = torch.softmax(self.scale_gate(energy), dim=-1)

        c_ms = (band_emb * gate.unsqueeze(-1)).sum(dim=1)
        return comps, c_ms

    def _compute_periodic(self, time_feat: torch.Tensor, b: int, n: int) -> torch.Tensor:
        t = time_feat.shape[1]
        tf_flat = time_feat.reshape(b, t * self.time_feat_dim)
        return self.periodic_mlp(tf_flat).unsqueeze(1).expand(b, n, -1)

    def forward(self, x: torch.Tensor, time_feat: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N), time_feat: (B, T, 4), c: (B, N, D)
        b, t, n = x.shape
        if t != self.input_len:
            raise ValueError(f"LMTC expected input_len={self.input_len}, got T={t}")

        s = x.permute(0, 2, 1).reshape(b * n, 1, t)  # (B*N, 1, T)

        if self.ablate_mode == "wo_multiscale":
            c_rho = self._compute_periodic(time_feat, b, n)
            return c + self.single_branch_out(c_rho)

        _, c_ms = self._compute_multiscale(s)
        c_ms = c_ms.reshape(b, n, -1)

        if self.ablate_mode == "wo_periodic":
            return c + self.single_branch_out(c_ms)

        c_rho = self._compute_periodic(time_feat, b, n)
        c_fused = self.fuse_mlp(torch.cat([c_ms, c_rho], dim=-1))
        return c + c_fused


class GRUEncoder(nn.Module):
    """
    Spatio-temporal encoder:
      1) per-node GRU over (x, time features)
      2) optional spatial GCN refinement
      3) optional LMTC fusion
    """

    def __init__(self, T_in: int = 12, hidden_dim: int = 128,
                 time_feat_dim: int = 4, adj: torch.Tensor = None,
                 enc_gcn_layers: int = 2, use_lmtc: bool = True,
                 lmtc_ablate: str = "none"):
        super().__init__()
        self.T_in = T_in
        self.D = hidden_dim
        self.time_feat_dim = time_feat_dim

        self.gru = nn.GRU(
            input_size=1 + time_feat_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.enc_gcns = nn.ModuleList()
        self.enc_norms = nn.ModuleList()
        for _ in range(enc_gcn_layers):
            self.enc_gcns.append(GCNLayer(hidden_dim, hidden_dim, adj, add_self_loop=True))
            self.enc_norms.append(nn.LayerNorm(hidden_dim))

        self.lmtc = LMTC(
            cond_dim=hidden_dim,
            input_len=T_in,
            time_feat_dim=time_feat_dim,
            n_scales=3,
            branch_dim=max(32, hidden_dim // 2),
            ablate_mode=lmtc_ablate,
        ) if use_lmtc else None

    def forward(self, x: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N), time_feat: (B, T, 4)
        b, t, n = x.shape
        x_node = x.permute(0, 2, 1).reshape(b * n, t, 1)
        tf = time_feat.unsqueeze(1).expand(b, n, t, self.time_feat_dim)
        tf = tf.reshape(b * n, t, self.time_feat_dim)

        gru_in = torch.cat([x_node, tf], dim=-1)
        _, h = self.gru(gru_in)
        h = h.squeeze(0)
        h = self.proj(F.relu(h))
        h = h.reshape(b, n, self.D)

        for gcn, norm in zip(self.enc_gcns, self.enc_norms):
            h = norm(h + gcn(h))

        if self.lmtc is not None:
            h = self.lmtc(x, time_feat, h)
        return h


class VectorField(nn.Module):
    """
    Graph-coupled reaction-diffusion velocity field.

    Implements paper equation:
      v_theta = v_react + lambda * v_diff
    with adaptive gate lambda conditioned by pooled C and flow time s.
    """

    def __init__(self, T_out: int = 12, cond_dim: int = 128,
                 hidden_dim: int = 256, gcn_layers: int = 3,
                 adj: torch.Tensor = None, ablate_mode: str = "none"):
        super().__init__()
        if adj is None:
            raise ValueError("VectorField requires a valid adjacency matrix `adj`.")
        if ablate_mode not in VF_ABLATE_MODES:
            raise ValueError(
                f"Unknown VectorField ablate_mode={ablate_mode!r}, "
                f"expected one of {VF_ABLATE_MODES}"
            )
        self.ablate_mode = ablate_mode

        self.T_out = T_out
        self.D = cond_dim
        self.s_dim = 32

        self.s_embed = FourierEmbedding(self.s_dim)
        self.input_proj = nn.Linear(T_out, hidden_dim)

        # A_hat = D^{-1/2}(A+I)D^{-1/2}
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        A_hat = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        self.register_buffer("A_hat", A_hat)

        in_dim = hidden_dim + cond_dim + self.s_dim
        self.react_mlps = nn.ModuleList()
        self.diff_linears = nn.ModuleList()
        self.gate_c = nn.ModuleList()
        self.gate_s = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(gcn_layers):
            self.react_mlps.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ))
            self.diff_linears.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            self.gate_c.append(nn.Linear(cond_dim, hidden_dim, bias=False))
            self.gate_s.append(nn.Linear(self.s_dim, hidden_dim, bias=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, T_out),
        )

    def forward(self, Y_s: torch.Tensor, s: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Y_s: (B, N, T_out), s: (B,1,1) or (B,1), C: (B, N, D)
        b, n, _ = Y_s.shape

        s_global = self.s_embed(s.reshape(b, 1))  # (B, s_dim)
        s_node = s_global.unsqueeze(1).expand(b, n, self.s_dim)
        c_pool = C.mean(dim=1)  # (B, D)

        h = F.silu(self.input_proj(Y_s))

        for react_mlp, diff_linear, gate_c, gate_s, norm in zip(
            self.react_mlps, self.diff_linears, self.gate_c, self.gate_s, self.norms
        ):
            # v_react branch: node-wise local dynamics
            h_react = react_mlp(torch.cat([h, C, s_node], dim=-1))

            # v_diff branch: graph diffusion correction
            ah = torch.einsum("nm,bnf->bmf", self.A_hat, h_react)
            h_diff = diff_linear(ah)

            # lambda gate: (B, d_h) -> (B,1,d_h), broadcast over nodes
            lam = torch.sigmoid(gate_c(c_pool) + gate_s(s_global)).unsqueeze(1)

            if self.ablate_mode == "reaction_only":
                delta = h_react
            elif self.ablate_mode == "diffusion_only":
                delta = lam * h_diff
            elif self.ablate_mode == "wo_gate":
                delta = h_react + h_diff
            else:
                delta = h_react + lam * h_diff

            h = norm(h + delta)

        return self.output_proj(h)


class GCFlowTeacher(nn.Module):
    def __init__(self, adj: torch.Tensor,
                 T_in: int = 12, T_out: int = 12,
                 enc_hidden: int = 128, vf_hidden: int = 256,
                 gcn_layers: int = 3, enc_gcn_layers: int = 2,
                 use_lmtc: bool = True, lmtc_ablate: str = "none",
                 vf_ablate: str = "none"):
        super().__init__()
        self.encoder = GRUEncoder(
            T_in=T_in, hidden_dim=enc_hidden, time_feat_dim=4,
            adj=adj, enc_gcn_layers=enc_gcn_layers,
            use_lmtc=use_lmtc, lmtc_ablate=lmtc_ablate,
        )
        self.vf = VectorField(
            T_out=T_out, cond_dim=enc_hidden,
            hidden_dim=vf_hidden, gcn_layers=gcn_layers, adj=adj,
            ablate_mode=vf_ablate,
        )

        sup_hidden = max(64, enc_hidden)
        self.sup_head = nn.Sequential(
            nn.Linear(enc_hidden, sup_hidden),
            nn.SiLU(),
            nn.Linear(sup_hidden, T_out),
        )

    def encode(self, X_hist: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        # X_hist: (B, T_in, N), time_feat: (B, T_in, 4)
        return self.encoder(X_hist, time_feat)

    def forward(self, Y_s: torch.Tensor, s: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Y_s: (B,N,T_out), s: (B,1,1), C: (B,N,D)
        return self.vf(Y_s, s, C)

    def predict_target(self, C: torch.Tensor) -> torch.Tensor:
        # C: (B, N, D) -> (B, N, T_out)
        return self.sup_head(C)


if __name__ == "__main__":
    B, T_in, T_out, N, D = 4, 12, 12, 325, 128
    adj = torch.eye(N)

    model = GCFlowTeacher(
        adj=adj, T_in=T_in, T_out=T_out,
        enc_hidden=D, vf_hidden=256,
        gcn_layers=3, enc_gcn_layers=2,
    )
    x = torch.randn(B, T_in, N)
    tf = torch.randn(B, T_in, 4)
    y = torch.randn(B, N, T_out)
    s = torch.rand(B, 1, 1)

    C = model.encode(x, tf)
    v = model(y, s, C)
    total = sum(p.numel() for p in model.parameters())
    print(f"C shape : {C.shape}")
    print(f"v shape : {v.shape}")
    print(f"params  : {total:,}")

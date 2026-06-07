"""
sampler.py
Euler 采样器：给定条件 C，从噪声采样出未来 T_out 步
"""

import torch


@torch.no_grad()
def euler_sample(model, C: torch.Tensor, K: int = 20) -> torch.Tensor:
    """
    Euler 方法从 N(0,1) 采样 Y_hat。

    参数：
        model : GCFlowTeacher（已在正确 device 上）
        C     : (B, N, D)，来自 encoder 的条件表示
        K     : Euler 步数（越大越精确，20步足够闭环验证）

    返回：
        Y_hat : (B, N, T_out)，归一化空间中的预测值
    """
    B, N, D = C.shape
    T_out = model.vf.T_out
    device = C.device

    # 初始噪声
    Y = torch.randn(B, N, T_out, device=device)

    step = 1.0 / K
    for k in range(K):
        s_val = k / K                                    # 从0到(K-1)/K
        s = torch.full((B, 1, 1), s_val, device=device)
        v = model(Y, s, C)                               # (B, N, T_out)
        Y = Y + step * v

    # 检查 NaN
    assert not torch.isnan(Y).any(), "采样结果包含 NaN！请检查模型参数或学习率。"
    return Y  # (B, N, T_out)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from model import GCFlowTeacher

    B, N, D = 4, 325, 64
    adj = torch.eye(N)
    model = GCFlowTeacher(adj=adj)
    model.eval()

    X = torch.randn(B, 12, N)
    C = model.encode(X)
    Y_hat = euler_sample(model, C, K=20)
    print(f"Y_hat shape: {Y_hat.shape}")   # (4, 325, 12)
    print(f"Y_hat stats: mean={Y_hat.mean():.4f}, std={Y_hat.std():.4f}")

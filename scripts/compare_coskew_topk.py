"""
compare_coskew_topk.py

realデータの3次モーメント(coskewness)テンソルの中で、絶対値が大きい
(=構造的に意味のある)要素を上位k個だけ特定し、その同じ位置での値を
  - real (真の値)
  - fake_approx (ミニバッチ近似 coskew_batch=500 で生成した既存版)
  - fake_exact  (近似なし coskew_batch=5000 で生成した検証版)
の3者で比較する。

全体のノルム平均(subloss)は「平均的に合っているか」しか見えないが、
ここでは「値が大きく、分類に効いている可能性が高い少数の要素」に絞って
近似の影響を確認する。

使い方:
  python compare_coskew_topk.py --class_id 0 \
      --fake_exact_npz /home/nakano/server/moment_data/deletion/third_order/_verify_exact/class0.npz \
      --topk 200
"""
import argparse
import heapq
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cifar10_loader import load_cifar10
from deletion_third_order_fast import compute_coskewness_slice  # 数値検証済みの正確な実装


def find_topk_coskew_indices(X, k, slice_size=64, device="cuda"):
    """
    X: (N, D) の中心化前データ。coskewnessをスライドしながら計算し、
    |値| が大きい上位k個の (j,k_idx,l) インデックスと値を返す。
    D^3を全部保持せず、スライスごとに部分的なtop-kをheapで維持する。
    """
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    D = X.shape[1]
    heap = []  # (abs_value, value, j, k_idx, l) の最小ヒープ、サイズk

    for s in range(0, D, slice_size):
        e = min(s + slice_size, D)
        with torch.no_grad():
            block = compute_coskewness_slice(X, s, e, slice_dim=0)  # (e-s, D, D)
        block_cpu = block.cpu().numpy()
        # 上三角(k_idx <= l)のみ見る(対称性: block[a,k,l]==block[a,l,k])
        for a in range(block_cpu.shape[0]):
            j = s + a
            mat = block_cpu[a]  # (D, D)
            # 絶対値が大きい候補だけ効率的に拾う(全D*Dを毎回heapに通すと遅いので、
            # このスライス内でのtop-2kをまず絞ってからheapへ)
            flat = mat[np.triu_indices(D)]
            idx_flat = np.triu_indices(D)
            if flat.size == 0:
                continue
            local_k = min(2 * k, flat.size)
            top_local = np.argpartition(-np.abs(flat), local_k - 1)[:local_k]
            for t in top_local:
                kk, ll = idx_flat[0][t], idx_flat[1][t]
                val = float(flat[t])
                item = (abs(val), val, j, int(kk), int(ll))
                if len(heap) < k:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

    heap.sort(key=lambda x: -x[0])
    return heap  # [(abs_val, val, j, k_idx, l), ...] 降順


def value_at_indices(X, indices, device="cuda"):
    """指定した (j,k,l) 位置でのcoskewness値をXから直接計算(topk評価用、軽量)。"""
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    Xc = X - X.mean(dim=0)
    out = []
    for (_, _, j, k_idx, l) in indices:
        v = (Xc[:, j] * Xc[:, k_idx] * Xc[:, l]).mean().item()
        out.append(v)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class_id", type=int, required=True)
    ap.add_argument("--fake_approx_npz", type=Path, default=None,
                     help="近似版(coskew_batch=500)のclass npz。省略時は既存の"
                          "_partial/class{c}.npz を使う。")
    ap.add_argument("--fake_exact_npz", type=Path, required=True,
                     help="近似なし版(coskew_batch=5000)のclass npz。")
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--slice_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    approx_path = args.fake_approx_npz or Path(
        f"/home/nakano/server/moment_data/deletion/third_order/_partial/class{args.class_id}.npz"
    )

    print(f"[class {args.class_id}] loading real CIFAR-10 ...")
    x_train, y_train, _, _ = load_cifar10()
    H, W, C = x_train.shape[1:]
    D = H * W * C
    x_real = x_train[y_train == args.class_id].reshape(-1, D).astype(np.float32)

    print(f"[class {args.class_id}] loading fake (approx, exact) ...")
    d_approx = np.load(approx_path)
    d_exact = np.load(args.fake_exact_npz)
    x_approx = d_approx["pixel_values"].reshape(-1, D).astype(np.float32)
    x_exact = d_exact["pixel_values"].reshape(-1, D).astype(np.float32)

    print(f"[class {args.class_id}] finding top-{args.topk} |coskew| entries in REAL data "
          f"(this scans the full D={D} slice-wise, may take a minute)...")
    topk = find_topk_coskew_indices(x_real, args.topk,
                                     slice_size=args.slice_size, device=args.device)

    real_vals = np.array([v for (_, v, *_ ) in topk])
    approx_vals = value_at_indices(x_approx, topk, device=args.device)
    exact_vals = value_at_indices(x_exact, topk, device=args.device)

    err_approx = np.abs(approx_vals - real_vals)
    err_exact = np.abs(exact_vals - real_vals)

    print(f"\n=== top-{args.topk} |coskew| entries (class {args.class_id}) ===")
    print(f"real value range: [{real_vals.min():.6f}, {real_vals.max():.6f}] "
          f"(|.|>={abs(real_vals).min():.6f})")
    print(f"\napprox (coskew_batch=500): "
          f"mean|err|={err_approx.mean():.6f}  max|err|={err_approx.max():.6f}  "
          f"corr={np.corrcoef(real_vals, approx_vals)[0,1]:.4f}")
    print(f"exact  (coskew_batch=5000): "
          f"mean|err|={err_exact.mean():.6f}  max|err|={err_exact.max():.6f}  "
          f"corr={np.corrcoef(real_vals, exact_vals)[0,1]:.4f}")

    ratio = err_approx.mean() / max(err_exact.mean(), 1e-12)
    print(f"\napprox/exact mean|err| ratio: {ratio:.2f}x")
    if ratio > 1.5:
        print(">> 近似版の方が誤差が大きい: ミニバッチ近似が精度低下の一因である可能性が高い")
    elif ratio < 0.67:
        print(">> 近似版の方が誤差が小さい(想定外): 何か別の要因を疑う必要あり")
    else:
        print(">> 近似版と正確版で誤差はほぼ同水準: 近似は主因ではなく、"
              "CIFAR-10で3次情報の追加寄与が小さいという解釈を支持")

    # 上位10件だけ詳細表示
    print(f"\n--- top 10 entries detail ---")
    print(f"{'j':>5} {'k':>5} {'l':>5} {'real':>10} {'approx':>10} {'exact':>10}")
    for i in range(min(10, len(topk))):
        _, v, j, kk, ll = topk[i]
        print(f"{j:5d} {kk:5d} {ll:5d} {v:10.6f} {approx_vals[i]:10.6f} {exact_vals[i]:10.6f}")


if __name__ == "__main__":
    main()
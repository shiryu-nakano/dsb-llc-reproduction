"""
sweep_coskew_weight.py

third_order_sample_gram の coskew_weight を複数試し、各々の final_eval
(means/cov/coskewの絶対値)を比較する。Gram trick版は1クラス約1分なので、
複数の重みを試しても数分で終わる。

使い方:
  python sweep_coskew_weight.py --class_id 0 --weights 0.1 1 10 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from cifar10_loader import load_cifar10
from deletion_third_order_gram import third_order_sample_gram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class_id", type=int, default=0)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--n_steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"loading CIFAR-10 class {args.class_id} ...")
    x_train, y_train, _, _ = load_cifar10()
    H, W, C = x_train.shape[1:]
    D = H * W * C
    x_class = x_train[y_train == args.class_id].reshape(-1, D)
    n_c = x_class.shape[0]
    print(f"class {args.class_id}: {n_c} real samples, D={D}")

    print(f"\nsweeping coskew_weight over {args.weights}\n")
    for w in args.weights:
        third_order_sample_gram(
            x_class, n_c,
            n_steps=args.n_steps,
            lr=args.lr,
            coskew_weight=w,
            device=args.device,
            verbose=False,       # final_evalの1行だけ見る
            final_eval=True,
            desc=f"class{args.class_id}_w{w}",
        )


if __name__ == "__main__":
    main()
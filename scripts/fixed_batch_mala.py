"""
固定した1つの評価バッチを使い、check_everyステップごとに
(直前スナップショット時点のパラメータ) vs (現在のパラメータ)
を同一バッチ上で比較して受理確率を計算するコールバック。

devinterp標準の MalaAcceptanceRate は毎drawごとに異なるミニバッチの
loss同士を比較しており、ミニバッチノイズが受理率に混入する問題がある。
これに対応するため、固定バッチ上で(20ステップごとの)パラメータ前後比較を行う。
まずは尤度項のみ(localization項は含めない簡易版)。

calibrate_gamma_epsilon_torch.py (vision_plain較正) からそのまま移植。
"""
import torch
import torch.nn.functional as F

from devinterp.slt.mala import mala_acceptance_probability
from devinterp.slt.callback import SamplerCallback


class FixedBatchMalaAcceptanceRate(SamplerCallback):
    """
    固定した1つの評価バッチを使い、check_everyステップごとに
    (直前スナップショット時点のパラメータ) vs (現在のパラメータ)
    を同一バッチ上で比較して受理確率を計算する(JAX版と同じロジック)。
    まずは尤度項のみで近似(localization項は含めない)。
    """

    def __init__(self, fixed_batch, epsilon, nbeta, check_every=20, device="cuda"):
        super().__init__(device=device)
        self.fixed_x, self.fixed_y = fixed_batch
        self.epsilon = epsilon
        self.nbeta = nbeta
        self.check_every = check_every
        self.accept_probs = []
        self._prev = None

    def _evaluate_fixed_batch(self, model):
        model.zero_grad()
        out = model(pixel_values=self.fixed_x, labels=None)
        logits = out.logits if hasattr(out, "logits") else out
        loss = F.cross_entropy(logits, self.fixed_y)
        mala_loss = loss.detach() * self.nbeta
        loss.backward()

        params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        grads = [p.grad.detach().clone() * self.nbeta for p in model.parameters() if p.requires_grad]
        model.zero_grad()
        return params, grads, mala_loss

    def __call__(self, i, model, **kwargs):
        if i % self.check_every != 0:
            return
        with torch.enable_grad():
            params, grads, mala_loss = self._evaluate_fixed_batch(model)
            if self._prev is not None:
                prev_params, prev_grads, prev_loss = self._prev
                prob = mala_acceptance_probability(
                    prev_params, prev_grads, prev_loss,
                    params, grads, mala_loss,
                    self.epsilon,
                )
                self.accept_probs.append([i, float(prob)])
            self._prev = (params, grads, mala_loss)

    def get_results(self):
        return {
            "fixed_batch_mala_accept/trace": self.accept_probs,
            "fixed_batch_mala_accept/mean": (
                sum(p for _, p in self.accept_probs) / len(self.accept_probs)
                if self.accept_probs else float("nan")
            ),
        }
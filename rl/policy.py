"""GRPO policy: categorical distribution over discrete rubric-editing actions."""

import json
import numpy as np
from pathlib import Path


class ActionPolicy:
    """Categorical policy over K named actions, updated via group-relative PO.

    Logits are stored raw; probabilities are computed via stable softmax on
    demand.  A GRPO step normalises rewards within a group of K rollouts and
    updates logits proportionally to the resulting advantages.
    """

    def __init__(
        self,
        action_names: list[str],
        logits: "np.ndarray | None" = None,
    ) -> None:
        self.action_names: list[str] = list(action_names)
        self.logits: np.ndarray = (
            np.array(logits, dtype=float)
            if logits is not None
            else np.zeros(len(action_names), dtype=float)
        )

    # ── Core ──────────────────────────────────────────────────────────────────

    def probabilities(self) -> np.ndarray:
        z = self.logits - self.logits.max()   # numerical stability
        e = np.exp(z)
        return e / e.sum()

    def grpo_update(
        self,
        rewards: dict[str, float],
        lr: float = 0.1,
        eps: float = 1e-8,
        beta: float = 0.0,
    ) -> dict[str, float]:
        """One GRPO step: group-normalise rewards, then update logits.

        Skipped / missing actions contribute NaN rewards.  They are excluded
        from the group mean/std but receive a zero-advantage (no logit change).

        beta > 0 adds a KL penalty that pulls the policy back toward the uniform
        reference distribution, preventing premature collapse onto one action:
            Δ logit_i = lr · (A_i  −  β · ∂KL(π ‖ π_ref)/∂logit_i)
        where ∂KL/∂logit_i = π_i · (log(π_i / π_ref_i) − KL).

        Returns a per-action advantage dict (before KL subtraction) for logging.
        """
        r = np.array([rewards.get(a, float("nan")) for a in self.action_names])
        valid = ~np.isnan(r)

        if valid.sum() < 2:
            return {a: 0.0 for a in self.action_names}

        mu = float(r[valid].mean())
        sigma = float(r[valid].std())

        # A_i = (r_i − μ) / (σ + ε)  for valid actions, 0 for skipped
        advantages = np.where(valid, (r - mu) / (sigma + eps), 0.0)

        update = advantages.copy()
        if beta > 0.0:
            pi     = self.probabilities()
            pi_ref = np.ones(len(self.action_names)) / len(self.action_names)
            kl     = float(np.sum(pi * np.log(pi / pi_ref + 1e-10)))
            kl_grad = pi * (np.log(pi / pi_ref + 1e-10) - kl)
            update  = advantages - beta * kl_grad

        self.logits += lr * update

        return {a: float(advantages[i]) for i, a in enumerate(self.action_names)}

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(self, rng: "np.random.RandomState | None" = None) -> str:
        if rng is None:
            rng = np.random.RandomState()
        probs = self.probabilities()
        return self.action_names[int(rng.choice(len(self.action_names), p=probs))]

    def best_action(self) -> str:
        return self.action_names[int(np.argmax(self.probabilities()))]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "action_names": self.action_names,
            "logits": self.logits.tolist(),
            "probabilities": self.probabilities().tolist(),
        }

    def save(self, path: "Path | str") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: "Path | str") -> "ActionPolicy":
        d = json.loads(Path(path).read_text())
        return cls(
            action_names=d["action_names"],
            logits=np.array(d["logits"]),
        )

    def __repr__(self) -> str:
        probs = self.probabilities()
        items = ", ".join(
            f"{a}={p:.3f}" for a, p in zip(self.action_names, probs)
        )
        return f"ActionPolicy({items})"

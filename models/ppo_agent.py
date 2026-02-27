
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = []
        self.values = []
        self.bad_masks = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.masks.clear()
        self.values.clear()
        self.bad_masks.clear()


def _safe_tensor(x: torch.Tensor, clip: float = 1e6):
    x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    if clip is not None:
        x = torch.clamp(x, -clip, clip)
    return x


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )

    @staticmethod
    def apply_action_mask(logits: torch.Tensor, action_mask: torch.Tensor | None):
        logits = _safe_tensor(logits, clip=1e6)

        if action_mask is None:
            return logits, False

        mask = action_mask.to(dtype=torch.bool, device=logits.device)
        masked = logits.masked_fill(~mask, float("-inf"))

        bad = False
        if masked.dim() == 1:
            bad = torch.isneginf(masked).all().item()
        else:
            bad = torch.isneginf(masked).all(dim=-1).any().item()

        masked = _safe_tensor(masked, clip=1e6)
        return masked, bool(bad)

    def act(self, state: torch.Tensor, action_mask: torch.Tensor | None):
        state = _safe_tensor(state, clip=1e6)

        logits = self.actor(state)
        masked_logits, bad = self.apply_action_mask(logits, action_mask)

        if bad:
            masked_logits = torch.zeros_like(masked_logits)

        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        value = self.critic(state).squeeze(-1)
        value = _safe_tensor(value, clip=1e6)

        return action, action_logprob, value, bad

    def act_deterministic(self, state: torch.Tensor, action_mask: torch.Tensor | None):
        """评估用：mask 内 argmax（确定性）。"""
        state = _safe_tensor(state, clip=1e6)

        logits = self.actor(state)
        masked_logits, bad = self.apply_action_mask(logits, action_mask)

        if bad:
            # 没有可行动作：返回 0（上层必须统计 bad/fallback）
            return torch.tensor(0, device=state.device), True

        action = torch.argmax(masked_logits, dim=-1)
        return action, False

    def evaluate(self, states, actions, action_masks: torch.Tensor | None, bad_mask_flags: torch.Tensor | None):
        states = _safe_tensor(states, clip=1e6)

        logits = self.actor(states)
        if action_masks is None:
            masked_logits = _safe_tensor(logits, clip=1e6)
        else:
            mask = action_masks.to(dtype=torch.bool, device=logits.device)
            masked_logits = logits.masked_fill(~mask, float("-inf"))
            masked_logits = _safe_tensor(masked_logits, clip=1e6)

        if bad_mask_flags is not None and bad_mask_flags.any():
            masked_logits = masked_logits.clone()
            masked_logits[bad_mask_flags] = 0.0

        dist = Categorical(logits=masked_logits)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states).squeeze(-1)
        state_values = _safe_tensor(state_values, clip=1e6)

        return action_logprobs, state_values, dist_entropy


class PPOAgent:
    def __init__(
        self,
        state_dim=153,
        action_dim=120,
        device=None,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        K_epochs=4,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        minibatch_size=256,
        target_kl=0.02,
    ):
        self.device = device if device is not None else torch.device("cpu")

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse = nn.MSELoss()

        self.bad_mask_steps = 0
        self.last_update_stats = {}

    def select_action(self, state, action_mask=None):
        with torch.no_grad():
            state_t = state if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32)
            state_t = _safe_tensor(state_t.to(self.device), clip=1e6)

            if action_mask is None:
                mask_t = torch.ones(self.policy_old.action_dim, dtype=torch.bool, device=self.device)
            else:
                mask_t = action_mask if torch.is_tensor(action_mask) else torch.tensor(action_mask, dtype=torch.bool)
                mask_t = mask_t.to(self.device).to(dtype=torch.bool)

            action, logprob, value, bad = self.policy_old.act(state_t, action_mask=mask_t)

        self.buffer.states.append(state_t.detach().cpu())
        self.buffer.actions.append(action.detach().cpu())
        self.buffer.logprobs.append(logprob.detach().cpu())
        self.buffer.masks.append(mask_t.detach().cpu())
        self.buffer.values.append(value.detach().cpu())
        self.buffer.bad_masks.append(bool(bad))

        if bad:
            self.bad_mask_steps += 1

        return int(action.item())

    def select_action_deterministic(self, state, action_mask=None):
        with torch.no_grad():
            state_t = state if torch.is_tensor(state) else torch.tensor(state, dtype=torch.float32)
            state_t = _safe_tensor(state_t.to(self.device), clip=1e6)

            if action_mask is None:
                mask_t = torch.ones(self.policy_old.action_dim, dtype=torch.bool, device=self.device)
            else:
                mask_t = action_mask if torch.is_tensor(action_mask) else torch.tensor(action_mask, dtype=torch.bool)
                mask_t = mask_t.to(self.device).to(dtype=torch.bool)

            action, bad = self.policy_old.act_deterministic(state_t, action_mask=mask_t)
        return int(action.item()), bool(bad)

    def _compute_gae(self, rewards, dones, values, last_value=0.0):
        T = len(rewards)
        adv = torch.zeros(T, dtype=torch.float32, device=values.device)
        gae = 0.0

        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            nonterminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            adv[t] = gae

        returns = adv + values
        return adv, returns

    def update(self):
        T = len(self.buffer.states)
        if not (len(self.buffer.actions) == T == len(self.buffer.logprobs) == len(self.buffer.masks) == len(self.buffer.values) == len(self.buffer.bad_masks)):
            raise RuntimeError(
                f"[PPO] rollout core fields length mismatch: "
                f"states={len(self.buffer.states)}, actions={len(self.buffer.actions)}, "
                f"logprobs={len(self.buffer.logprobs)}, masks={len(self.buffer.masks)}, "
                f"values={len(self.buffer.values)}, bad_masks={len(self.buffer.bad_masks)}"
            )
        if not (len(self.buffer.rewards) == T == len(self.buffer.is_terminals)):
            raise RuntimeError(
                f"[PPO] reward/done length mismatch: "
                f"T={T}, rewards={len(self.buffer.rewards)}, dones={len(self.buffer.is_terminals)}. "
                f"Do NOT append reward/done when no action was taken."
            )

        if T == 0:
            self.last_update_stats = {"skipped": True}
            return

        states = torch.stack(self.buffer.states).to(self.device)
        actions = torch.stack(self.buffer.actions).to(self.device).view(-1)
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device).view(-1)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.buffer.is_terminals, dtype=torch.bool, device=self.device)
        values = torch.stack(self.buffer.values).to(self.device).view(-1)
        masks = torch.stack(self.buffer.masks).to(self.device)
        bad_flags = torch.tensor(self.buffer.bad_masks, dtype=torch.bool, device=self.device)

        states = _safe_tensor(states, clip=1e6)
        values = _safe_tensor(values, clip=1e6)
        old_logprobs = _safe_tensor(old_logprobs, clip=1e6)
        rewards = _safe_tensor(rewards, clip=1e6)

        with torch.no_grad():
            adv, returns = self._compute_gae(rewards, dones, values, last_value=0.0)
            adv = _safe_tensor(adv, clip=1e6)
            returns = _safe_tensor(returns, clip=1e6)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = torch.arange(T, device=self.device)

        last_actor_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        last_kl = 0.0
        updated_steps = 0

        for _epoch in range(self.K_epochs):
            perm = idx[torch.randperm(T)]
            approx_kl_sum = 0.0
            n_mb = 0

            for start in range(0, T, self.minibatch_size):
                mb_idx = perm[start:start + self.minibatch_size]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_adv = adv[mb_idx]
                mb_returns = returns[mb_idx]
                mb_masks = masks[mb_idx]
                mb_bad = bad_flags[mb_idx]

                logprobs, state_values, entropy = self.policy.evaluate(
                    mb_states, mb_actions, action_masks=mb_masks, bad_mask_flags=mb_bad
                )

                logprobs = _safe_tensor(logprobs, clip=1e6)
                state_values = _safe_tensor(state_values, clip=1e6)
                entropy = _safe_tensor(entropy, clip=1e6)

                ratios = torch.exp(logprobs - mb_old_logprobs)
                ratios = _safe_tensor(ratios, clip=1e6)

                surr1 = ratios * mb_adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = self.mse(state_values, mb_returns)
                entropy_loss = -entropy.mean()

                loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_logprobs - logprobs).mean()
                    approx_kl_sum += float(approx_kl)
                    n_mb += 1

                last_actor_loss = float(actor_loss.detach().cpu())
                last_value_loss = float(value_loss.detach().cpu())
                last_entropy = float(entropy.detach().mean().cpu())
                updated_steps += 1

            mean_kl = approx_kl_sum / max(1, n_mb)
            last_kl = float(mean_kl)

            if self.target_kl is not None and mean_kl > 1.5 * self.target_kl:
                break

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.last_update_stats = {
            "actor_loss": last_actor_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "kl": last_kl,
            "bad_mask_steps_in_rollout": int(bad_flags.sum().item()),
            "updated_minibatches": int(updated_steps),
        }

        self.buffer.clear()
        self.bad_mask_steps = 0

    def state_dict(self):
        return {
            "policy": self.policy.state_dict(),
            "policy_old": self.policy_old.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "eps_clip": self.eps_clip,
                "K_epochs": self.K_epochs,
                "entropy_coef": self.entropy_coef,
                "value_coef": self.value_coef,
                "max_grad_norm": self.max_grad_norm,
                "minibatch_size": self.minibatch_size,
                "target_kl": self.target_kl,
            }
        }

    def load_state_dict(self, ckpt, strict=True):
        self.policy.load_state_dict(ckpt["policy"], strict=strict)
        self.policy_old.load_state_dict(ckpt["policy_old"], strict=strict)
        self.optimizer.load_state_dict(ckpt["optimizer"])

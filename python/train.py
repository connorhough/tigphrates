"""
Tigris & Euphrates RL training script. Single-file PPO.
This is the file you modify — architecture, hyperparameters, reward shaping, etc.
Usage: python train.py
"""

import os
import sys
import time
import math
import copy
import glob
import random
import pathlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tigphrates_env import TigphratesEnv, ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS
from evaluate import (
    TIME_BUDGET,
    EVAL_GAMES,
    PLAYER_COUNT,
    evaluate_vs_heuristic,
    evaluate_vs_pool,
    update_persistent_elo,
    ELO_AGENT_KEY,
    print_summary,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

# PPO (env-var overridable for the sweep harness)
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 3e-4))
GAMMA = float(os.environ.get("GAMMA", 0.99))
GAE_LAMBDA = float(os.environ.get("GAE_LAMBDA", 0.95))
PPO_EPOCHS = int(os.environ.get("PPO_EPOCHS", 4))
CLIP_EPS = float(os.environ.get("CLIP_EPS", 0.2))
ENTROPY_COEF = float(os.environ.get("ENTROPY_COEF", 0.01))
VALUE_COEF = float(os.environ.get("VALUE_COEF", 0.5))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 0.5))
MINIBATCH_SIZE = int(os.environ.get("MINIBATCH_SIZE", 256))

# Rollout
ROLLOUT_STEPS = int(os.environ.get("ROLLOUT_STEPS", 1024))
MAX_EPISODE_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", 2000))

# Reward shaping
SCORE_DELTA_COEF = float(os.environ.get("SCORE_DELTA_COEF", 1.5))
SCORE_AVG_WEIGHT = float(os.environ.get("SCORE_AVG_WEIGHT", 0.5))
MARGIN_DELTA_COEF = float(os.environ.get("MARGIN_DELTA_COEF", 1.0))

# BC auxiliary (Phase 6.3)
BC_COEF = float(os.environ.get("BC_COEF", 0.1))
BC_QUERY_PROB = float(os.environ.get("BC_QUERY_PROB", 0.25))

# Architecture (env-var overridable so a sweep can scale capacity)
BOARD_CONV_CHANNELS = int(os.environ.get("BOARD_CONV_CHANNELS", 32))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 256))
NUM_HIDDEN_LAYERS = int(os.environ.get("NUM_HIDDEN_LAYERS", 2))

# Compute device. Apple Silicon Mac minis have a Metal GPU exposed via the
# MPS backend — picks up a 3-5× speedup on inference + training over CPU.
# Falls back to CUDA, then CPU. Override with TORCH_DEVICE=cpu for debugging.
def _pick_device() -> torch.device:
    forced = os.environ.get("TORCH_DEVICE")
    if forced:
        return torch.device(forced)
    # On Mac mini M-series, MPS provides a 3-5× speedup for the PPO update
    # batch (256-row minibatches), but adds ~5-10s of one-time kernel-compile
    # warmup AND adds per-call transfer overhead that hurts small (B≤4)
    # rollout inference. Net benefit appears at TIME_BUDGET >= 60s with the
    # default model size; smaller experiments are faster on CPU.
    # Default to CPU; opt in with TORCH_DEVICE=mps once the warmup amortizes.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _pick_device()

# Self-play / league (env-var overridable so sweeps can isolate per-run pools)
RUN_DIR = pathlib.Path(os.environ.get("RUN_DIR", "models"))
POOL_DIR = pathlib.Path(os.environ.get("POOL_DIR", str(RUN_DIR / "pool")))
POOL_OPPONENT_PROB = float(os.environ.get("POOL_OPPONENT_PROB", 0.5))
POOL_MAX_SIZE = int(os.environ.get("POOL_MAX_SIZE", 12))
TRAIN_VS_HEURISTIC_PROB = float(os.environ.get("TRAIN_VS_HEURISTIC_PROB", 0.25))

# Curriculum: optionally decay the heuristic-opponent share over training so
# early rollouts are dominated by a stable opponent (heuristic) and later
# rollouts shift toward self-play / league. Smoother early learning when the
# pool would otherwise be full of weak random snapshots.
CURRICULUM_ENABLED = os.environ.get("CURRICULUM_ENABLED", "0") == "1"
CURRICULUM_HEURISTIC_START = float(os.environ.get("CURRICULUM_HEURISTIC_START", 1.0))
CURRICULUM_HEURISTIC_END = float(os.environ.get("CURRICULUM_HEURISTIC_END", 0.1))

# League scheduler granularity. "per_episode" picks a fresh opponent for an
# env when its episode resets, so a single rollout exposes the policy to
# many different opponents. "per_rollout" uses one opponent for all envs
# until the next rollout — simpler but less diverse.
LEAGUE_SCHEDULER = "per_episode"

# Parallel rollouts
NUM_ENVS = 4              # parallel Tigphrates envs (one Node subprocess each)

# ---------------------------------------------------------------------------
# Policy Network
# ---------------------------------------------------------------------------

class BoardEncoder(nn.Module):
    """CNN encoder for the 13-channel 11x16 board tensor."""
    def __init__(self, in_channels=BOARD_CHANNELS, out_channels=BOARD_CONV_CHANNELS):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_dim = out_channels * BOARD_ROWS * BOARD_COLS

    def forward(self, x):
        # x: (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.view(x.size(0), -1)  # flatten


class PolicyValueNetwork(nn.Module):
    """Actor-critic network with CNN board encoder + MLP head."""
    def __init__(self):
        super().__init__()
        self.board_encoder = BoardEncoder()

        # Hand_seq is 6 slots × 5-way one-hot (4 colors + empty) = 30 features.
        # Lets the policy reason about WHICH tile sits at each hand position so
        # position-indexed swap actions (server.ts BASE_SWAP) carry meaningful
        # signal instead of being effectively random over a count vector.
        # Extra features: hand(4) + hand_seq(30) + scores(4) + meta(8) +
        # conflict(7) + leaders(8) + opp_scores(4) + opp_leaders(8) = 73
        extra_dim = 4 + 30 + 4 + 8 + 7 + 8 + 4 + 8
        trunk_input_dim = self.board_encoder.out_dim + extra_dim

        # Shared trunk
        layers = []
        in_dim = trunk_input_dim
        for _ in range(NUM_HIDDEN_LAYERS):
            layers.append(nn.Linear(in_dim, HIDDEN_DIM))
            layers.append(nn.ReLU())
            in_dim = HIDDEN_DIM
        self.trunk = nn.Sequential(*layers)

        # Policy head
        self.policy_head = nn.Linear(HIDDEN_DIM, ACTION_SPACE_SIZE)

        # Value head
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init for policy and value output
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs):
        board = obs["board"]           # (B, 15, 11, 16)
        hand = obs["hand"]             # (B, 4)
        hand_seq = obs["hand_seq"]     # (B, 6) ints in {-1, 0, 1, 2, 3}
        scores = obs["scores"]         # (B, 4)
        meta = obs["meta"]             # (B, 8)
        conflict = obs["conflict"]     # (B, 7)
        leaders = obs["leaders"]       # (B, 8)
        opp_scores = obs["opp_scores"] # (B, 4)
        opp_leaders = obs["opp_leaders"] # (B, 8)

        # One-hot hand_seq: shape (B, 6, 5) → flatten to (B, 30). Slot 4 = empty.
        # Map -1 → 4, then one_hot. Cast to int64 first since one_hot needs long.
        idx = hand_seq.to(torch.int64)
        idx = torch.where(idx < 0, torch.full_like(idx, 4), idx)
        hand_seq_oh = F.one_hot(idx, num_classes=5).to(torch.float32).reshape(idx.size(0), -1)

        board_features = self.board_encoder(board)
        extra = torch.cat([hand, hand_seq_oh, scores, meta, conflict, leaders, opp_scores, opp_leaders], dim=-1)
        combined = torch.cat([board_features, extra], dim=-1)

        trunk_out = self.trunk(combined)
        logits = self.policy_head(trunk_out)
        value = self.value_head(trunk_out).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs, action_mask, action=None):
        logits, value = self.forward(obs)

        # Mask invalid actions. Place mask on the same device as logits so
        # the masked_fill stays on GPU when DEVICE is mps/cuda.
        mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(~mask_tensor, float("-inf"))

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

# ---------------------------------------------------------------------------
# Observation batching utilities
# ---------------------------------------------------------------------------

def obs_to_tensors(obs):
    """Convert a single observation dict to batched tensors (B=1) on DEVICE."""
    return {
        "board": torch.tensor(obs["board"], dtype=torch.float32, device=DEVICE).unsqueeze(0),
        "hand": torch.tensor(obs["hand"], dtype=torch.float32, device=DEVICE).unsqueeze(0),
        "hand_seq": torch.tensor(obs["hand_seq"], dtype=torch.int64, device=DEVICE).unsqueeze(0),
        "scores": torch.tensor(obs["scores"], dtype=torch.float32, device=DEVICE).unsqueeze(0),
        "meta": torch.tensor(obs["meta"], dtype=torch.float32, device=DEVICE).unsqueeze(0),
        "conflict": torch.tensor(obs["conflict"], dtype=torch.float32, device=DEVICE).unsqueeze(0),
        "leaders": torch.tensor(obs["leaders"], dtype=torch.float32, device=DEVICE).unsqueeze(0),
        "opp_scores": torch.tensor(obs["opp_scores"], dtype=torch.float32, device=DEVICE).unsqueeze(0),
        "opp_leaders": torch.tensor(obs["opp_leaders"], dtype=torch.float32, device=DEVICE).unsqueeze(0),
    }

def stack_obs(obs_list):
    """Stack a list of observation dicts into batched tensors on DEVICE."""
    return {
        "board": torch.stack([torch.tensor(o["board"], dtype=torch.float32) for o in obs_list]).to(DEVICE),
        "hand": torch.stack([torch.tensor(o["hand"], dtype=torch.float32) for o in obs_list]).to(DEVICE),
        "hand_seq": torch.stack([torch.tensor(o["hand_seq"], dtype=torch.int64) for o in obs_list]).to(DEVICE),
        "scores": torch.stack([torch.tensor(o["scores"], dtype=torch.float32) for o in obs_list]).to(DEVICE),
        "meta": torch.stack([torch.tensor(o["meta"], dtype=torch.float32) for o in obs_list]).to(DEVICE),
        "conflict": torch.stack([torch.tensor(o["conflict"], dtype=torch.float32) for o in obs_list]).to(DEVICE),
        "leaders": torch.stack([torch.tensor(o["leaders"], dtype=torch.float32) for o in obs_list]).to(DEVICE),
        "opp_scores": torch.stack([torch.tensor(o["opp_scores"], dtype=torch.float32) for o in obs_list]).to(DEVICE),
        "opp_leaders": torch.stack([torch.tensor(o["opp_leaders"], dtype=torch.float32) for o in obs_list]).to(DEVICE),
    }

# ---------------------------------------------------------------------------
# PPO Training
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """Compute Generalized Advantage Estimation."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val = next_value if t == n - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * lam * non_terminal * last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


class VecTigphratesEnv:
    """Holds N independent TigphratesEnv instances. Each env owns its own
    Node bridge subprocess, so bridge calls from different envs proceed in
    parallel. A ThreadPoolExecutor overlaps the per-env bridge waits — Python
    releases the GIL on subprocess.readline / write, so wall time scales
    sub-linearly with N. Model inference is also batched across envs each
    outer step."""

    def __init__(self, n_envs: int, **env_kwargs):
        self.envs = [TigphratesEnv(**env_kwargs) for _ in range(n_envs)]
        self.n = n_envs
        self._executor = ThreadPoolExecutor(max_workers=max(1, n_envs))

    def reset_all(self):
        return list(self._executor.map(lambda e: e.reset(), self.envs))

    def action_masks(self):
        return list(self._executor.map(lambda e: e.action_mask(), self.envs))

    def step_all(self, actions: list[int]) -> list[tuple]:
        """Step each env with its action, in parallel. Returns the list of
        (obs, reward, terminated, truncated, info) tuples."""
        def _step(pair):
            env, action = pair
            return env.step(action)
        return list(self._executor.map(_step, zip(self.envs, actions)))

    def set_opponent_policy(self, policy):
        for e in self.envs:
            e.opponent_policy = policy

    def close(self):
        try:
            self._executor.shutdown(wait=True)
        except Exception:
            pass
        for e in self.envs:
            try:
                e.close()
            except Exception:
                pass


def collect_rollout_vec(
    env_vec: VecTigphratesEnv,
    model,
    rollout_steps: int,
    on_reset=None,
):
    """Vectorized rollout. Splits rollout_steps across env_vec.n envs; total
    transitions == rollout_steps. Single batched forward pass per outer step
    over all envs.

    on_reset(env_idx) is called after each env reset (initial reset and
    after `done`). Used by the per-episode league scheduler to resample an
    opponent so a single rollout exposes the policy to many different
    opponents."""
    n = env_vec.n
    per_env_obs: list[list[dict]] = [[] for _ in range(n)]
    per_env_actions: list[list[int]] = [[] for _ in range(n)]
    per_env_log_probs: list[list[float]] = [[] for _ in range(n)]
    per_env_rewards: list[list[float]] = [[] for _ in range(n)]
    per_env_dones: list[list[float]] = [[] for _ in range(n)]
    per_env_values: list[list[float]] = [[] for _ in range(n)]
    per_env_masks: list[list[np.ndarray]] = [[] for _ in range(n)]
    per_env_expert: list[list[int]] = [[] for _ in range(n)]  # -1 = not queried

    episode_rewards: list[float] = []
    current_ep_rewards = [0.0] * n
    steps_collected = 0

    obs_per_env: list[dict] = [None] * n
    reset_results = env_vec.reset_all()
    for i, (o, _) in enumerate(reset_results):
        obs_per_env[i] = o
        if on_reset is not None:
            on_reset(i)

    outer_steps = max(1, rollout_steps // n)
    for _ in range(outer_steps):
        masks = env_vec.action_masks()
        # Reset envs that have no valid action.
        for i in range(n):
            if masks[i].sum() == 0:
                episode_rewards.append(current_ep_rewards[i])
                current_ep_rewards[i] = 0.0
                obs_per_env[i], _ = env_vec.envs[i].reset()
                if on_reset is not None:
                    on_reset(i)
                masks[i] = env_vec.envs[i].action_mask()

        # Skip this iteration if any env still has no valid actions
        # (pathological — pick action 0 which will be remapped to a no-op
        # downstream by the env's bookkeeping).
        active_mask = np.array([m.sum() > 0 for m in masks])

        batched_obs = stack_obs(obs_per_env)
        batched_masks = np.stack(masks, axis=0)
        with torch.no_grad():
            logits, values_t = model.forward(batched_obs)
            mask_t = torch.tensor(batched_masks, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask_t, float("-inf"))
            dist = Categorical(logits=logits)
            sampled = dist.sample()
            log_probs_t = dist.log_prob(sampled)

        # Snapshot pre-step shaping inputs per env.
        prev_metrics = []
        for i in range(n):
            scores = obs_per_env[i]["scores"][:4]
            prev_min = float(np.min(scores))
            prev_avg = float(np.mean(scores))
            prev_opp_min = float(np.min(obs_per_env[i]["opp_scores"][:4]))
            prev_metrics.append((prev_min, prev_avg, prev_opp_min - prev_min, prev_min - prev_opp_min))

        # Append pre-step transition data (no env mutation here). For a
        # fraction of steps we also record what the heuristic AI would have
        # done at this state, used downstream as the BC auxiliary target.
        actions_list: list[int] = []
        for i in range(n):
            if not active_mask[i]:
                actions_list.append(0)
                continue
            actions_list.append(sampled[i].item())
            per_env_obs[i].append(obs_per_env[i])
            per_env_actions[i].append(sampled[i].item())
            per_env_log_probs[i].append(log_probs_t[i].item())
            per_env_values[i].append(values_t[i].item())
            per_env_masks[i].append(masks[i])
            if random.random() < BC_QUERY_PROB:
                try:
                    expert_idx = env_vec.envs[i].expert_action_index()
                except Exception:
                    expert_idx = -1
                per_env_expert[i].append(expert_idx)
            else:
                per_env_expert[i].append(-1)

        # Step all envs in parallel — each env owns its own bridge subprocess.
        step_results = env_vec.step_all(actions_list)

        for i in range(n):
            if not active_mask[i]:
                continue
            obs_new, reward, terminated, truncated, _ = step_results[i]
            done = terminated or truncated

            prev_min, prev_avg, _ignore, prev_margin = prev_metrics[i]
            new_min = float(np.min(obs_new["scores"][:4]))
            new_avg = float(np.mean(obs_new["scores"][:4]))
            new_opp_min = float(np.min(obs_new["opp_scores"][:4]))
            new_margin = new_min - new_opp_min

            min_delta = new_min - prev_min
            avg_delta = new_avg - prev_avg
            blended = (1.0 - SCORE_AVG_WEIGHT) * min_delta + SCORE_AVG_WEIGHT * avg_delta
            margin_delta = new_margin - prev_margin
            shaped = reward + SCORE_DELTA_COEF * blended + MARGIN_DELTA_COEF * margin_delta

            per_env_rewards[i].append(shaped)
            per_env_dones[i].append(float(done))
            current_ep_rewards[i] += shaped
            steps_collected += 1

            if done:
                episode_rewards.append(current_ep_rewards[i])
                current_ep_rewards[i] = 0.0
                obs_new, _ = env_vec.envs[i].reset()
                if on_reset is not None:
                    on_reset(i)
            obs_per_env[i] = obs_new

    # Bootstrap values for last states.
    last_values = []
    for i in range(n):
        next_mask = env_vec.envs[i].action_mask()
        if next_mask.sum() == 0:
            last_values.append(0.0)
            continue
        obs_t = obs_to_tensors(obs_per_env[i])
        with torch.no_grad():
            _, _, _, v = model.get_action_and_value(obs_t, next_mask)
        last_values.append(v.item())

    # Per-env GAE, then concatenate.
    all_obs: list = []
    all_actions: list = []
    all_log_probs: list = []
    all_values: list = []
    all_masks: list = []
    all_adv: list = []
    all_ret: list = []
    all_expert: list = []
    for i in range(n):
        if not per_env_obs[i]:
            continue
        adv, ret = compute_gae(per_env_rewards[i], per_env_values[i], per_env_dones[i], last_values[i])
        all_obs.extend(per_env_obs[i])
        all_actions.extend(per_env_actions[i])
        all_log_probs.extend(per_env_log_probs[i])
        all_values.extend(per_env_values[i])
        all_masks.extend(per_env_masks[i])
        all_adv.extend(adv.tolist())
        all_ret.extend(ret.tolist())
        all_expert.extend(per_env_expert[i])

    return {
        "obs": all_obs,
        "actions": np.array(all_actions),
        "log_probs": np.array(all_log_probs, dtype=np.float32),
        "advantages": np.array(all_adv, dtype=np.float32),
        "returns": np.array(all_ret, dtype=np.float32),
        "masks": all_masks,
        "values": np.array(all_values, dtype=np.float32),
        "expert_actions": np.array(all_expert, dtype=np.int64),
        "episode_rewards": episode_rewards,
        "steps": steps_collected,
    }


def collect_rollout(env, model, rollout_steps):
    """Collect a rollout of experience from the environment."""
    obs_list = []
    action_list = []
    log_prob_list = []
    reward_list = []
    done_list = []
    value_list = []
    mask_list = []

    obs, info = env.reset()
    episode_rewards = []
    current_episode_reward = 0.0
    steps_collected = 0

    for _ in range(rollout_steps):
        action_mask = env.action_mask()

        # If no valid actions, treat as episode end
        if action_mask.sum() == 0:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            obs, info = env.reset()
            continue

        obs_tensor = obs_to_tensors(obs)

        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(
                obs_tensor,
                action_mask,
            )

        obs_list.append(obs)
        action_list.append(action.item())
        log_prob_list.append(log_prob.item())
        value_list.append(value.item())
        mask_list.append(action_mask)

        # Record score before step for reward shaping
        prev_scores = obs["scores"][:4]
        prev_min_score = float(np.min(prev_scores))
        prev_avg_score = float(np.mean(prev_scores))
        prev_opp_min = float(np.min(obs["opp_scores"][:4]))
        prev_margin = prev_min_score - prev_opp_min

        obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        # Dense reward: blend of min-score delta (aligns with objective) and avg-score delta (rewards any scoring)
        new_scores = obs["scores"][:4]
        new_min_score = float(np.min(new_scores))
        new_avg_score = float(np.mean(new_scores))
        min_delta = new_min_score - prev_min_score
        avg_delta = new_avg_score - prev_avg_score
        blended_delta = (1.0 - SCORE_AVG_WEIGHT) * min_delta + SCORE_AVG_WEIGHT * avg_delta
        reward = reward + SCORE_DELTA_COEF * blended_delta

        # Margin-based reward: relative score improvement vs opponent
        new_opp_min = float(np.min(obs["opp_scores"][:4]))
        new_margin = new_min_score - new_opp_min
        margin_delta = new_margin - prev_margin
        reward = reward + MARGIN_DELTA_COEF * margin_delta

        reward_list.append(reward)
        done_list.append(float(done))
        current_episode_reward += reward
        steps_collected += 1

        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            obs, info = env.reset()

    # Handle edge case: no data collected
    if len(obs_list) == 0:
        return {
            "obs": [], "actions": np.array([]), "log_probs": np.array([], dtype=np.float32),
            "advantages": np.array([], dtype=np.float32), "returns": np.array([], dtype=np.float32),
            "masks": [], "values": np.array([], dtype=np.float32),
            "episode_rewards": episode_rewards, "steps": 0,
        }

    # Bootstrap value for last state
    with torch.no_grad():
        next_mask = env.action_mask()
        if next_mask.sum() == 0:
            next_value = 0.0
        else:
            next_obs_tensor = obs_to_tensors(obs)
            _, _, _, next_value = model.get_action_and_value(next_obs_tensor, next_mask)
            next_value = next_value.item()

    advantages, returns = compute_gae(reward_list, value_list, done_list, next_value)

    return {
        "obs": obs_list,
        "actions": np.array(action_list),
        "log_probs": np.array(log_prob_list, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "masks": mask_list,
        "values": np.array(value_list, dtype=np.float32),
        "episode_rewards": episode_rewards,
        "steps": steps_collected,
    }


def ppo_update(model, optimizer, rollout):
    """Run PPO optimization epochs on collected rollout data."""
    n = len(rollout["obs"])
    if n == 0:
        return {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0}
    obs_batch = stack_obs(rollout["obs"])
    actions = torch.tensor(rollout["actions"], dtype=torch.long, device=DEVICE)
    old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32, device=DEVICE)
    advantages = torch.tensor(rollout["advantages"], dtype=torch.float32, device=DEVICE)
    returns = torch.tensor(rollout["returns"], dtype=torch.float32, device=DEVICE)
    expert_actions = torch.tensor(
        rollout.get("expert_actions", np.full(n, -1, dtype=np.int64)),
        dtype=torch.long, device=DEVICE,
    )

    # Normalize advantages
    if advantages.std() > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_pg_loss = 0
    total_vf_loss = 0
    total_entropy = 0
    total_bc_loss = 0.0
    num_updates = 0

    for _ in range(PPO_EPOCHS):
        indices = np.random.permutation(n)
        for start in range(0, n, MINIBATCH_SIZE):
            end = min(start + MINIBATCH_SIZE, n)
            mb_idx = indices[start:end]

            mb_obs = {k: v[mb_idx] for k, v in obs_batch.items()}
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]
            mb_expert = expert_actions[mb_idx]

            # Build combined mask for this minibatch
            mb_masks = np.array([rollout["masks"][i] for i in mb_idx])

            # Sanity: every taken action must be inside its own mask. Catches
            # off-by-one bugs in mask construction or stale masks after a
            # phase-changing engine event. Cheap — one indexed lookup per row.
            if __debug__:
                taken = mb_actions.cpu().numpy()
                rows = np.arange(len(taken))
                if not mb_masks[rows, taken].all():
                    bad = np.where(mb_masks[rows, taken] == 0)[0]
                    raise AssertionError(
                        f"PPO minibatch contains {len(bad)} action(s) outside their own mask "
                        f"(first bad row: action={taken[bad[0]]}, mask sum={mb_masks[bad[0]].sum()})"
                    )

            logits, values = model.forward(mb_obs)
            mask_tensor = torch.tensor(mb_masks, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask_tensor, float("-inf"))

            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            # PPO clipped objective
            ratio = (new_log_probs - mb_old_log_probs).exp()
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
            pg_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            vf_loss = F.mse_loss(values, mb_returns)

            # Auxiliary BC loss: cross-entropy against the heuristic's chosen
            # action on the subset of rows where it was queried (expert >= 0).
            # Keeps the policy from drifting too far from heuristic-quality
            # play during exploration; small weight so PPO still drives.
            bc_mask = mb_expert >= 0
            if bc_mask.any():
                bc_logits = logits[bc_mask]
                bc_targets = mb_expert[bc_mask]
                bc_loss = F.cross_entropy(bc_logits, bc_targets)
            else:
                bc_loss = torch.tensor(0.0)

            # Total loss
            loss = (
                pg_loss
                + VALUE_COEF * vf_loss
                - ENTROPY_COEF * entropy
                + BC_COEF * bc_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy.item()
            total_bc_loss += float(bc_loss.item())
            num_updates += 1

    return {
        "pg_loss": total_pg_loss / max(num_updates, 1),
        "vf_loss": total_vf_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
        "bc_loss": total_bc_loss / max(num_updates, 1),
    }

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def make_policy_fn(model):
    """Create a policy function for evaluation."""
    def policy_fn(obs, action_mask):
        obs_tensor = obs_to_tensors(obs)
        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs_tensor, action_mask)
        return action.item()
    return policy_fn


# ---------------------------------------------------------------------------
# Self-play pool / league
# ---------------------------------------------------------------------------

def _ensure_pool_dir():
    POOL_DIR.mkdir(parents=True, exist_ok=True)


def list_pool() -> list[pathlib.Path]:
    if not POOL_DIR.exists():
        return []
    return sorted(POOL_DIR.glob("policy_*.pt"))


def add_to_pool(model: nn.Module, label: str | None = None) -> pathlib.Path:
    """Snapshot the current model into the pool. Prunes oldest if over cap."""
    _ensure_pool_dir()
    if label is None:
        label = f"{int(time.time())}"
    path = POOL_DIR / f"policy_{label}.pt"
    torch.save(model.state_dict(), path)
    members = list_pool()
    while len(members) > POOL_MAX_SIZE:
        oldest = members.pop(0)
        try:
            oldest.unlink()
        except OSError:
            pass
    return path


def load_policy_from_path(path: str) -> nn.Module:
    m = PolicyValueNetwork()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.train(False)
    m.to(DEVICE)
    return m


def heuristic_prob_now(elapsed: float, total: float) -> float:
    """Linear curriculum from CURRICULUM_HEURISTIC_START → END over training.
    Returns the static TRAIN_VS_HEURISTIC_PROB when curriculum is off."""
    if not CURRICULUM_ENABLED:
        return TRAIN_VS_HEURISTIC_PROB
    t = max(0.0, min(1.0, elapsed / max(total, 1.0)))
    return CURRICULUM_HEURISTIC_START * (1 - t) + CURRICULUM_HEURISTIC_END * t


def sample_opponent_policy(current_model: nn.Module, heuristic_prob: float | None = None):
    """Pick an opponent for self-play. Returns (policy_fn or None, label).
    None means use the heuristic AI (env's default). Otherwise the policy_fn
    closure is callable as policy_fn(obs, mask) -> int.

    `heuristic_prob` overrides TRAIN_VS_HEURISTIC_PROB — used by the
    curriculum to inject a time-varying mix without rewiring globals."""
    h = TRAIN_VS_HEURISTIC_PROB if heuristic_prob is None else heuristic_prob
    pool = list_pool()
    r = random.random()
    if r < h:
        return None, "heuristic"
    if pool and r < h + POOL_OPPONENT_PROB:
        path = random.choice(pool)
        opp = load_policy_from_path(str(path))
        return make_policy_fn(opp), f"pool:{path.stem}"
    snap = copy.deepcopy(current_model)
    snap.train(False)
    return make_policy_fn(snap), "self"


if __name__ == "__main__":
    t_start = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    # Build model + optimizer. Three startup paths in priority order:
    #   1. RESUME_FROM env var → resume from a checkpoint that includes
    #      optimizer + RNG state (long-running session survives interruption).
    #   2. <RUN_DIR>/policy_bc.pt exists → BC warm start, fresh optimizer.
    #   3. Random init.
    model = PolicyValueNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    resume_path_env = os.environ.get("RESUME_FROM")
    resumed = False
    if resume_path_env:
        resume_path = pathlib.Path(resume_path_env)
        if resume_path.exists():
            # weights_only=False so the numpy RNG state inside the checkpoint
            # (which is a numpy object, not a tensor) loads. We generated
            # the file ourselves in this same script, so it is trusted.
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                if "torch_rng" in ckpt:
                    torch.set_rng_state(ckpt["torch_rng"])
                if "numpy_rng" in ckpt:
                    np.random.set_state(ckpt["numpy_rng"])
                resumed = True
                print(f"Resumed from {resume_path}")
            else:
                # Bare state_dict: load weights, fresh optimizer.
                model.load_state_dict(ckpt)
                resumed = True
                print(f"Loaded weights from {resume_path} (fresh optimizer)")
        else:
            print(f"RESUME_FROM={resume_path} not found; starting fresh")

    if not resumed:
        bc_path = RUN_DIR / "policy_bc.pt"
        if bc_path.exists():
            try:
                model.load_state_dict(torch.load(bc_path, map_location="cpu"))
                print(f"Loaded BC warm start from {bc_path}")
            except Exception as e:
                print(f"Failed to load BC warm start ({e}); starting from random init")
    model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Action space: {ACTION_SPACE_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Time budget: {TIME_BUDGET}s")
    print()

    # Create vectorized environments. Each env owns its own Node bridge
    # subprocess. Opponent policy is hot-swapped per rollout for self-play /
    # league sampling — start with the heuristic anchor.
    env_vec = VecTigphratesEnv(
        n_envs=NUM_ENVS,
        player_count=PLAYER_COUNT,
        agent_player=0,
        max_turns=MAX_EPISODE_STEPS,
        opponent_policy=None,
    )

    # Seed pool with the initial random policy so early rollouts have a non-
    # heuristic opponent to learn against. Subsequent snapshots get added
    # whenever a rollout finishes (capped at POOL_MAX_SIZE).
    if not list_pool():
        add_to_pool(model, label="init")

    # Training loop
    total_training_time = 0.0
    total_steps = 0
    total_episodes = 0
    rollout_num = 0
    all_episode_rewards = []
    opponent_label = "heuristic"

    while total_training_time < TIME_BUDGET:
        t0 = time.time()

        # Pick opponents. In per_episode mode, each env gets its own opponent
        # at start (and resamples on reset inside the rollout) for a wider
        # mix per training batch; in per_rollout mode, one global opponent
        # is shared by all envs until the next rollout.
        h_prob = heuristic_prob_now(total_training_time, TIME_BUDGET)
        if LEAGUE_SCHEDULER == "per_episode":
            labels = []
            for i in range(NUM_ENVS):
                opp_policy_i, label_i = sample_opponent_policy(model, heuristic_prob=h_prob)
                env_vec.envs[i].opponent_policy = opp_policy_i
                labels.append(label_i)
            opponent_label = ",".join(sorted(set(labels)))[:18]

            def _on_reset(env_idx: int, _h=h_prob) -> None:
                opp, _ = sample_opponent_policy(model, heuristic_prob=_h)
                env_vec.envs[env_idx].opponent_policy = opp
            on_reset_cb = _on_reset
        else:
            opp_policy, opponent_label = sample_opponent_policy(model, heuristic_prob=h_prob)
            env_vec.set_opponent_policy(opp_policy)
            on_reset_cb = None

        rollout = collect_rollout_vec(env_vec, model, ROLLOUT_STEPS, on_reset=on_reset_cb)
        total_steps += rollout["steps"]
        total_episodes += len(rollout["episode_rewards"])
        all_episode_rewards.extend(rollout["episode_rewards"])

        losses = ppo_update(model, optimizer, rollout)

        t1 = time.time()
        dt = t1 - t0
        total_training_time += dt
        rollout_num += 1

        # Periodically snapshot current model into the pool.
        if rollout_num % 4 == 0:
            add_to_pool(model, label=f"r{rollout_num:04d}")

        remaining = max(0, TIME_BUDGET - total_training_time)
        pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
        recent_rewards = all_episode_rewards[-10:] if all_episode_rewards else [0]
        avg_reward = np.mean(recent_rewards)

        print(
            f"\rrollout {rollout_num:04d} ({pct_done:.1f}%) | "
            f"opp: {opponent_label[:18]:<18s} | "
            f"pg: {losses['pg_loss']:.3f} | "
            f"vf: {losses['vf_loss']:.3f} | "
            f"H: {losses['entropy']:.2f} | "
            f"R: {avg_reward:.3f} | "
            f"eps: {total_episodes} | "
            f"steps: {total_steps} | "
            f"left: {remaining:.0f}s   ",
            end="", flush=True,
        )

    print()  # newline after \r log
    # Final pool snapshot.
    add_to_pool(model, label=f"final_r{rollout_num:04d}")
    env_vec.close()

    t_end_training = time.time()

    # --- Evaluation ---
    print(f"\nEvaluating over {EVAL_GAMES} games vs heuristic AI...")
    model.train(False)
    policy_fn = make_policy_fn(model)
    eval_results = evaluate_vs_heuristic(policy_fn, num_games=EVAL_GAMES)

    # --- League evaluation: vs every snapshot in the pool ---
    pool_paths = [str(p) for p in list_pool()]
    games_per_opp = max(2, EVAL_GAMES // max(len(pool_paths), 1))
    pool_results = evaluate_vs_pool(
        policy_fn,
        opponent_loader=load_policy_from_path,
        opponent_paths=pool_paths,
        games_per_opponent=games_per_opp,
        max_turns=MAX_EPISODE_STEPS,
    )

    # Persistent Elo ladder. Update both agent and each opponent rating in
    # models/pool/elo.json so the league has memory across runs.
    if pool_results.get("per_opponent"):
        elo_table = update_persistent_elo(
            POOL_DIR,
            pool_results["per_opponent"],
            games_per_opponent=games_per_opp,
        )
        pool_results["persistent_elo"] = elo_table.get(ELO_AGENT_KEY)

    t_end = time.time()

    # --- Summary ---
    print_summary(
        win_rate=eval_results["win_rate"],
        avg_min_score=eval_results["avg_min_score"],
        avg_margin=eval_results["avg_margin"],
        training_seconds=total_training_time,
        total_seconds=t_end - t_start,
        num_episodes=total_episodes,
        num_steps=total_steps,
        num_params=num_params,
        vs_pool_win_rate=pool_results.get("vs_pool_win_rate"),
        pool_size=pool_results.get("n_opponents", 0),
        elo=pool_results.get("elo"),
        persistent_elo=pool_results.get("persistent_elo"),
    )

    # --- Save model ---
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    model_path = RUN_DIR / "policy_final.pt"
    torch.save(model.state_dict(), str(model_path))
    print(f"\nModel saved to {model_path}")

    # Also save a resumable checkpoint with optimizer + RNG state so a
    # long training session can survive interruption (Mac mini sleep,
    # power blip, etc.). Pass RESUME_FROM=<path> to pick up where this run
    # left off.
    resume_path = RUN_DIR / "checkpoint_resumable.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }, str(resume_path))
    print(f"Resumable checkpoint saved to {resume_path}")

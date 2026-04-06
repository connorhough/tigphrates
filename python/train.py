"""
Tigris & Euphrates RL training script. Single-file PPO.
This is the file you modify — architecture, hyperparameters, reward shaping, etc.
Usage: python train.py
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tigphrates_env import TigphratesEnv, ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS
from evaluate import TIME_BUDGET, EVAL_GAMES, PLAYER_COUNT, evaluate_vs_heuristic, print_summary

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

# PPO
LEARNING_RATE = 3e-4
GAMMA = 0.99              # discount factor
GAE_LAMBDA = 0.95         # GAE lambda — reverted to standard; 0.98 degraded performance
PPO_EPOCHS = 4            # optimization epochs per rollout — halved to reduce policy overfitting per batch
CLIP_EPS = 0.2            # PPO clipping epsilon
ENTROPY_COEF = 0.01       # entropy bonus coefficient
VALUE_COEF = 0.5          # value loss coefficient
MAX_GRAD_NORM = 0.5       # gradient clipping
# PPO_EPOCHS moved above (set to 8)
MINIBATCH_SIZE = 256      # minibatch size for PPO updates — larger batches stabilize gradient estimates

# Rollout
ROLLOUT_STEPS = 1024      # steps per rollout before update — larger for sparse-reward long games
MAX_EPISODE_STEPS = 2000  # max steps per episode

# Reward shaping
SCORE_DELTA_COEF = 1.5    # weight on per-step score delta — further boosted to amplify learning signal
SCORE_AVG_WEIGHT = 0.5    # blend: 0=min-score only, 1=avg-score only (avg encourages any scoring, min aligns with objective)

# Architecture
BOARD_CONV_CHANNELS = 32  # channels in board CNN
HIDDEN_DIM = 256          # MLP hidden dimension — doubled to increase policy capacity for complex game state
NUM_HIDDEN_LAYERS = 2     # number of hidden layers in MLP head

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

        # Extra features: hand(4) + scores(4) + meta(8) + conflict(7) = 23
        extra_dim = 4 + 4 + 8 + 7
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
        board = obs["board"]       # (B, 13, 11, 16)
        hand = obs["hand"]         # (B, 4)
        scores = obs["scores"]     # (B, 4)
        meta = obs["meta"]         # (B, 8)
        conflict = obs["conflict"] # (B, 7)

        board_features = self.board_encoder(board)
        extra = torch.cat([hand, scores, meta, conflict], dim=-1)
        combined = torch.cat([board_features, extra], dim=-1)

        trunk_out = self.trunk(combined)
        logits = self.policy_head(trunk_out)
        value = self.value_head(trunk_out).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs, action_mask, action=None):
        logits, value = self.forward(obs)

        # Mask invalid actions
        mask_tensor = torch.tensor(action_mask, dtype=torch.bool)
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
    """Convert a single observation dict to batched tensors (B=1)."""
    return {
        "board": torch.tensor(obs["board"], dtype=torch.float32).unsqueeze(0),
        "hand": torch.tensor(obs["hand"], dtype=torch.float32).unsqueeze(0),
        "scores": torch.tensor(obs["scores"], dtype=torch.float32).unsqueeze(0),
        "meta": torch.tensor(obs["meta"], dtype=torch.float32).unsqueeze(0),
        "conflict": torch.tensor(obs["conflict"], dtype=torch.float32).unsqueeze(0),
    }

def stack_obs(obs_list):
    """Stack a list of observation dicts into batched tensors."""
    return {
        "board": torch.stack([torch.tensor(o["board"], dtype=torch.float32) for o in obs_list]),
        "hand": torch.stack([torch.tensor(o["hand"], dtype=torch.float32) for o in obs_list]),
        "scores": torch.stack([torch.tensor(o["scores"], dtype=torch.float32) for o in obs_list]),
        "meta": torch.stack([torch.tensor(o["meta"], dtype=torch.float32) for o in obs_list]),
        "conflict": torch.stack([torch.tensor(o["conflict"], dtype=torch.float32) for o in obs_list]),
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
    actions = torch.tensor(rollout["actions"], dtype=torch.long)
    old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32)
    advantages = torch.tensor(rollout["advantages"], dtype=torch.float32)
    returns = torch.tensor(rollout["returns"], dtype=torch.float32)

    # Normalize advantages
    if advantages.std() > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_pg_loss = 0
    total_vf_loss = 0
    total_entropy = 0
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

            # Build combined mask for this minibatch
            mb_masks = np.array([rollout["masks"][i] for i in mb_idx])

            logits, values = model.forward(mb_obs)
            mask_tensor = torch.tensor(mb_masks, dtype=torch.bool)
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

            # Total loss
            loss = pg_loss + VALUE_COEF * vf_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy.item()
            num_updates += 1

    return {
        "pg_loss": total_pg_loss / max(num_updates, 1),
        "vf_loss": total_vf_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
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


if __name__ == "__main__":
    t_start = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    # Build model + optimizer
    model = PolicyValueNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Action space: {ACTION_SPACE_SIZE}")
    print(f"Time budget: {TIME_BUDGET}s")
    print()

    # Create environment
    env = TigphratesEnv(player_count=PLAYER_COUNT, agent_player=0, max_turns=MAX_EPISODE_STEPS)

    # Training loop
    t_start_training = time.time()
    total_training_time = 0.0
    total_steps = 0
    total_episodes = 0
    rollout_num = 0
    all_episode_rewards = []

    while total_training_time < TIME_BUDGET:
        t0 = time.time()

        # Collect rollout
        rollout = collect_rollout(env, model, ROLLOUT_STEPS)
        total_steps += rollout["steps"]
        total_episodes += len(rollout["episode_rewards"])
        all_episode_rewards.extend(rollout["episode_rewards"])

        # PPO update
        losses = ppo_update(model, optimizer, rollout)

        t1 = time.time()
        dt = t1 - t0
        total_training_time += dt
        rollout_num += 1

        # Logging
        remaining = max(0, TIME_BUDGET - total_training_time)
        pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
        recent_rewards = all_episode_rewards[-10:] if all_episode_rewards else [0]
        avg_reward = np.mean(recent_rewards)

        print(
            f"\rrollout {rollout_num:04d} ({pct_done:.1f}%) | "
            f"pg_loss: {losses['pg_loss']:.4f} | "
            f"vf_loss: {losses['vf_loss']:.4f} | "
            f"entropy: {losses['entropy']:.3f} | "
            f"avg_reward: {avg_reward:.4f} | "
            f"episodes: {total_episodes} | "
            f"steps: {total_steps} | "
            f"remaining: {remaining:.0f}s    ",
            end="", flush=True,
        )

    print()  # newline after \r log
    env.close()

    t_end_training = time.time()

    # --- Evaluation ---
    print(f"\nEvaluating over {EVAL_GAMES} games vs heuristic AI...")
    model.eval()
    policy_fn = make_policy_fn(model)
    eval_results = evaluate_vs_heuristic(policy_fn, num_games=EVAL_GAMES)

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
    )

    # --- Save model ---
    os.makedirs("models", exist_ok=True)
    model_path = "models/policy_final.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

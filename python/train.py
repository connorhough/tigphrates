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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tigphrates_env import (
    TigphratesEnv, ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS,
    compute_leader_shaping_bonus,
    compute_event_shaping_bonus,
    _is_leader_placement as is_leader_placement_action,
    _is_monument_build as is_monument_build_action,
)
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
from shaping_config_dump import dump_shaping_config

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

# Potential-based reward shaping (Ng, Harada, Russell 1999):
#   F(s, s') = coef * (gamma * Phi(s') - Phi(s))
# with Phi(s) = active-player min-color score. Policy-invariant by
# construction: the optimal policy is unchanged regardless of coef, but
# the dense intra-game signal accelerates learning of "raise your floor
# color" — the dimension that defines the final score.
#
# When > 0, the legacy SCORE_DELTA_COEF * Δmin_score term is replaced by
# this potential form (cleaner switch — they would otherwise double-count
# the same signal). When = 0, falls back to the legacy shape exactly.
POTENTIAL_GAMMA_SHAPING_COEF = float(os.environ.get("POTENTIAL_GAMMA_SHAPING_COEF", 1.0))


def compute_potential_shaping(
    min_score_prev: float,
    min_score_next: float,
    terminal: bool,
    gamma: float,
    coef: float,
) -> float:
    """Pure helper for potential-based shaping reward.

    F(s, s') = coef * (gamma * Phi(s') - Phi(s))

    At terminal steps, Phi(s') := 0 (absorbing-state convention from
    Ng et al. 1999). This is *necessary* for policy invariance: it makes
    the telescoping sum collapse to -Phi(s_0). A consequence is that
    terminating with a high min-score yields a negative shaping reward
    at the boundary; the raw terminal reward implicitly absorbs the
    same magnitude in expectation.
    """
    if coef == 0.0:
        return 0.0
    phi_next = 0.0 if terminal else float(min_score_next)
    phi_prev = float(min_score_prev)
    return coef * (gamma * phi_next - phi_prev)

# BC auxiliary (Phase 6.3)
BC_COEF = float(os.environ.get("BC_COEF", 0.1))
BC_QUERY_PROB = float(os.environ.get("BC_QUERY_PROB", 0.25))
# When BC_AUX_DISABLED=1, effective_bc_coef() returns 0 — used after a
# BC pretrain checkpoint is loaded via BC_INIT_CHECKPOINT, since the
# auxiliary CE loss during PPO becomes redundant (and can pull the
# policy off-distribution from its already-imitative starting point).
BC_AUX_DISABLED = os.environ.get("BC_AUX_DISABLED", "0") == "1"
# Optional path to a BC-pretrained checkpoint (state_dict). Loaded BEFORE
# PPO begins, replacing the random init and the legacy policy_bc.pt path.
BC_INIT_CHECKPOINT = os.environ.get("BC_INIT_CHECKPOINT", "").strip() or None


def effective_bc_coef() -> float:
    """BC auxiliary coefficient actually applied in the PPO loss.
    Forced to 0 when BC_AUX_DISABLED=1 — this is the post-BC-pretrain
    setup where the auxiliary loss is no longer needed."""
    return 0.0 if BC_AUX_DISABLED else BC_COEF


def bc_init_checkpoint_path() -> str | None:
    """Path to the BC pretrain checkpoint, or None if unset."""
    return BC_INIT_CHECKPOINT

# V-trace replay buffer (Phase 11.2). On-policy PPO discards each rollout
# after a single update; mixing in past rollouts with a truncated importance-
# sampling correction extracts more learning per Mac mini step. Replay rows
# get rho = min(c_bar, pi_new(a|s) / pi_old(a|s)) before PPO clipping; current
# rollout rows keep the standard PPO ratio. Value targets for replay rows are
# slightly stale (we do not recompute GAE under the current value head); at
# REPLAY_K=2 the bias is small and PPO's clip keeps gradients bounded.
REPLAY_K = int(os.environ.get("REPLAY_K", 2))
REPLAY_RHO_BAR = float(os.environ.get("REPLAY_RHO_BAR", 1.0))

# Architecture (env-var overridable so a sweep can scale capacity).
# Phase 12 upgrade: AlphaZero-style residual conv tower + spatial heads.
# - BOARD_CONV_CHANNELS: width of every conv layer in the trunk. 64 gives
#   enough capacity for kingdom-level reasoning without blowing up step time.
# - RES_BLOCKS: depth of the residual tower. Each block has receptive-field
#   reach +2 cells (two 3x3 convs); 6 blocks reaches the full 11x16 board.
# - HIDDEN_DIM / NUM_HIDDEN_LAYERS: shared trunk after pooling for the type
#   head, value head, and non-spatial param slots.
BOARD_CONV_CHANNELS = int(os.environ.get("BOARD_CONV_CHANNELS", 64))
RES_BLOCKS = int(os.environ.get("RES_BLOCKS", 6))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 256))
NUM_HIDDEN_LAYERS = int(os.environ.get("NUM_HIDDEN_LAYERS", 2))

# Optional LSTM core (off by default). Tigris & Euphrates has hidden info
# (opponent hand composition) that a feedforward net cannot model across
# turns; an LSTM after the conv tower can carry that belief state. Default
# OFF so the running BC pretrain's checkpoint loads cleanly into the
# unchanged feedforward path; flip USE_LSTM_CORE=1 to enable.
#
# When enabled, an nn.LSTM(trunk_input_dim -> LSTM_HIDDEN) sits between the
# pooled conv features and the trunk MLP. Spatial heads (placeTile,
# placeLeader, placeCatastrophe) still come straight off the conv map -
# the LSTM only feeds the type_head, value_head, and nonspatial_head. The
# forward signature also gains an optional hidden_state kwarg and returns a
# 4-tuple including the new hidden state; the no-LSTM path keeps the
# existing 3-tuple signature unchanged.
USE_LSTM_CORE = os.environ.get("USE_LSTM_CORE", "0") == "1"
LSTM_HIDDEN = int(os.environ.get("LSTM_HIDDEN", 256))

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
CURRICULUM_ENABLED = os.environ.get("CURRICULUM_ENABLED", "1") == "1"
# Less aggressive than the original 1.0 → 0.1. Starting at 0.9 keeps a small
# share of self-play / pool opponents in the first rollouts (so the policy
# does not overfit to heuristic quirks); ending at 0.3 keeps a meaningful
# heuristic anchor late so the pool's noise can't fully dominate. These are
# the recommended defaults after the leader-placement bottleneck fix.
CURRICULUM_HEURISTIC_START = float(os.environ.get("CURRICULUM_HEURISTIC_START", 0.9))
CURRICULUM_HEURISTIC_END = float(os.environ.get("CURRICULUM_HEURISTIC_END", 0.3))

# League scheduler granularity. "per_episode" picks a fresh opponent for an
# env when its episode resets, so a single rollout exposes the policy to
# many different opponents. "per_rollout" uses one opponent for all envs
# until the next rollout — simpler but less diverse.
LEAGUE_SCHEDULER = "per_episode"

# Parallel rollouts
NUM_ENVS = 4              # parallel Tigphrates envs (one Node subprocess each)

# ---------------------------------------------------------------------------
# Hierarchical action layout (must match src/bridge/encoder.ts)
# ---------------------------------------------------------------------------
# Two-stage policy: type_head picks one of NUM_ACTION_TYPES action types,
# then param_head conditional on type picks within that type's slot range.
# Per-state mask sums collapse from up to 1728 down to ~10 types and
# typically <50 params per type, so the entropy regularizer no longer
# pushes mass into invalid regions.

NUM_ACTION_TYPES = 10
TYPE_PARAM_SIZES = [
    4 * BOARD_ROWS * BOARD_COLS,  # placeTile
    4 * BOARD_ROWS * BOARD_COLS,  # placeLeader
    4,                             # withdrawLeader
    BOARD_ROWS * BOARD_COLS,       # placeCatastrophe
    64,                            # swapTiles
    1,                             # pass
    64,                            # commitSupport
    4,                             # chooseWarOrder
    6,                             # buildMonument
    1,                             # declineMonument
]
TYPE_BASES = []
_acc = 0
for _s in TYPE_PARAM_SIZES:
    TYPE_BASES.append(_acc)
    _acc += _s
assert _acc == ACTION_SPACE_SIZE, (
    f"hierarchical layout total {_acc} != ACTION_SPACE_SIZE {ACTION_SPACE_SIZE}"
)
MAX_PARAMS = max(TYPE_PARAM_SIZES)

# Precomputed gather/decoding tensors. Built on CPU at import; moved per-call
# onto the same device as the logits so this works under MPS / CUDA / CPU.
_GATHER_IDX_NP = np.zeros((NUM_ACTION_TYPES, MAX_PARAMS), dtype=np.int64)
_PAD_MASK_NP = np.zeros((NUM_ACTION_TYPES, MAX_PARAMS), dtype=bool)
_FLAT_TO_TYPE_NP = np.zeros(ACTION_SPACE_SIZE, dtype=np.int64)
_FLAT_TO_PARAM_NP = np.zeros(ACTION_SPACE_SIZE, dtype=np.int64)
for _t in range(NUM_ACTION_TYPES):
    _base = TYPE_BASES[_t]
    _size = TYPE_PARAM_SIZES[_t]
    for _p in range(_size):
        _GATHER_IDX_NP[_t, _p] = _base + _p
        _PAD_MASK_NP[_t, _p] = True
        _FLAT_TO_TYPE_NP[_base + _p] = _t
        _FLAT_TO_PARAM_NP[_base + _p] = _p

GATHER_IDX = torch.tensor(_GATHER_IDX_NP, dtype=torch.long, device=DEVICE)
PAD_MASK = torch.tensor(_PAD_MASK_NP, dtype=torch.bool, device=DEVICE)
TYPE_BASES_T = torch.tensor(TYPE_BASES, dtype=torch.long, device=DEVICE)
FLAT_TO_TYPE = torch.tensor(_FLAT_TO_TYPE_NP, dtype=torch.long, device=DEVICE)
FLAT_TO_PARAM = torch.tensor(_FLAT_TO_PARAM_NP, dtype=torch.long, device=DEVICE)


# ---------------------------------------------------------------------------
# Symmetry augmentation (Phase 11.4)
# ---------------------------------------------------------------------------
# T&E's canonical river layout is NOT exactly column-symmetric, so the
# mirror is a virtual "mirror-world" T&E board with a different river
# pattern. The policy sees the river as one of its input channels, so
# training on the mirrored sample teaches it to play on either layout —
# useful regularization, not a true game-equivariance. log_prob, value,
# reward, advantage, and return are reused as-if on-policy; PPO clipping
# bounds the policy-mismatch bias.
#
# Implementation: build a fixed permutation MIRROR_PERM of the flat action
# space, where slot i maps to the reflected-cell action of the same type
# (non-spatial types are identity). Mask augmentation is `mask[MIRROR_PERM]`,
# board augmentation is `board[:, :, ::-1]`.
def _compute_mirror_index(idx: int) -> int:
    type_idx = int(_FLAT_TO_TYPE_NP[idx])
    param_idx = int(_FLAT_TO_PARAM_NP[idx])
    base = TYPE_BASES[type_idx]
    if type_idx in (0, 1):
        # placeTile / placeLeader: param = ci * (R*C) + r * C + c
        ci, rest = divmod(param_idx, BOARD_ROWS * BOARD_COLS)
        r, c = divmod(rest, BOARD_COLS)
        c_mirror = BOARD_COLS - 1 - c
        return base + ci * BOARD_ROWS * BOARD_COLS + r * BOARD_COLS + c_mirror
    if type_idx == 3:
        # placeCatastrophe: param = r * C + c
        r, c = divmod(param_idx, BOARD_COLS)
        return base + r * BOARD_COLS + (BOARD_COLS - 1 - c)
    # withdrawLeader, swapTiles, pass, commitSupport, chooseWarOrder,
    # buildMonument, declineMonument: not spatial, identity.
    return idx


_MIRROR_PERM_NP = np.array(
    [_compute_mirror_index(i) for i in range(ACTION_SPACE_SIZE)], dtype=np.int64
)
# Mirror is involutive — verify once at import to catch any indexing bugs.
assert (_MIRROR_PERM_NP[_MIRROR_PERM_NP] == np.arange(ACTION_SPACE_SIZE)).all(), (
    "MIRROR_PERM is not involutive; spatial action mirror has an off-by-one"
)

SYMMETRY_AUG = os.environ.get("SYMMETRY_AUG", "1") == "1"


def _mirror_obs_action(obs: dict, action: int, mask: np.ndarray) -> tuple[dict, int, np.ndarray]:
    """Reflect a transition along the column axis. Returns (mirrored_obs,
    mirrored_action_idx, mirrored_mask). Non-spatial obs fields and action
    types pass through unchanged."""
    new_obs = dict(obs)
    new_obs["board"] = obs["board"][:, :, ::-1].copy()

    # leaders / opp_leaders: 4 (leader, position) pairs as flat 8-vector
    # [r0,c0,r1,c1,r2,c2,r3,c3]; col fields at odd indices. -1 indicates
    # "off-board" (leader not placed) — preserve.
    leaders = obs["leaders"].copy()
    cols = leaders[1::2]
    leaders[1::2] = np.where(cols >= 0, BOARD_COLS - 1 - cols, cols)
    new_obs["leaders"] = leaders

    opp_leaders = obs["opp_leaders"].copy()
    cols = opp_leaders[1::2]
    opp_leaders[1::2] = np.where(cols >= 0, BOARD_COLS - 1 - cols, cols)
    new_obs["opp_leaders"] = opp_leaders

    new_action = int(_MIRROR_PERM_NP[action])
    new_mask = mask[_MIRROR_PERM_NP]
    return new_obs, new_action, new_mask


def _adapt_state_dict(state_dict: dict) -> dict:
    """Translate a pre-Phase-11.1 state dict (flat `policy_head`) into the
    hierarchical layout. The old policy_head's row weights map directly to
    the new param_head (same output dim = ACTION_SPACE_SIZE). type_head
    keeps its random init so the policy starts confused over types but
    inherits the parameter-level structure."""
    if "policy_head.weight" in state_dict and "param_head.weight" not in state_dict:
        adapted = dict(state_dict)
        adapted["param_head.weight"] = adapted.pop("policy_head.weight")
        if "policy_head.bias" in adapted:
            adapted["param_head.bias"] = adapted.pop("policy_head.bias")
        return adapted
    return state_dict


# ---------------------------------------------------------------------------
# Policy Network
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Standard AlphaZero residual block: Conv -> BN -> ReLU -> Conv -> BN
    -> add skip -> ReLU. The convs preserve spatial resolution (stride 1,
    padding 1, 3x3 kernel) so we can stack them without downsampling the
    11x16 board."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)


class BoardEncoder(nn.Module):
    """Residual conv tower for the 15-channel 11x16 board tensor.

    forward_features(x) -> (B, C, 11, 16) is the spatial map consumed by the
    spatial heads (placeTile, placeLeader, placeCatastrophe).
    forward(x) keeps the older flat-vector signature for compatibility.

    Tower depth controlled by RES_BLOCKS (env var, default 6). With a 3x3
    kernel and 6 blocks (12 conv layers + 1 stem), the receptive field is
    25x25 — comfortably bigger than the 11x16 board, so kingdoms and river
    paths can influence per-cell logits anywhere on the map.
    """

    def __init__(
        self,
        in_channels: int = BOARD_CHANNELS,
        out_channels: int = BOARD_CONV_CHANNELS,
        num_blocks: int = RES_BLOCKS,
    ):
        super().__init__()
        # Stem: project input channels to the trunk width.
        self.stem_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(out_channels)
        # Residual tower. ModuleList so tests can introspect `len(res_blocks)`.
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(num_blocks)]
        )
        self.channels = out_channels
        self.out_dim = out_channels * BOARD_ROWS * BOARD_COLS

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        return feat.view(feat.size(0), -1)


class PolicyValueNetwork(nn.Module):
    """Actor-critic with a residual conv tower + spatial conv heads.

    Spatial action types (placeTile, placeLeader, placeCatastrophe) read
    per-cell logits directly from a 1x1 conv on the encoder's spatial map,
    which gives the policy strong inductive bias on this 11x16 board: a
    placeTile logit at cell (r,c) depends primarily on the encoder's local
    receptive field around (r,c), not a flat 1728-way linear that has to
    learn cell identity from scratch.

    Non-spatial action types (withdrawLeader, swapTiles, pass, commitSupport,
    chooseWarOrder, buildMonument, declineMonument) get logits from a small
    MLP on the pooled trunk; their slot count is small (144 total).

    The hierarchical (type, param) sampling interface in get_action_and_value
    is unchanged — only the structure of param_logits is.
    """

    def __init__(self):
        super().__init__()
        self.board_encoder = BoardEncoder()
        C = self.board_encoder.channels

        # Extras: hand(4) + hand_seq(30) + scores(4) + meta(8) + conflict(7)
        #       + leaders(8) + opp_scores(4) + opp_leaders(8) = 73
        extra_dim = 4 + 30 + 4 + 8 + 7 + 8 + 4 + 8
        # Pool the spatial map (global average pool) before concat with extras
        # for the type / value / nonspatial heads. Keeps the trunk dim O(C),
        # not O(C*H*W), so the heads stay small even with a wider tower.
        trunk_input_dim = C + extra_dim

        # Optional recurrent core. When USE_LSTM_CORE=0 (default), self.lstm
        # is None and the trunk MLP consumes the (pooled-feat + extras) vector
        # directly - identical to the pre-LSTM architecture, so the BC
        # pretrain checkpoint's state_dict loads cleanly. When enabled, the
        # LSTM consumes (B, T, trunk_input_dim) and emits (B, T, LSTM_HIDDEN)
        # which then feeds the trunk; spatial heads bypass the LSTM and read
        # the raw conv map.
        self.use_lstm = USE_LSTM_CORE
        self.lstm_hidden = LSTM_HIDDEN if self.use_lstm else 0
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=trunk_input_dim,
                hidden_size=LSTM_HIDDEN,
                num_layers=1,
                batch_first=True,
            )
            trunk_in_dim = LSTM_HIDDEN
        else:
            self.lstm = None
            trunk_in_dim = trunk_input_dim

        layers = []
        in_dim = trunk_in_dim
        for _ in range(NUM_HIDDEN_LAYERS):
            layers.append(nn.Linear(in_dim, HIDDEN_DIM))
            layers.append(nn.ReLU())
            in_dim = HIDDEN_DIM
        self.trunk = nn.Sequential(*layers)

        # Type-level head.
        self.type_head = nn.Linear(HIDDEN_DIM, NUM_ACTION_TYPES)

        # Spatial heads. 1x1 convs straight from the encoder feature map.
        self.place_tile_head = nn.Conv2d(C, 4, kernel_size=1)
        self.place_leader_head = nn.Conv2d(C, 4, kernel_size=1)
        self.place_catastrophe_head = nn.Conv2d(C, 1, kernel_size=1)

        # Non-spatial param head: outputs concatenated logits for the seven
        # non-spatial action types in TYPE_BASES order:
        #   withdrawLeader (4) + swapTiles (64) + pass (1) + commitSupport (64)
        #   + chooseWarOrder (4) + buildMonument (6) + declineMonument (1) = 144
        nonspatial_total = (
            TYPE_PARAM_SIZES[2] + TYPE_PARAM_SIZES[4] + TYPE_PARAM_SIZES[5]
            + TYPE_PARAM_SIZES[6] + TYPE_PARAM_SIZES[7] + TYPE_PARAM_SIZES[8]
            + TYPE_PARAM_SIZES[9]
        )
        self.nonspatial_head = nn.Linear(HIDDEN_DIM, nonspatial_total)
        self._nonspatial_total = nonspatial_total

        # Value head.
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # Identity init: weight=1, bias=0 — keeps the residual
                # connection close to identity at the start of training.
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Small init on the policy heads keeps the initial action distribution
        # close to uniform after softmax.
        nn.init.orthogonal_(self.type_head.weight, gain=0.01)
        nn.init.orthogonal_(self.place_tile_head.weight, gain=0.01)
        nn.init.orthogonal_(self.place_leader_head.weight, gain=0.01)
        nn.init.orthogonal_(self.place_catastrophe_head.weight, gain=0.01)
        nn.init.orthogonal_(self.nonspatial_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs, hidden_state=None):
        """Forward pass.

        When self.use_lstm is False (default), returns the legacy 3-tuple
        (type_logits, param_logits, value) and the `hidden_state` kwarg is
        ignored - this keeps the no-LSTM code path byte-identical to the
        pre-LSTM architecture so the BC pretrain checkpoint loads cleanly.

        When self.use_lstm is True, the pooled-feat+extras vector is fed
        through a single-step LSTM (T=1) before reaching the trunk MLP,
        and the forward returns a 4-tuple (..., new_hidden_state) where
        new_hidden_state is the LSTM's (h, c) output. Spatial heads still
        read the raw conv map directly - only type/value/nonspatial heads
        consume the LSTM output. Pass hidden_state to carry belief state
        across consecutive single-step calls; pass None for a fresh start
        (zeros).
        """
        board = obs["board"]           # (B, BOARD_CHANNELS, 11, 16)
        hand = obs["hand"]             # (B, 4)
        hand_seq = obs["hand_seq"]     # (B, 6) ints in {-1, 0, 1, 2, 3}
        scores = obs["scores"]         # (B, 4)
        meta = obs["meta"]             # (B, 8)
        conflict = obs["conflict"]     # (B, 7)
        leaders = obs["leaders"]       # (B, 8)
        opp_scores = obs["opp_scores"] # (B, 4)
        opp_leaders = obs["opp_leaders"] # (B, 8)

        idx = hand_seq.to(torch.int64)
        idx = torch.where(idx < 0, torch.full_like(idx, 4), idx)
        hand_seq_oh = F.one_hot(idx, num_classes=5).to(torch.float32).reshape(idx.size(0), -1)

        # Spatial feature map: (B, C, H, W).
        feat = self.board_encoder.forward_features(board)
        B = feat.shape[0]

        # Pool to (B, C) for the trunk.
        pooled = feat.mean(dim=(2, 3))
        extra = torch.cat(
            [hand, hand_seq_oh, scores, meta, conflict, leaders, opp_scores, opp_leaders],
            dim=-1,
        )
        trunk_in = torch.cat([pooled, extra], dim=-1)

        new_hidden_state = None
        if self.use_lstm:
            # Single-step inference path: insert a length-1 time dim, run the
            # LSTM, then squeeze it back out. The hidden state is carried
            # forward by the caller across consecutive forward calls.
            lstm_in = trunk_in.unsqueeze(1)  # (B, 1, F)
            lstm_out, new_hidden_state = self.lstm(lstm_in, hidden_state)
            trunk_in = lstm_out.squeeze(1)   # (B, LSTM_HIDDEN)

        trunk_out = self.trunk(trunk_in)

        type_logits = self.type_head(trunk_out)
        value = self.value_head(trunk_out).squeeze(-1)

        # Spatial heads. Reshape (B, ci, r, c) -> (B, ci*r*c) walks ci, r, c
        # in row-major order, exactly matching `param = ci*(R*C) + r*C + c`.
        R, C_ = BOARD_ROWS, BOARD_COLS
        place_tile_spatial = self.place_tile_head(feat)        # (B, 4, R, C)
        place_leader_spatial = self.place_leader_head(feat)    # (B, 4, R, C)
        place_cata_spatial = self.place_catastrophe_head(feat) # (B, 1, R, C)

        place_tile_flat = place_tile_spatial.reshape(B, 4 * R * C_)
        place_leader_flat = place_leader_spatial.reshape(B, 4 * R * C_)
        place_cata_flat = place_cata_spatial.reshape(B, R * C_)

        # Non-spatial logits: split the concatenated nonspatial head output
        # into the per-type slices in TYPE_BASES order.
        nonspatial = self.nonspatial_head(trunk_out)  # (B, 144)
        ns_slices = {}
        acc = 0
        for t in (2, 4, 5, 6, 7, 8, 9):
            sz = TYPE_PARAM_SIZES[t]
            ns_slices[t] = nonspatial[:, acc:acc + sz]
            acc += sz

        # Concatenate in flat-action order. Single torch.cat — autograd-clean
        # and ONNX-traceable.
        param_logits = torch.cat(
            [
                place_tile_flat,    # type 0 placeTile (704)
                place_leader_flat,  # type 1 placeLeader (704)
                ns_slices[2],       # type 2 withdrawLeader (4)
                place_cata_flat,    # type 3 placeCatastrophe (176)
                ns_slices[4],       # type 4 swapTiles (64)
                ns_slices[5],       # type 5 pass (1)
                ns_slices[6],       # type 6 commitSupport (64)
                ns_slices[7],       # type 7 chooseWarOrder (4)
                ns_slices[8],       # type 8 buildMonument (6)
                ns_slices[9],       # type 9 declineMonument (1)
            ],
            dim=-1,
        )
        if self.use_lstm:
            return type_logits, param_logits, value, new_hidden_state
        return type_logits, param_logits, value

    def hierarchical_dists(self, type_logits, param_logits, flat_mask):
        """Build masked type distribution and the (B, NT, MP) padded
        param-logits tensor. flat_mask: (B, ACTION_SPACE_SIZE) bool/int.
        Returns (type_dist, param_logits_padded, type_mask).

        param_logits_padded is already masked: invalid (per-action-mask AND
        out-of-range pad) slots set to -inf, so callers can build a
        Categorical for any chosen type by indexing the row directly.

        type_mask is (B, NT) — true wherever ANY param within that type is
        legal in the current state. type_logits get -inf'd outside type_mask.
        """
        gather_idx = GATHER_IDX.to(type_logits.device)
        pad_mask = PAD_MASK.to(type_logits.device)

        flat_mask = torch.as_tensor(flat_mask, dtype=torch.bool, device=type_logits.device)
        if flat_mask.dim() == 1:
            flat_mask = flat_mask.unsqueeze(0)

        # (B, NT, MP) view of param logits, plus the matching mask view.
        param_logits_padded = param_logits[:, gather_idx]
        flat_mask_padded = flat_mask[:, gather_idx]
        valid = flat_mask_padded & pad_mask.unsqueeze(0)

        param_logits_padded = param_logits_padded.masked_fill(~valid, float("-inf"))
        type_mask = valid.any(dim=-1)
        type_logits_masked = type_logits.masked_fill(~type_mask, float("-inf"))
        type_dist = Categorical(logits=type_logits_masked)
        return type_dist, param_logits_padded, type_mask

    def get_action_and_value(self, obs, action_mask, action=None):
        # The LSTM-enabled forward returns a 4-tuple; callers of
        # get_action_and_value have not been updated to thread hidden state
        # yet (deferred to a follow-up that touches policy_server / evaluate
        # / mcts). For now we drop the new_hidden_state silently so a stray
        # USE_LSTM_CORE=1 doesn't crash the PPO update path - it just runs
        # the LSTM with a zero hidden state every call (effectively
        # feedforward). When the propagation work lands, this will switch
        # to plumbing hidden_state through.
        out = self.forward(obs)
        type_logits, param_logits, value = out[0], out[1], out[2]
        type_dist, param_logits_padded, _type_mask = self.hierarchical_dists(
            type_logits, param_logits, action_mask
        )
        B = type_logits.shape[0]
        device = type_logits.device

        if action is None:
            type_action = type_dist.sample()  # (B,)
            chosen_param_logits = param_logits_padded[torch.arange(B, device=device), type_action]
            param_dist = Categorical(logits=chosen_param_logits)
            param_action = param_dist.sample()
            base = TYPE_BASES_T.to(device)[type_action]
            action_t = base + param_action
        else:
            action_t = action if isinstance(action, torch.Tensor) else torch.tensor(
                action, dtype=torch.long, device=device
            )
            if action_t.dim() == 0:
                action_t = action_t.unsqueeze(0)
            flat_to_type = FLAT_TO_TYPE.to(device)
            flat_to_param = FLAT_TO_PARAM.to(device)
            type_action = flat_to_type[action_t]
            param_action = flat_to_param[action_t]
            chosen_param_logits = param_logits_padded[torch.arange(B, device=device), type_action]
            param_dist = Categorical(logits=chosen_param_logits)

        # Joint log-prob via chain rule. Joint entropy approximated as
        # H(type) + H(param | sampled-type) — exact for the sampled action,
        # a Monte-Carlo estimate of E_t[H(param|t)] for the regularizer.
        type_log_prob = type_dist.log_prob(type_action)
        param_log_prob = param_dist.log_prob(param_action)
        log_prob = type_log_prob + param_log_prob
        entropy = type_dist.entropy() + param_dist.entropy()

        return action_t, log_prob, entropy, value

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
    global_step: int = 0,
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
    # Per-env leader-placement counters for shaping bonus cap. Reset on episode
    # boundary so the cap applies per-game.
    leader_placements: list[int] = [0] * n
    # Per-env monument-build counters for the monument bonus cap. Same reset
    # semantics as leader_placements.
    monument_builds: list[int] = [0] * n

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
            sampled, log_probs_t, _entropy_t, values_t = model.get_action_and_value(
                batched_obs, batched_masks
            )

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

            # Potential-based shaping (Ng et al. 1999) replaces the legacy
            # SCORE_DELTA_COEF * Δmin_score term when its coef is > 0. The
            # two are alternative encodings of "raise your floor color";
            # using both would double-count. Margin term is left intact.
            if POTENTIAL_GAMMA_SHAPING_COEF > 0.0:
                potential_shape = compute_potential_shaping(
                    min_score_prev=prev_min,
                    min_score_next=new_min,
                    terminal=bool(done),
                    gamma=GAMMA,
                    coef=POTENTIAL_GAMMA_SHAPING_COEF,
                )
                shaped = reward + potential_shape + MARGIN_DELTA_COEF * margin_delta
            else:
                shaped = reward + SCORE_DELTA_COEF * blended + MARGIN_DELTA_COEF * margin_delta

            # Per-event shaping (leader/king/kingdom + treasure + monument).
            # Single combined helper; bumps leader/monument counters only when
            # the corresponding event actually fired so the per-game caps stay
            # honest. Decays to zero over SHAPING_DECAY_STEPS.
            shaping = compute_event_shaping_bonus(
                action_index=actions_list[i],
                prev_obs=obs_per_env[i],
                next_obs=obs_new,
                global_step=global_step + steps_collected,
                leader_placements_so_far=leader_placements[i],
                monument_builds_so_far=monument_builds[i],
            )
            if shaping > 0.0:
                shaped += shaping
                # Only count caps if this action was actually the gated event.
                if (is_leader_placement_action(actions_list[i])
                        and leader_placements[i] < 4):
                    leader_placements[i] += 1
                if (is_monument_build_action(actions_list[i])
                        and monument_builds[i] < 2):
                    monument_builds[i] += 1

            per_env_rewards[i].append(shaped)
            per_env_dones[i].append(float(done))
            current_ep_rewards[i] += shaped
            steps_collected += 1

            if done:
                episode_rewards.append(current_ep_rewards[i])
                current_ep_rewards[i] = 0.0
                leader_placements[i] = 0  # reset per-game cap
                monument_builds[i] = 0    # reset per-game cap
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

    # Symmetry augmentation: append a mirrored copy of every transition.
    # Doubles the PPO batch size for free at the engine layer (no extra
    # rollouts), at the cost of ~2× compute per update step. The mirrored
    # transition reuses the original log_prob/value/return/advantage as-if
    # on-policy; PPO clipping bounds the policy-mismatch bias.
    if SYMMETRY_AUG and all_obs:
        n_orig = len(all_obs)
        for i in range(n_orig):
            m_obs, m_action, m_mask = _mirror_obs_action(
                all_obs[i], int(all_actions[i]), all_masks[i]
            )
            all_obs.append(m_obs)
            all_actions.append(m_action)
            all_log_probs.append(all_log_probs[i])
            all_values.append(all_values[i])
            all_masks.append(m_mask)
            all_adv.append(all_adv[i])
            all_ret.append(all_ret[i])
            # Also mirror the BC expert target so the auxiliary loss stays
            # supervised on the mirrored frame.
            exp = all_expert[i]
            all_expert.append(int(_MIRROR_PERM_NP[exp]) if exp >= 0 else -1)

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

        # Potential-based shaping (Ng et al. 1999) replaces the legacy
        # SCORE_DELTA_COEF * Δmin_score term when its coef is > 0. The
        # two are alternative encodings of "raise your floor color";
        # using both would double-count.
        if POTENTIAL_GAMMA_SHAPING_COEF > 0.0:
            reward = reward + compute_potential_shaping(
                min_score_prev=prev_min_score,
                min_score_next=new_min_score,
                terminal=bool(done),
                gamma=GAMMA,
                coef=POTENTIAL_GAMMA_SHAPING_COEF,
            )
        else:
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


def combine_with_replay(current: dict, replay_buf) -> dict:
    """Concatenate the current rollout with every rollout in the replay
    buffer, tagging each row with `is_replay` (False for current, True for
    replay). The combined dict feeds straight into ppo_update."""
    if not replay_buf:
        out = dict(current)
        out["is_replay"] = np.zeros(len(current["obs"]), dtype=bool)
        return out

    obs = list(current["obs"])
    actions = list(current["actions"])
    log_probs = list(current["log_probs"])
    advantages = list(current["advantages"])
    returns = list(current["returns"])
    masks = list(current["masks"])
    values = list(current["values"])
    expert = list(current["expert_actions"])
    is_replay = [False] * len(current["obs"])

    for r in replay_buf:
        if not r["obs"]:
            continue
        obs.extend(r["obs"])
        actions.extend(r["actions"])
        log_probs.extend(r["log_probs"])
        advantages.extend(r["advantages"])
        returns.extend(r["returns"])
        masks.extend(r["masks"])
        values.extend(r["values"])
        expert.extend(r["expert_actions"])
        is_replay.extend([True] * len(r["obs"]))

    return {
        "obs": obs,
        "actions": np.array(actions, dtype=np.int64),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "advantages": np.array(advantages, dtype=np.float32),
        "returns": np.array(returns, dtype=np.float32),
        "masks": masks,
        "values": np.array(values, dtype=np.float32),
        "expert_actions": np.array(expert, dtype=np.int64),
        "is_replay": np.array(is_replay, dtype=bool),
        "episode_rewards": current.get("episode_rewards", []),
        "steps": current.get("steps", 0),
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
    is_replay = torch.tensor(
        rollout.get("is_replay", np.zeros(n, dtype=bool)),
        dtype=torch.bool, device=DEVICE,
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
            mb_replay = is_replay[mb_idx]

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

            type_logits, param_logits, values = model.forward(mb_obs)
            type_dist, param_logits_padded, _type_mask = model.hierarchical_dists(
                type_logits, param_logits, mb_masks
            )
            B = type_logits.shape[0]
            device = type_logits.device

            # Decompose taken action into (type, param) so we can recompute
            # log-prob under the current (post-update) hierarchical policy.
            flat_to_type = FLAT_TO_TYPE.to(device)
            flat_to_param = FLAT_TO_PARAM.to(device)
            type_action = flat_to_type[mb_actions]
            param_action = flat_to_param[mb_actions]
            chosen_param_logits = param_logits_padded[torch.arange(B, device=device), type_action]
            param_dist = Categorical(logits=chosen_param_logits)

            new_log_probs = type_dist.log_prob(type_action) + param_dist.log_prob(param_action)
            entropy = (type_dist.entropy() + param_dist.entropy()).mean()

            # PPO clipped objective with V-trace truncated IS for replay rows.
            # Current-rollout rows: standard ratio. Replay rows: ratio capped
            # at REPLAY_RHO_BAR so stale data never up-weights the gradient.
            # PPO's [1-eps, 1+eps] clip then runs over the (possibly capped)
            # ratio for both populations, so off-policy data can't drift the
            # update past the standard PPO trust region.
            ratio = (new_log_probs - mb_old_log_probs).exp()
            effective_ratio = torch.where(
                mb_replay,
                torch.clamp(ratio, max=REPLAY_RHO_BAR),
                ratio,
            )
            surr1 = effective_ratio * mb_advantages
            surr2 = torch.clamp(effective_ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
            pg_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            vf_loss = F.mse_loss(values, mb_returns)

            # Auxiliary BC loss: hierarchical cross-entropy against the
            # heuristic's chosen action on the subset of rows where it was
            # queried (expert >= 0). Keeps the policy from drifting too far
            # from heuristic-quality play during exploration. Decomposed
            # into type CE + parameter CE so each head learns directly.
            bc_idx = torch.where(mb_expert >= 0)[0]
            if bc_idx.numel() > 0:
                expert_flat = mb_expert[bc_idx]
                expert_type = flat_to_type[expert_flat]
                expert_param = flat_to_param[expert_flat]

                # type_dist.logits is already log-softmax (normalized) so
                # nll_loss is the correct masked CE; F.cross_entropy would
                # double-normalize but log_softmax is idempotent so it's fine
                # too — using nll_loss for clarity.
                bc_type_log_probs = type_dist.logits[bc_idx]
                type_ce = F.nll_loss(bc_type_log_probs, expert_type)

                bc_param_padded = param_logits_padded[bc_idx]  # (n_bc, NT, MP)
                bc_chosen_logits = bc_param_padded[
                    torch.arange(bc_idx.numel(), device=device), expert_type
                ]  # (n_bc, MP) — raw masked logits, run log_softmax now.
                bc_param_log_probs = F.log_softmax(bc_chosen_logits, dim=-1)
                param_ce = F.nll_loss(bc_param_log_probs, expert_param)

                bc_loss = type_ce + param_ce
            else:
                bc_loss = torch.tensor(0.0, device=device)

            # Total loss
            loss = (
                pg_loss
                + VALUE_COEF * vf_loss
                - ENTROPY_COEF * entropy
                + effective_bc_coef() * bc_loss
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
    raw = torch.load(path, map_location="cpu")
    m.load_state_dict(_adapt_state_dict(raw), strict=False)
    m.train(False)
    m.to(DEVICE)
    return m


def build_policy(load_path: str | None) -> nn.Module:
    """Construct a PolicyValueNetwork, optionally loading weights from a
    BC pretrain checkpoint.

    - load_path is None: random init (existing behavior).
    - load_path exists: load state_dict (with `_adapt_state_dict` for
      pre-Phase-11.1 compatibility).
    - load_path doesn't exist: print a warning and fall back to random
      init so a missing checkpoint doesn't silently break a training run.

    Used by main() at startup. Factored out so unit tests can verify the
    init path without booting the full training loop.
    """
    m = PolicyValueNetwork()
    if load_path is None:
        return m
    p = pathlib.Path(load_path)
    if not p.exists():
        print(f"build_policy: checkpoint not found at {p}; using random init",
              file=sys.stderr)
        return m
    raw = torch.load(p, map_location="cpu")
    if isinstance(raw, dict) and "model_state" in raw:
        raw = raw["model_state"]
    m.load_state_dict(_adapt_state_dict(raw), strict=False)
    print(f"build_policy: loaded BC init from {p}")
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
                model.load_state_dict(_adapt_state_dict(ckpt["model_state"]), strict=False)
                # Optimizer state can only be reused if the parameter shapes
                # match — adapted (renamed) state dicts do, but if the source
                # was pre-hierarchical the new type_head has no optimizer
                # entries. load_state_dict will raise; catch and skip.
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except (ValueError, KeyError) as e:
                    print(f"  optimizer state incompatible ({e}); using fresh optimizer")
                if "torch_rng" in ckpt:
                    torch.set_rng_state(ckpt["torch_rng"])
                if "numpy_rng" in ckpt:
                    np.random.set_state(ckpt["numpy_rng"])
                resumed = True
                print(f"Resumed from {resume_path}")
            else:
                # Bare state_dict: load weights, fresh optimizer.
                model.load_state_dict(_adapt_state_dict(ckpt), strict=False)
                resumed = True
                print(f"Loaded weights from {resume_path} (fresh optimizer)")
        else:
            print(f"RESUME_FROM={resume_path} not found; starting fresh")

    if not resumed:
        # Priority: BC_INIT_CHECKPOINT (new BC pretrain stage) →
        # legacy <RUN_DIR>/policy_bc.pt → random init.
        bc_init = bc_init_checkpoint_path()
        if bc_init:
            init_path = pathlib.Path(bc_init)
            if init_path.exists():
                try:
                    raw = torch.load(init_path, map_location="cpu")
                    if isinstance(raw, dict) and "model_state" in raw:
                        raw = raw["model_state"]
                    model.load_state_dict(_adapt_state_dict(raw), strict=False)
                    print(f"Loaded BC pretrain from {init_path} (BC_INIT_CHECKPOINT)")
                except Exception as e:
                    print(f"Failed to load BC_INIT_CHECKPOINT ({e}); starting from random init")
            else:
                print(f"BC_INIT_CHECKPOINT={init_path} not found; falling back to legacy path")
        else:
            bc_path = RUN_DIR / "policy_bc.pt"
            if bc_path.exists():
                try:
                    raw = torch.load(bc_path, map_location="cpu")
                    model.load_state_dict(_adapt_state_dict(raw), strict=False)
                    print(f"Loaded BC warm start from {bc_path}")
                except Exception as e:
                    print(f"Failed to load BC warm start ({e}); starting from random init")
        if BC_AUX_DISABLED:
            print(f"BC_AUX_DISABLED=1 → effective BC_COEF forced to 0 in PPO loss")
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

    # Seed pool with the initial policy so early rollouts have a non-heuristic
    # opponent to learn against. Subsequent snapshots get added whenever a
    # rollout finishes (capped at POOL_MAX_SIZE).
    #
    # FRESH_INIT_CHECKPOINT env var, if set to an existing path, is preferred
    # over the in-memory model. Lets the pool reset workflow (`python
    # reset_pool.py --init-checkpoint <path>`) seed a known-good random init
    # without having to re-run BC pretraining first.
    if not list_pool():
        fresh_init = os.environ.get("FRESH_INIT_CHECKPOINT")
        if fresh_init and pathlib.Path(fresh_init).exists():
            _ensure_pool_dir()
            init_path = POOL_DIR / "policy_init.pt"
            try:
                import shutil as _shutil
                _shutil.copyfile(fresh_init, init_path)
                print(f"Seeded pool init from {fresh_init} -> {init_path}")
            except OSError as e:
                print(f"Failed to copy FRESH_INIT_CHECKPOINT ({e}); seeding from in-memory model")
                add_to_pool(model, label="init")
        else:
            add_to_pool(model, label="init")

    # Training loop
    total_training_time = 0.0
    total_steps = 0
    total_episodes = 0
    rollout_num = 0
    all_episode_rewards = []
    opponent_label = "heuristic"
    # V-trace replay buffer. Holds the K most recent rollouts; each PPO
    # update mixes the current rollout with the buffer so each transition is
    # reused ~K times before being dropped. K=0 disables replay (pure on-
    # policy PPO behavior). Stored rollouts use the original behavior-policy
    # log_probs as the IS denominator.
    replay_buffer = deque(maxlen=REPLAY_K)

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

        rollout = collect_rollout_vec(
            env_vec, model, ROLLOUT_STEPS,
            on_reset=on_reset_cb,
            global_step=total_steps,
        )
        total_steps += rollout["steps"]
        total_episodes += len(rollout["episode_rewards"])
        all_episode_rewards.extend(rollout["episode_rewards"])

        combined = combine_with_replay(rollout, replay_buffer) if REPLAY_K > 0 else rollout
        losses = ppo_update(model, optimizer, combined)
        if REPLAY_K > 0 and rollout["obs"]:
            replay_buffer.append(rollout)

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
    dump_shaping_config(RUN_DIR / "shaping_config.json")

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

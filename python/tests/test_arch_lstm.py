"""LSTM-core architecture tests.

The LSTM core is opt-in via env var USE_LSTM_CORE=1. The default code path
must be byte-identical to the pre-LSTM architecture (same state_dict keys
and shapes, same forward signature) so the long-running BC pretrain's
checkpoint can be loaded cleanly.

Run: ./python/venv/bin/python -m pytest python/tests/test_arch_lstm.py -xvs
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest
import torch


# ---------------------------------------------------------------------------
# Baseline state_dict contract: pre-edit key set with shapes. Snapshot taken
# at the start of this work; updating this dict is the explicit way to opt
# into a state-dict-incompatible change to the no-LSTM path. If you find
# yourself updating it casually, you are about to break the running BC
# pretrain's checkpoint load.
# ---------------------------------------------------------------------------
BASELINE_STATE_DICT_SHAPES: dict[str, tuple[int, ...]] = {
    "board_encoder.stem_conv.weight": (64, 15, 3, 3),
    "board_encoder.stem_bn.weight": (64,),
    "board_encoder.stem_bn.bias": (64,),
    "board_encoder.stem_bn.running_mean": (64,),
    "board_encoder.stem_bn.running_var": (64,),
    "board_encoder.stem_bn.num_batches_tracked": (),
    **{
        f"board_encoder.res_blocks.{i}.{name}": shape
        for i in range(6)
        for name, shape in [
            ("conv1.weight", (64, 64, 3, 3)),
            ("bn1.weight", (64,)),
            ("bn1.bias", (64,)),
            ("bn1.running_mean", (64,)),
            ("bn1.running_var", (64,)),
            ("bn1.num_batches_tracked", ()),
            ("conv2.weight", (64, 64, 3, 3)),
            ("bn2.weight", (64,)),
            ("bn2.bias", (64,)),
            ("bn2.running_mean", (64,)),
            ("bn2.running_var", (64,)),
            ("bn2.num_batches_tracked", ()),
        ]
    },
    "trunk.0.weight": (256, 137),
    "trunk.0.bias": (256,),
    "trunk.2.weight": (256, 256),
    "trunk.2.bias": (256,),
    "type_head.weight": (10, 256),
    "type_head.bias": (10,),
    "place_tile_head.weight": (4, 64, 1, 1),
    "place_tile_head.bias": (4,),
    "place_leader_head.weight": (4, 64, 1, 1),
    "place_leader_head.bias": (4,),
    "place_catastrophe_head.weight": (1, 64, 1, 1),
    "place_catastrophe_head.bias": (1,),
    "nonspatial_head.weight": (144, 256),
    "nonspatial_head.bias": (144,),
    "value_head.weight": (1, 256),
    "value_head.bias": (1,),
}


def _reload_train(monkeypatch, *, lstm: bool, hidden: int | None = None):
    """Force-reload the train module after setting env vars.

    Architecture toggles are read from env at module top, so module-level
    constants like USE_LSTM_CORE / LSTM_HIDDEN must be re-imported. We pop
    and re-import so the test's monkeypatched env wins.
    """
    if lstm:
        monkeypatch.setenv("USE_LSTM_CORE", "1")
    else:
        monkeypatch.delenv("USE_LSTM_CORE", raising=False)
    if hidden is not None:
        monkeypatch.setenv("LSTM_HIDDEN", str(hidden))
    sys.modules.pop("train", None)
    return importlib.import_module("train")


def _make_obs_batch(B: int):
    from tigphrates_env import BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS

    return {
        "board": torch.zeros(B, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS, dtype=torch.float32),
        "hand": torch.zeros(B, 4, dtype=torch.float32),
        "hand_seq": torch.full((B, 6), -1, dtype=torch.int64),
        "scores": torch.zeros(B, 4, dtype=torch.float32),
        "meta": torch.zeros(B, 8, dtype=torch.float32),
        "conflict": torch.zeros(B, 7, dtype=torch.float32),
        "leaders": torch.full((B, 8), -1, dtype=torch.float32),
        "opp_scores": torch.zeros(B, 4, dtype=torch.float32),
        "opp_leaders": torch.full((B, 8), -1, dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# 1. No-LSTM regression: state_dict keys + shapes must exactly match the
#    pre-edit baseline so the running BC pretrain's checkpoint loads cleanly.
# ---------------------------------------------------------------------------
def test_no_lstm_state_dict_matches_baseline(monkeypatch):
    train = _reload_train(monkeypatch, lstm=False)
    model = train.PolicyValueNetwork()
    sd = model.state_dict()
    actual = {k: tuple(v.shape) for k, v in sd.items()}

    missing = set(BASELINE_STATE_DICT_SHAPES) - set(actual)
    extra = set(actual) - set(BASELINE_STATE_DICT_SHAPES)
    assert not missing, f"keys missing from no-LSTM net: {sorted(missing)}"
    assert not extra, f"unexpected keys on no-LSTM net: {sorted(extra)}"

    for k, expected in BASELINE_STATE_DICT_SHAPES.items():
        assert actual[k] == expected, (
            f"shape drift on {k}: expected {expected}, got {actual[k]}"
        )


# ---------------------------------------------------------------------------
# 2. Shape parity (no-LSTM). Forward returns 3-tuple identical to before.
# ---------------------------------------------------------------------------
def test_no_lstm_forward_shape_parity(monkeypatch):
    train = _reload_train(monkeypatch, lstm=False)
    from tigphrates_env import ACTION_SPACE_SIZE

    model = train.PolicyValueNetwork()
    model.eval()
    obs = _make_obs_batch(1)
    out = model.forward(obs)

    assert isinstance(out, tuple) and len(out) == 3, (
        f"no-LSTM forward must return 3-tuple, got len={len(out)}"
    )
    type_logits, param_logits, value = out
    assert type_logits.shape == (1, train.NUM_ACTION_TYPES)
    assert param_logits.shape == (1, ACTION_SPACE_SIZE)
    assert value.shape == (1,)


# ---------------------------------------------------------------------------
# 3. LSTM enabled: state_dict has lstm.* keys.
# ---------------------------------------------------------------------------
def test_lstm_state_dict_has_lstm_params(monkeypatch):
    train = _reload_train(monkeypatch, lstm=True)
    model = train.PolicyValueNetwork()
    sd = model.state_dict()
    lstm_keys = [k for k in sd if k.startswith("lstm.")]
    assert lstm_keys, f"expected lstm.* keys in LSTM-enabled state_dict, got {sorted(sd)}"


# ---------------------------------------------------------------------------
# 4. LSTM enabled: forward returns 4-tuple including new hidden state.
# ---------------------------------------------------------------------------
def test_lstm_forward_returns_hidden_state(monkeypatch):
    hidden = 256
    train = _reload_train(monkeypatch, lstm=True, hidden=hidden)
    model = train.PolicyValueNetwork()
    model.eval()
    obs = _make_obs_batch(1)
    out = model.forward(obs)

    assert isinstance(out, tuple) and len(out) == 4, (
        f"LSTM forward must return 4-tuple, got len={len(out)}"
    )
    _type_logits, _param_logits, _value, new_hidden = out
    assert isinstance(new_hidden, tuple) and len(new_hidden) == 2, (
        "new_hidden must be (h, c) for nn.LSTM"
    )
    h, c = new_hidden
    # nn.LSTM hidden state shape is (num_layers, B, hidden).
    assert h.shape == (1, 1, hidden), h.shape
    assert c.shape == (1, 1, hidden), c.shape


# ---------------------------------------------------------------------------
# 5. Hidden state propagates: same input + same h_0 -> same output.
# ---------------------------------------------------------------------------
def test_lstm_hidden_state_reproducible(monkeypatch):
    hidden = 256
    train = _reload_train(monkeypatch, lstm=True, hidden=hidden)
    model = train.PolicyValueNetwork()
    model.eval()
    obs = _make_obs_batch(1)

    type_a, param_a, value_a, _hid_a = model.forward(obs)
    # Re-run from the same fresh hidden state (None implicitly = zeros). Must
    # match exactly bit-for-bit (same module, same input, eval mode).
    type_b, param_b, value_b, _hid_b = model.forward(obs)
    torch.testing.assert_close(type_a, type_b)
    torch.testing.assert_close(param_a, param_b)
    torch.testing.assert_close(value_a, value_b)

    # Now drive a step from a NON-zero hidden state and confirm passing the
    # same h_0 gives identical output across two invocations.
    h0 = torch.randn(1, 1, hidden)
    c0 = torch.randn(1, 1, hidden)
    out1 = model.forward(obs, hidden_state=(h0, c0))
    out2 = model.forward(obs, hidden_state=(h0.clone(), c0.clone()))
    torch.testing.assert_close(out1[0], out2[0])
    torch.testing.assert_close(out1[1], out2[1])
    torch.testing.assert_close(out1[2], out2[2])


# ---------------------------------------------------------------------------
# 6. Hidden state actually feeds the prediction.
# ---------------------------------------------------------------------------
def test_lstm_different_hidden_state_changes_output(monkeypatch):
    hidden = 256
    train = _reload_train(monkeypatch, lstm=True, hidden=hidden)
    torch.manual_seed(0)
    model = train.PolicyValueNetwork()
    model.eval()
    obs = _make_obs_batch(1)

    h_a = torch.zeros(1, 1, hidden)
    c_a = torch.zeros(1, 1, hidden)
    h_b = torch.randn(1, 1, hidden)
    c_b = torch.randn(1, 1, hidden)

    out_a = model.forward(obs, hidden_state=(h_a, c_a))
    out_b = model.forward(obs, hidden_state=(h_b, c_b))
    # type_logits must differ - proves the hidden state is wired into the head.
    assert not torch.allclose(out_a[0], out_b[0]), (
        "type_logits identical under different hidden states; LSTM not wired in"
    )


# ---------------------------------------------------------------------------
# 7. Old (no-LSTM) checkpoint loads on a fresh no-LSTM net via
#    _adapt_state_dict + strict=False, the same path used to load the BC
#    pretrain checkpoint.
# ---------------------------------------------------------------------------
def test_old_checkpoint_loads_on_no_lstm_net(tmp_path, monkeypatch):
    train = _reload_train(monkeypatch, lstm=False)
    src = train.PolicyValueNetwork()
    ckpt_path = tmp_path / "policy_init.pt"
    torch.save(src.state_dict(), ckpt_path)

    dst = train.PolicyValueNetwork()
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    adapted = train._adapt_state_dict(raw)
    missing, unexpected = dst.load_state_dict(adapted, strict=False)
    assert not missing, f"missing keys when reloading fresh ckpt: {missing}"
    assert not unexpected, f"unexpected keys when reloading fresh ckpt: {unexpected}"

    # Sanity: dst now matches src element-wise on every parameter.
    for (k, v_src), (_k2, v_dst) in zip(
        src.state_dict().items(), dst.state_dict().items()
    ):
        torch.testing.assert_close(v_src, v_dst)

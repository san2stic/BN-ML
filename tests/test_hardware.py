from __future__ import annotations

import bn_ml.hardware as hw


def test_resolve_xgb_device_auto_no_gpu(monkeypatch) -> None:
    monkeypatch.setattr(hw, "has_nvidia_gpu", lambda: False)
    device, reason = hw.resolve_xgb_device("auto")

    assert device == "cpu"
    assert "without NVIDIA" in reason


def test_resolve_xgb_device_auto_with_gpu(monkeypatch) -> None:
    monkeypatch.setattr(hw, "has_nvidia_gpu", lambda: True)
    device, reason = hw.resolve_xgb_device("auto")

    assert device == "cuda"
    assert "NVIDIA" in reason


def test_resolve_xgb_device_forced_cuda_without_gpu(monkeypatch) -> None:
    monkeypatch.setattr(hw, "has_nvidia_gpu", lambda: False)
    device, reason = hw.resolve_xgb_device("cuda")

    assert device == "cpu"
    assert "fallback" in reason

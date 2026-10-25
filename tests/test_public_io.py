"""Public image and training-checkpoint contracts without downloaded weights."""

import numpy as np
from PIL import Image
import pytest
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from torch import nn

from safir.infer import load_pair, paired_images, save_luminance
from safir.train import restore_predecessor, save_stage


def image_dirs(tmp_path, names=("first", "second")):
    visible, infrared = tmp_path / "visible", tmp_path / "infrared"
    visible.mkdir()
    infrared.mkdir()
    for name in names:
        Image.new("RGB", (37, 35), (51, 102, 153)).save(visible / f"{name}.png")
        Image.new("L", (37, 35), 204).save(infrared / f"{name}.png")
    return visible, infrared


def test_manifest_preserves_population_and_order(tmp_path):
    visible, infrared = image_dirs(tmp_path)
    manifest = tmp_path / "split.txt"
    manifest.write_text("# selected order\nsecond.png\nfirst\n", encoding="utf-8")
    assert [name for name, _, _ in paired_images(visible, infrared, manifest)] == ["second", "first"]
    manifest.write_text("first\nFIRST.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate samples"):
        paired_images(visible, infrared, manifest)
    manifest.write_text("missing\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Missing paired"):
        paired_images(visible, infrared, manifest)


def test_ambiguous_and_unpaired_inputs_are_rejected(tmp_path):
    visible, infrared = image_dirs(tmp_path)
    (infrared / "second.png").unlink()
    with pytest.raises(ValueError, match="sample sets differ"):
        paired_images(visible, infrared)
    Image.new("RGB", (37, 35)).save(visible / "first.jpg")
    with pytest.raises(ValueError, match="Duplicate image stem"):
        paired_images(visible, infrared)


def test_loading_preserves_native_pixels_and_restores_original_extent(tmp_path):
    visible, infrared = image_dirs(tmp_path, ("sample",))
    vi, ir, size = load_pair(visible / "sample.png", infrared / "sample.png", "cpu")
    assert size == (35, 37)
    assert vi.shape == (1, 3, 48, 48)
    assert ir.shape == (1, 1, 48, 48)
    np.testing.assert_allclose(vi[0, :, :35, :37].numpy(),
                               np.broadcast_to(np.array([0.2, 0.4, 0.6])[:, None, None], (3, 35, 37)))
    torch.testing.assert_close(ir, torch.full_like(ir, 0.8))
    result = tmp_path / "result.png"
    save_luminance(torch.full_like(ir, 0.5), result, size)
    with Image.open(result) as image:
        assert image.size == (37, 35)
        assert image.mode == "L"
        assert np.all(np.array(image) == 127)


def test_unaligned_or_too_small_images_are_rejected(tmp_path):
    visible, infrared = image_dirs(tmp_path, ("sample",))
    vi_path, ir_path = visible / "sample.png", infrared / "sample.png"
    Image.new("L", (36, 35)).save(ir_path)
    with pytest.raises(ValueError, match="Unaligned image sizes"):
        load_pair(vi_path, ir_path, "cpu")
    Image.new("RGB", (31, 35)).save(vi_path)
    Image.new("L", (31, 35)).save(ir_path)
    with pytest.raises(ValueError, match="at least 32 pixels"):
        load_pair(vi_path, ir_path, "cpu")


def stage_model():
    model = nn.Module()
    model.add_module("fusion", nn.Linear(3, 2))
    model.register_buffer("step", torch.tensor(7, dtype=torch.int64))
    model.add_module("prior_extractor", nn.Module())
    for name in ("seg_extractor", "det_extractor"):
        extractor = nn.Module()
        extractor.add_module("backbone", nn.BatchNorm2d(3))
        model.prior_extractor.add_module(name, extractor)
    return model


def test_stage_round_trip_keeps_external_prior_buffers(tmp_path):
    source, restored = stage_model(), stage_model()
    for name in ("seg_extractor", "det_extractor"):
        getattr(source.prior_extractor, name).backbone.num_batches_tracked.fill_(9)
        getattr(restored.prior_extractor, name).backbone.num_batches_tracked.fill_(11)
    target = tmp_path / "stage.safetensors"
    save_stage(source, target, "a223")
    with safe_open(target, framework="pt") as checkpoint:
        assert checkpoint.metadata()["stage"] == "a223"
        assert set(checkpoint.keys()) == {"fusion.weight", "fusion.bias", "step"}
    restore_predecessor(restored, target, "a223")
    torch.testing.assert_close(restored.fusion.weight, source.fusion.weight)
    torch.testing.assert_close(restored.fusion.bias, source.fusion.bias)
    assert restored.step.item() == 7
    for name in ("seg_extractor", "det_extractor"):
        assert getattr(restored.prior_extractor, name).backbone.num_batches_tracked.item() == 11


def test_wrong_stage_or_tensor_metadata_fails_before_restore(tmp_path):
    model = stage_model()
    target = tmp_path / "stage.safetensors"
    save_stage(model, target, "a215")
    with pytest.raises(ValueError, match="Expected a a217 checkpoint"):
        restore_predecessor(model, target, "a217")
    state = load_file(target)
    for index, replacement in enumerate((torch.zeros(1), state["fusion.weight"].double())):
        damaged = {**state, "fusion.weight": replacement}
        damaged_path = tmp_path / f"damaged_{index}.safetensors"
        save_file(damaged, damaged_path, metadata={"stage": "a215"})
        with pytest.raises(ValueError, match="Predecessor tensor mismatch"):
            restore_predecessor(model, damaged_path, "a215")

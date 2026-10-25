import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("PyTorch is required for numerical smoke tests") from exc

from PIL import Image

from data.dataset import get_test_loader
from data.smoke import sample_morphology, synthesize_smoke


def test_smoke_is_exactly_reproducible_and_bounded():
    clean = torch.linspace(0, 1, 3 * 48 * 64).reshape(3, 48, 64)
    first = synthesize_smoke(clean, "medium", seed=1234)
    second = synthesize_smoke(clean, "medium", seed=1234)
    for a, b in zip(first[:3], second[:3]):
        assert torch.equal(a, b)
    smoke, mask, transmission, metadata = first
    assert smoke.shape == clean.shape
    assert mask.shape == transmission.shape == (1, 48, 64)
    assert torch.all((0 <= smoke) & (smoke <= 1))
    assert torch.allclose(mask + transmission, torch.ones_like(mask), atol=1e-7)
    assert metadata["seed"] == 1234
    assert metadata["severity"] == "medium"


def test_severity_has_monotonic_mean_transmission_over_fixed_seeds():
    clean = torch.rand(3, 64, 80, generator=torch.Generator().manual_seed(8))
    means = {}
    for severity in ("mild", "medium", "heavy"):
        values = [
            synthesize_smoke(clean, severity, seed)[2].mean().item()
            for seed in range(20, 40)
        ]
        means[severity] = sum(values) / len(values)
    assert means["mild"] > means["medium"] > means["heavy"]


def test_smoke_generation_does_not_mutate_clean_tensor():
    clean = torch.rand(3, 32, 32)
    original = clean.clone()
    smoke, _, _, _ = synthesize_smoke(clean, "heavy", seed=99)
    assert torch.equal(clean, original)
    assert not torch.equal(smoke, clean)


def test_paired_severities_share_spatial_field_and_airlight():
    clean = torch.rand(3, 64, 80, generator=torch.Generator().manual_seed(9))
    results = {
        severity: synthesize_smoke(
            clean, severity, seed=20260820, morphology="patchy"
        )
        for severity in ("mild", "medium", "heavy")
    }

    means = [results[level][2].mean().item()
             for level in ("mild", "medium", "heavy")]
    assert means[0] > means[1] > means[2]

    mild_tau = -results["mild"][2].log()
    medium_tau = -results["medium"][2].log()
    mild_shape = mild_tau / mild_tau.mean()
    medium_shape = medium_tau / medium_tau.mean()
    assert torch.allclose(mild_shape, medium_shape, atol=2e-5, rtol=2e-5)

    airlights = {
        results[level][3]["airlight_mean"]
        for level in ("mild", "medium", "heavy")
    }
    assert len(airlights) == 1


def test_morphology_sampling_is_configurable_and_deterministic():
    first = torch.Generator().manual_seed(77)
    second = torch.Generator().manual_seed(77)
    sequence_a = [sample_morphology(first) for _ in range(100)]
    sequence_b = [sample_morphology(second) for _ in range(100)]
    assert sequence_a == sequence_b
    assert set(sequence_a) == {"local", "patchy", "diffuse"}

    forced = [
        sample_morphology(
            first,
            morphologies=("patchy",),
            probabilities=(1.0,),
        )
        for _ in range(10)
    ]
    assert forced == ["patchy"] * 10


def test_morphology_presets_produce_distinct_transmission_fields():
    clean = torch.rand(3, 48, 64, generator=torch.Generator().manual_seed(5))
    fields = {
        morphology: synthesize_smoke(
            clean, "medium", seed=123, morphology=morphology
        )[2]
        for morphology in ("local", "patchy", "diffuse")
    }
    assert not torch.equal(fields["local"], fields["patchy"])
    assert not torch.equal(fields["patchy"], fields["diffuse"])


def test_msrs_validation_from_train_split_disables_random_augmentation():
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        vi_dir = root / "train" / "vi"
        ir_dir = root / "train" / "ir"
        label_dir = root / "train" / "Segmentation_labels"
        vi_dir.mkdir(parents=True)
        ir_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        width = height = 32
        rgb = Image.new("RGB", (width, height))
        rgb.putdata([
            (x * 7 % 256, y * 9 % 256, (x + y) * 5 % 256)
            for y in range(height)
            for x in range(width)
        ])
        infrared = Image.new("L", (width, height))
        infrared.putdata([
            (x * 3 + y * 11) % 256
            for y in range(height)
            for x in range(width)
        ])
        label = Image.new("L", (width, height))
        label.putdata([
            (x // 4 + y // 4) % 9
            for y in range(height)
            for x in range(width)
        ])
        rgb.save(vi_dir / "sample.png")
        infrared.save(ir_dir / "sample.png")
        label.save(label_dir / "sample.png")

        loader = get_test_loader(
            "msrs",
            root,
            batch_size=1,
            num_workers=0,
            img_size=(16, 16),
            smoke_config={
                "enabled": True,
                "probability": 1.0,
                "severities": ["medium"],
                "morphologies": ["patchy"],
                "morphology_probabilities": [1.0],
                "base_seed": 12026,
                "deterministic": True,
            },
            split="train",
            has_label=True,
        )

        assert loader.dataset.augment is False
        first = loader.dataset[0]
        second = loader.dataset[0]
        for key in (
            "vi_clean",
            "vi_smoke",
            "smoke_mask",
            "transmission",
            "ir",
            "label",
        ):
            assert torch.equal(first[key], second[key])
        assert first["severity"] == second["severity"] == "medium"
        assert first["morphology"] == second["morphology"] == "patchy"
        assert first["seed"] == second["seed"]

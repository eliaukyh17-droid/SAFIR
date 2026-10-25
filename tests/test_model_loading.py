"""Loader contract and selective-scan checks, runnable with unittest."""

import os
import unittest

import torch

from models.selective_scan import selective_scan_reference
from safir.model import PRIOR_PREFIXES, _validate_state, load_model


class ScanTests(unittest.TestCase):
    def test_known_scalar_recurrence(self):
        # x[t] = 0.5*x[t-1] + 2, y[t] = 3*x[t] + 0.25.
        u = torch.ones(1, 1, 3)
        A = torch.tensor([[-0.6931471805599453]])
        B = torch.full((1, 1, 1, 3), 2.0)
        C = torch.full_like(B, 3.0)
        y, state = selective_scan_reference(
            u, torch.ones_like(u), A, B, C, torch.tensor([0.25]),
            return_last_state=True,
        )
        torch.testing.assert_close(y, torch.tensor([[[6.25, 9.25, 10.75]]]))
        torch.testing.assert_close(state, torch.tensor([[[3.5]]]))

    def test_grouped_scan_matches_independent_channels_and_gradients(self):
        torch.manual_seed(7)
        u = torch.randn(2, 6, 5, requires_grad=True)
        delta = torch.randn_like(u, requires_grad=True)
        A = -torch.rand(6, 4)
        B = torch.randn(2, 2, 4, 5)
        C = torch.randn_like(B)
        bias = torch.randn(6)
        result = selective_scan_reference(u, delta, A, B, C, delta_bias=bias, delta_softplus=True)
        independent = []
        for channel in range(6):
            group = channel // 3
            independent.append(selective_scan_reference(
                u[:, channel:channel + 1], delta[:, channel:channel + 1],
                A[channel:channel + 1], B[:, group:group + 1], C[:, group:group + 1],
                delta_bias=bias[channel:channel + 1], delta_softplus=True,
            ))
        torch.testing.assert_close(result, torch.cat(independent, dim=1))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(u.grad).all())
        self.assertGreater(float(u.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(delta.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA needed to compare original mamba kernel")
    def test_reference_matches_mamba_cuda(self):
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        except (ImportError, OSError):
            self.skipTest("mamba-ssm is not installed")
        torch.manual_seed(17)
        u = torch.randn(2, 8, 29, device="cuda")
        delta = torch.randn_like(u)
        A = -torch.rand(8, 5, device="cuda")
        B = torch.randn(2, 4, 5, 29, device="cuda")
        C = torch.randn_like(B)
        D = torch.randn(8, device="cuda")
        bias = torch.randn_like(D)
        ref = selective_scan_reference(u, delta, A, B, C, D, delta_bias=bias, delta_softplus=True)
        actual = selective_scan_fn(u, delta, A, B, C, D, delta_bias=bias, delta_softplus=True)
        torch.testing.assert_close(ref, actual, rtol=2e-4, atol=2e-4)


class LoaderContractTests(unittest.TestCase):
    def setUp(self):
        self.retained = {f"component.weight{i}": torch.zeros(1) for i in range(807)}
        self.excluded = {PRIOR_PREFIXES[0] + str(i): torch.zeros(1) for i in range(719)}
        expected = {**self.retained, **self.excluded}
        self.model = type("StateFixture", (), {"state_dict": lambda instance: expected})()

    def test_prior_exclusion_is_exact(self):
        self.assertEqual(_validate_state(self.model, self.retained), self.excluded.keys())

    def test_equal_count_does_not_hide_wrong_keys(self):
        bad = dict(self.retained)
        bad.pop("component.weight0")
        bad["unexpected.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "Incompatible checkpoint"):
            _validate_state(self.model, bad)

    def test_shape_and_dtype_are_checked(self):
        for replacement in (torch.zeros(2), torch.zeros(1, dtype=torch.float64)):
            bad = {**self.retained, "component.weight0": replacement}
            with self.assertRaisesRegex(RuntimeError, "shape/dtype mismatch"):
                _validate_state(self.model, bad)


@unittest.skipUnless(os.environ.get("SAFIR_TEST_CHECKPOINT"), "Set SAFIR_TEST_CHECKPOINT and prior paths for integration")
class ReleasedModelTests(unittest.TestCase):
    def test_released_checkpoint_loads_and_routes(self):
        torch.set_num_threads(2)
        model = load_model(
            os.environ["SAFIR_TEST_CHECKPOINT"],
            segformer_path=os.environ["SAFIR_TEST_SEGFORMER"],
            yolo_path=os.environ["SAFIR_TEST_YOLO"],
            device="cpu", scan_backend="reference",
        )
        self.assertEqual(model.load_report["retained_tensors_loaded"], 807)
        self.assertEqual(model.load_report["prior_tensors_loaded_externally"], 719)
        torch.manual_seed(4)
        visible, infrared = torch.rand(1, 3, 64, 64), torch.rand(1, 1, 64, 64)
        with torch.inference_mode():
            normal, normal_aux = model(visible, infrared, degradation_condition="normal", return_aux=True)
            smoke, smoke_aux = model(visible, infrared, degradation_condition="smoke", return_aux=True)
        self.assertFalse(normal_aux["expert_evaluated"])
        self.assertTrue(torch.equal(normal, normal_aux["canonical"]))
        self.assertTrue(smoke_aux["expert_evaluated"])
        self.assertEqual(smoke.shape, infrared.shape)
        self.assertTrue(torch.isfinite(smoke).all())
        self.assertLessEqual(float((smoke - smoke_aux["canonical"]).abs().max()), 0.150001)
        with self.assertRaises(ValueError):
            model(visible, infrared, degradation_condition="unknown")


if __name__ == "__main__":
    unittest.main()

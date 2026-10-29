import torch

from torchquantlib.calibration import OptionSurfaceQuote, calibrate_option_surface
from torchquantlib.models.stochastic_model import StochasticModel
from torchquantlib.models.stochastic_volatility.heston import Heston
from torchquantlib.models.stochastic_volatility.sabr import SABR


class DummyModel(StochasticModel):
    def simulate(self, S0, T, N, **kwargs):
        return torch.ones(N, device=self.device, dtype=self.dtype) * S0


def test_default_dtype_is_float32():
    model = DummyModel({"dummy": torch.tensor(1.0, requires_grad=True)})
    assert model.dtype == torch.float32
    assert model.params["dummy"].dtype == torch.float32


def test_dtype_kwarg_at_construction():
    model = DummyModel(
        {"dummy": torch.tensor(1.0, dtype=torch.float64, requires_grad=True)},
        dtype=torch.float64,
    )
    assert model.dtype == torch.float64
    assert model.params["dummy"].dtype == torch.float64
    assert model.params["dummy"].requires_grad


def test_build_parameters_dtype():
    params = StochasticModel.build_parameters({"a": 0.5}, dtype="float64")
    assert params["a"].dtype == torch.float64
    assert params["a"].requires_grad
    default = StochasticModel.build_parameters({"a": 0.5})
    assert default["a"].dtype == torch.float32


def test_set_dtype_converts_params_and_aux_tensors():
    model = SABR(alpha_init=0.2, beta_init=0.5, rho_init=-0.2, nu_init=0.3, F0=100.0)
    model.set_dtype(torch.float64)
    assert model.dtype == torch.float64
    for param in model.params.values():
        assert param.dtype == torch.float64
    for aux in model._auxiliary_tensors.values():
        assert aux.dtype == torch.float64
    assert model.F0.dtype == torch.float64
    # requires_grad preserved through conversion
    assert model.params["alpha"].requires_grad


def test_set_dtype_idempotent_and_rejects_non_float():
    model = DummyModel({"dummy": torch.tensor(1.0, requires_grad=True)})
    assert model.set_dtype(torch.float32) is model
    try:
        model.set_dtype(torch.int64)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_to_accepts_dtype_nn_module_style():
    model = DummyModel({"dummy": torch.tensor(1.0, requires_grad=True)})
    model.to(torch.float64)
    assert model.dtype == torch.float64
    assert model.params["dummy"].dtype == torch.float64
    model.to("cpu", torch.float32)
    assert model.dtype == torch.float32
    assert model.params["dummy"].dtype == torch.float32


def test_heston_characteristic_function_price_follows_model_dtype():
    kwargs = dict(kappa_init=1.0, theta_init=0.04, sigma_v_init=0.5, rho_init=-0.5, v0_init=0.04)
    model32 = Heston(**kwargs)
    price32 = model32.characteristic_function_price(S0=100.0, K=100.0, T=1.0, r=0.05)
    assert price32.dtype == torch.float32

    model64 = Heston(**kwargs)
    model64.set_dtype(torch.float64)
    price64 = model64.characteristic_function_price(S0=100.0, K=100.0, T=1.0, r=0.05)
    assert price64.dtype == torch.float64
    loss = price64
    loss.backward()
    assert model64.params["kappa"].grad is not None
    assert model64.params["kappa"].grad.dtype == torch.float64


def test_calibrate_option_surface_dtype_propagates():
    true_model = SABR(alpha_init=0.22, beta_init=0.5, rho_init=-0.2, nu_init=0.35, F0=100.0)
    true_model.set_dtype(torch.float64)
    fit_model = SABR(alpha_init=0.15, beta_init=0.5, rho_init=-0.2, nu_init=0.35, F0=100.0)
    for name in ("beta", "rho", "nu"):
        fit_model.params[name].requires_grad_(False)

    quotes = [
        OptionSurfaceQuote(
            strike=strike,
            expiry=1.0,
            target=float(true_model.implied_volatility(K=strike, T=1.0, r=0.01).item()),
            rate=0.01,
            target_type="implied_vol",
            model="lognormal",
            name=f"quote_{strike}",
        )
        for strike in [90.0, 100.0, 110.0]
    ]

    calibration = calibrate_option_surface(
        fit_model, quotes, num_epochs=150, lr=0.05, verbose=False, dtype=torch.float64
    )
    assert fit_model.dtype == torch.float64
    for param in fit_model.params.values():
        assert param.dtype == torch.float64
    assert calibration.result.final_loss < 1e-6
    assert abs(fit_model.params["alpha"].item() - 0.22) < 0.02

import torch

from torchquantlib.calibration import (
    OptionSurfaceQuote,
    build_option_surface_instruments,
    calibrate_option_surface,
    calibrate_option_surface_multistart,
)
from torchquantlib.models.stochastic_volatility.heston import Heston
from torchquantlib.models.stochastic_volatility.sabr import SABR


def test_build_option_surface_instruments_for_heston_price_quote():
    model = Heston()
    quote = OptionSurfaceQuote(strike=100.0, expiry=1.0, target=10.0, spot=100.0, rate=0.05, target_type="price")
    instruments = build_option_surface_instruments(model, [quote])

    assert len(instruments) == 1
    prediction = instruments[0].pricer(model)
    assert torch.is_tensor(prediction)
    assert prediction.item() >= 0


def test_option_surface_calibration_for_sabr_implied_vols():
    true_model = SABR(alpha_init=0.22, beta_init=0.5, rho_init=-0.2, nu_init=0.35, F0=100.0)
    fit_model = SABR(alpha_init=0.15, beta_init=0.5, rho_init=-0.2, nu_init=0.35, F0=100.0)
    fit_model.params["beta"].requires_grad_(False)
    fit_model.params["rho"].requires_grad_(False)
    fit_model.params["nu"].requires_grad_(False)

    strikes = [90.0, 100.0, 110.0]
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
        for strike in strikes
    ]

    calibration = calibrate_option_surface(fit_model, quotes, num_epochs=150, lr=0.05, verbose=False)

    assert calibration.result.final_loss < 1e-6
    assert abs(fit_model.params["alpha"].item() - 0.22) < 0.02


def test_option_surface_multistart_returns_best_run():
    true_model = SABR(alpha_init=0.22, beta_init=0.5, rho_init=-0.2, nu_init=0.35, F0=100.0)
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

    def model_factory():
        model = SABR(alpha_init=0.15, beta_init=0.5, rho_init=-0.2, nu_init=0.35, F0=100.0)
        model.params["beta"].requires_grad_(False)
        model.params["rho"].requires_grad_(False)
        model.params["nu"].requires_grad_(False)
        return model

    result = calibrate_option_surface_multistart(model_factory, quotes, starts=2, num_epochs=100, lr=0.05, verbose=False)
    assert len(result.runs) == 2
    assert result.best.result.final_loss == min(result.final_losses)
    assert "mean_loss" in result.summary
    assert "std_loss" in result.summary
    assert "alpha" in result.parameter_dispersion
    assert "mean" in result.parameter_dispersion["alpha"]

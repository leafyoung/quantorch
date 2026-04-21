import torch

from torchquantlib.calibration import (
    YieldCurvePoint,
    build_yield_curve_instruments,
    calibrate_yield_curve,
    calibrate_yield_curve_multistart,
)
from torchquantlib.models.interest_rate.black_karasinski import BlackKarasinski
from torchquantlib.models.interest_rate.hull_white import HullWhiteModel
from torchquantlib.models.interest_rate.vasicek import Vasicek


def test_build_yield_curve_instruments_for_vasicek():
    model = Vasicek()
    points = [YieldCurvePoint(maturity=1.0, target=0.97, target_type="price")]
    instruments = build_yield_curve_instruments(model, points)

    assert len(instruments) == 1
    prediction = instruments[0].pricer(model)
    assert torch.is_tensor(prediction)
    assert prediction.item() > 0


def test_calibrate_yield_curve_recovers_vasicek_r0_from_synthetic_prices():
    true_model = Vasicek(kappa_init=0.2, theta_init=0.04, sigma_init=0.01, r0_init=0.03)
    fit_model = Vasicek(kappa_init=0.2, theta_init=0.04, sigma_init=0.01, r0_init=0.01)
    fit_model.params["kappa"].requires_grad_(False)
    fit_model.params["theta"].requires_grad_(False)
    fit_model.params["sigma"].requires_grad_(False)

    maturities = [1.0, 2.0, 5.0]
    points = [
        YieldCurvePoint(
            maturity=maturity,
            target=float(true_model.zero_coupon_bond_price(maturity).item()),
            target_type="price",
            name=f"p_{maturity}",
        )
        for maturity in maturities
    ]

    calibration = calibrate_yield_curve(fit_model, points, num_epochs=200, lr=0.05, verbose=False)

    assert calibration.result.final_loss < 1e-6
    assert abs(fit_model.params["r0"].item() - 0.03) < 0.01


def test_calibrate_yield_curve_recovers_hull_white_r0_from_synthetic_prices():
    true_model = HullWhiteModel(a_init=0.2, sigma_init=0.01, r0_init=0.03)
    fit_model = HullWhiteModel(a_init=0.2, sigma_init=0.01, r0_init=0.01)
    fit_model.params["a"].requires_grad_(False)
    fit_model.params["sigma"].requires_grad_(False)

    maturities = [1.0, 2.0, 4.0]
    points = [
        YieldCurvePoint(
            maturity=maturity,
            target=float(true_model.zero_coupon_bond_price(maturity).item()),
            target_type="price",
            name=f"hw_{maturity}",
        )
        for maturity in maturities
    ]

    calibration = calibrate_yield_curve(fit_model, points, num_epochs=200, lr=0.05, verbose=False)

    assert calibration.result.final_loss < 1e-6
    assert abs(fit_model.params["r0"].item() - 0.03) < 0.01


def test_initial_discount_curve_fit_for_hull_white_and_black_karasinski():
    maturities = torch.tensor([1.0, 2.0, 5.0])
    discount_factors = torch.exp(-0.03 * maturities)

    hw_model = HullWhiteModel().fit_initial_discount_curve(maturities, discount_factors)
    bk_model = BlackKarasinski().fit_initial_discount_curve(maturities, discount_factors)

    assert torch.isclose(hw_model.zero_coupon_bond_price(2.0), discount_factors[1], atol=1e-6)
    assert torch.isclose(bk_model.zero_coupon_bond_price(2.0), discount_factors[1], atol=1e-6)
    assert torch.isclose(hw_model.instantaneous_forward_rate(2.0), torch.tensor(0.03), atol=5e-3)
    assert torch.isclose(bk_model.instantaneous_forward_rate(2.0), torch.tensor(0.03), atol=5e-3)
    assert torch.isfinite(hw_model.theta(2.0))


def test_yield_curve_multistart_returns_best_run():
    true_model = HullWhiteModel(a_init=0.2, sigma_init=0.01, r0_init=0.03)
    points = [
        YieldCurvePoint(maturity=maturity, target=float(true_model.zero_coupon_bond_price(maturity).item()), target_type="price")
        for maturity in [1.0, 2.0, 4.0]
    ]

    def model_factory():
        model = HullWhiteModel(a_init=0.2, sigma_init=0.01, r0_init=0.01)
        model.params["a"].requires_grad_(False)
        model.params["sigma"].requires_grad_(False)
        return model

    result = calibrate_yield_curve_multistart(model_factory, points, starts=2, num_epochs=100, lr=0.05, verbose=False)
    assert len(result.runs) == 2
    assert result.best.result.final_loss == min(result.final_losses)
    assert "mean_loss" in result.summary
    assert "std_loss" in result.summary

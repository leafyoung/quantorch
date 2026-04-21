import torch

from torchquantlib.calibration import YieldCurvePoint, build_yield_curve_instruments, calibrate_yield_curve
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

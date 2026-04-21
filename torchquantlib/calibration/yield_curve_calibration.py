from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from torchquantlib.calibration.market_calibrator import CalibrationInstrument, MarketCalibrator, MarketCalibrationResult


@dataclass(frozen=True)
class YieldCurvePoint:
    maturity: float
    target: float
    target_type: str = "price"
    weight: float = 1.0
    bid_ask_spread: float | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class YieldCurveCalibrationResult:
    calibrator: MarketCalibrator
    result: MarketCalibrationResult


@dataclass(frozen=True)
class YieldCurveMultiStartResult:
    best: YieldCurveCalibrationResult
    runs: tuple[YieldCurveCalibrationResult, ...]
    final_losses: tuple[float, ...]
    summary: dict[str, float] = field(default_factory=dict)
    parameter_dispersion: dict[str, dict[str, float]] = field(default_factory=dict)


def _point_name(point: YieldCurvePoint) -> str:
    if point.name is not None:
        return point.name
    return f"{point.target_type}_T{point.maturity:g}"


def fit_initial_curve_from_points(model, points: Sequence[YieldCurvePoint]):
    if not hasattr(model, "fit_initial_discount_curve"):
        raise TypeError("Model does not support initial discount-curve fitting.")
    if not points:
        raise ValueError("At least one yield-curve point is required.")

    sorted_points = sorted(points, key=lambda point: point.maturity)
    maturities = torch.tensor([point.maturity for point in sorted_points], device=model.device, dtype=torch.float32)
    discount_factors = []
    for point in sorted_points:
        maturity_tensor = torch.tensor(point.maturity, device=model.device, dtype=torch.float32)
        target_tensor = torch.tensor(point.target, device=model.device, dtype=torch.float32)
        if point.target_type == "price":
            discount_factors.append(target_tensor)
        elif point.target_type == "yield":
            discount_factors.append(torch.exp(-target_tensor * maturity_tensor))
        else:
            raise ValueError("target_type must be 'price' or 'yield'.")
    model.fit_initial_discount_curve(maturities, torch.stack(discount_factors))
    return model


def build_yield_curve_instruments(model, points: Sequence[YieldCurvePoint]) -> list[CalibrationInstrument]:
    if not hasattr(model, "zero_coupon_bond_price"):
        raise TypeError("Yield-curve calibration requires a model with zero_coupon_bond_price(maturity, r_t=None).")

    instruments: list[CalibrationInstrument] = []
    for point in points:
        if point.target_type not in {"price", "yield"}:
            raise ValueError("target_type must be 'price' or 'yield'.")

        def pricer(bound_model, maturity=point.maturity, target_type=point.target_type):
            if target_type == "price":
                return bound_model.zero_coupon_bond_price(maturity)
            return bound_model.zero_coupon_yield(maturity)

        instruments.append(
            CalibrationInstrument(
                name=_point_name(point),
                target=point.target,
                pricer=pricer,
                weight=point.weight,
                bid_ask_spread=point.bid_ask_spread,
                metadata={**point.metadata, "maturity": point.maturity, "target_type": point.target_type},
            )
        )
    return instruments


def calibrate_yield_curve(
    model,
    points: Sequence[YieldCurvePoint],
    *,
    num_epochs: int = 100,
    verbose: bool = True,
    **market_calibrator_kwargs,
) -> YieldCurveCalibrationResult:
    instruments = build_yield_curve_instruments(model, points)
    calibrator = MarketCalibrator(model, instruments, **market_calibrator_kwargs)
    result = calibrator.calibrate(num_epochs=num_epochs, verbose=verbose)
    return YieldCurveCalibrationResult(calibrator=calibrator, result=result)


def calibrate_yield_curve_multistart(
    model_factory: Callable[[], Any],
    points: Sequence[YieldCurvePoint],
    *,
    starts: int = 3,
    num_epochs: int = 100,
    verbose: bool = True,
    **market_calibrator_kwargs,
) -> YieldCurveMultiStartResult:
    runs: list[YieldCurveCalibrationResult] = []
    for _ in range(max(1, int(starts))):
        model = model_factory()
        runs.append(
            calibrate_yield_curve(
                model,
                points,
                num_epochs=num_epochs,
                verbose=verbose,
                **market_calibrator_kwargs,
            )
        )
    best = min(runs, key=lambda run: run.result.final_loss)
    final_losses = tuple(run.result.final_loss for run in runs)
    loss_tensor = torch.tensor(final_losses, dtype=torch.float32)

    parameter_dispersion: dict[str, dict[str, float]] = {}
    param_names = best.result.params.keys()
    for name in param_names:
        values = [run.result.params[name] for run in runs if isinstance(run.result.params[name], float)]
        if not values:
            continue
        tensor_values = torch.tensor(values, dtype=torch.float32)
        parameter_dispersion[name] = {
            "mean": float(tensor_values.mean().item()),
            "std": float(tensor_values.std(unbiased=False).item()),
            "min": float(tensor_values.min().item()),
            "max": float(tensor_values.max().item()),
        }

    return YieldCurveMultiStartResult(
        best=best,
        runs=tuple(runs),
        final_losses=final_losses,
        summary={
            "mean_loss": float(loss_tensor.mean().item()),
            "std_loss": float(loss_tensor.std(unbiased=False).item()),
            "best_loss": float(loss_tensor.min().item()),
            "worst_loss": float(loss_tensor.max().item()),
        },
        parameter_dispersion=parameter_dispersion,
    )


__all__ = [
    "YieldCurveCalibrationResult",
    "YieldCurveMultiStartResult",
    "YieldCurvePoint",
    "build_yield_curve_instruments",
    "calibrate_yield_curve",
    "calibrate_yield_curve_multistart",
    "fit_initial_curve_from_points",
]

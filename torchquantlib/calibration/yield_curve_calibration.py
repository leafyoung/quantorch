from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

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


def _point_name(point: YieldCurvePoint) -> str:
    if point.name is not None:
        return point.name
    return f"{point.target_type}_T{point.maturity:g}"


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


__all__ = [
    "YieldCurveCalibrationResult",
    "YieldCurvePoint",
    "build_yield_curve_instruments",
    "calibrate_yield_curve",
]

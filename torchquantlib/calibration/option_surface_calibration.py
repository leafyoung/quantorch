from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from torchquantlib.calibration.market_calibrator import CalibrationInstrument, MarketCalibrator, MarketCalibrationResult
from torchquantlib.core.asset_pricing.letsberational import implied_volatility as black_scholes_implied_volatility
from torchquantlib.models.stochastic_volatility.heston import Heston
from torchquantlib.models.stochastic_volatility.sabr import SABR, _bachelier_price, _black76_price


@dataclass(frozen=True)
class OptionSurfaceQuote:
    strike: float
    expiry: float
    target: float
    option_type: str = "call"
    rate: float = 0.0
    spot: float | None = None
    dividend: float = 0.0
    target_type: str = "price"
    model: str = "lognormal"
    shift: float | None = None
    weight: float = 1.0
    bid_ask_spread: float | None = None
    vega: float | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptionSurfaceCalibrationResult:
    calibrator: MarketCalibrator
    result: MarketCalibrationResult


def _quote_name(quote: OptionSurfaceQuote) -> str:
    if quote.name is not None:
        return quote.name
    return f"{quote.option_type}_K{quote.strike:g}_T{quote.expiry:g}_{quote.target_type}"


def _build_heston_pricer(model: Heston, quote: OptionSurfaceQuote):
    def pricer(bound_model):
        price = bound_model.characteristic_function_price(
            S0=quote.spot,
            K=quote.strike,
            T=quote.expiry,
            r=quote.rate,
            option_type=quote.option_type,
            q=quote.dividend,
        )
        if quote.target_type == "price":
            return price
        return black_scholes_implied_volatility(
            price,
            torch.tensor(quote.spot, device=bound_model.device, dtype=torch.float32),
            torch.tensor(quote.strike, device=bound_model.device, dtype=torch.float32),
            torch.tensor(quote.expiry, device=bound_model.device, dtype=torch.float32),
            torch.tensor(quote.rate, device=bound_model.device, dtype=torch.float32),
            is_call=quote.option_type == "call",
        )

    return pricer


def _build_sabr_pricer(model: SABR, quote: OptionSurfaceQuote):
    def pricer(bound_model):
        if quote.target_type == "implied_vol":
            return bound_model.implied_volatility(
                K=quote.strike,
                T=quote.expiry,
                r=quote.rate,
                q=quote.dividend,
                model=quote.model,
                shift=quote.shift,
            )

        expiry = bound_model.tensor(quote.expiry)
        rate = bound_model.tensor(quote.rate)
        strike = bound_model.tensor(quote.strike)
        forward = bound_model._base_forward_from_state(bound_model.F0, expiry, rate, quote.dividend)
        if quote.model == "lognormal":
            shifted_forward, shifted_strike, _ = bound_model._shifted_forward_and_strike(forward, strike, quote.shift, positive_required=True)
            vol = bound_model._hagan_lognormal_implied_volatility(strike, expiry, forward=forward, shift=quote.shift)
            return _black76_price(quote.option_type, shifted_forward, shifted_strike, expiry, vol, rate)

        shifted_forward, shifted_strike, _ = bound_model._shifted_forward_and_strike(forward, strike, quote.shift, positive_required=False)
        vol = bound_model._hagan_normal_implied_volatility(strike, expiry, forward=forward, shift=quote.shift)
        return _bachelier_price(quote.option_type, shifted_forward, shifted_strike, expiry, vol, rate)

    return pricer


def build_option_surface_instruments(model, quotes: Sequence[OptionSurfaceQuote]) -> list[CalibrationInstrument]:
    instruments: list[CalibrationInstrument] = []
    for quote in quotes:
        if quote.target_type not in {"price", "implied_vol"}:
            raise ValueError("target_type must be 'price' or 'implied_vol'.")
        if quote.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'.")

        if isinstance(model, Heston):
            if quote.spot is None:
                raise ValueError("Heston option-surface quotes require a spot value.")
            pricer = _build_heston_pricer(model, quote)
        elif isinstance(model, SABR):
            pricer = _build_sabr_pricer(model, quote)
        else:
            raise TypeError("build_option_surface_instruments currently supports Heston and SABR models only.")

        instruments.append(
            CalibrationInstrument(
                name=_quote_name(quote),
                target=quote.target,
                pricer=pricer,
                weight=quote.weight,
                bid_ask_spread=quote.bid_ask_spread,
                vega=quote.vega,
                metadata={
                    **quote.metadata,
                    "strike": quote.strike,
                    "expiry": quote.expiry,
                    "target_type": quote.target_type,
                },
            )
        )
    return instruments


def calibrate_option_surface(
    model,
    quotes: Sequence[OptionSurfaceQuote],
    *,
    num_epochs: int = 100,
    verbose: bool = True,
    **market_calibrator_kwargs,
) -> OptionSurfaceCalibrationResult:
    instruments = build_option_surface_instruments(model, quotes)
    calibrator = MarketCalibrator(model, instruments, **market_calibrator_kwargs)
    result = calibrator.calibrate(num_epochs=num_epochs, verbose=verbose)
    return OptionSurfaceCalibrationResult(calibrator=calibrator, result=result)


__all__ = [
    "OptionSurfaceCalibrationResult",
    "OptionSurfaceQuote",
    "build_option_surface_instruments",
    "calibrate_option_surface",
]

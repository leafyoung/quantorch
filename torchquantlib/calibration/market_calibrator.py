from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
import torch.optim as optim

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class CalibrationInstrument:
    """Single market target used in instrument-space calibration.

    Attributes:
        name: Human-readable label for diagnostics.
        target: Observed market quote, such as an option price or implied vol.
        pricer: Callable receiving the model and returning the model quote.
        weight: User-defined weight multiplier.
        bid_ask_spread: Optional spread scale for residual normalization.
        vega: Optional vega scale for residual normalization.
        metadata: Arbitrary context attached to the instrument.
    """

    name: str
    target: float | torch.Tensor
    pricer: Callable[[Any], torch.Tensor | float]
    weight: float = 1.0
    bid_ask_spread: float | None = None
    vega: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketCalibrationConfig:
    lr: float = 0.01
    objective: str = "mse"
    normalize_by_spread: bool = True
    normalize_by_vega: bool = False
    regularization_strength: float = 0.0
    parameter_priors: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketCalibrationResult:
    final_loss: float
    loss_history: tuple[float, ...]
    params: dict[str, Any]
    residuals: dict[str, float]
    duration_ms: float
    num_epochs: int
    objective_decomposition: dict[str, float] = field(default_factory=dict)
    predictions: dict[str, float] = field(default_factory=dict)
    targets: dict[str, float] = field(default_factory=dict)
    residual_summary: dict[str, float] = field(default_factory=dict)


class MarketCalibrator:
    """Calibrate a model directly to instrument-level market targets."""

    def __init__(
        self,
        model,
        instruments: Sequence[CalibrationInstrument],
        *,
        optimizer_cls=optim.Adam,
        lr: float = 0.01,
        objective: str = "mse",
        normalize_by_spread: bool = True,
        normalize_by_vega: bool = False,
        regularization_strength: float = 0.0,
        parameter_priors: dict[str, float] | None = None,
    ):
        if not instruments:
            raise ValueError("At least one calibration instrument is required.")
        self.model = model
        self.instruments = list(instruments)
        self.config = MarketCalibrationConfig(
            lr=lr,
            objective=objective,
            normalize_by_spread=normalize_by_spread,
            normalize_by_vega=normalize_by_vega,
            regularization_strength=regularization_strength,
            parameter_priors={} if parameter_priors is None else dict(parameter_priors),
        )
        self.optimizer_cls = optimizer_cls
        self.optimizer = self._setup_optimizer()
        self.loss_history: list[float] = []
        self.last_result: MarketCalibrationResult | None = None

    def _setup_optimizer(self):
        params = [param for param in self.model.params.values() if param.requires_grad]
        return self.optimizer_cls(params, lr=self.config.lr)

    def _tensor(self, value: float | torch.Tensor) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
        return tensor.to(device=self.model.device, dtype=torch.float32)

    def _weight_scale(self, instrument: CalibrationInstrument) -> torch.Tensor:
        scale = torch.tensor(float(instrument.weight), device=self.model.device, dtype=torch.float32)
        if self.config.normalize_by_spread and instrument.bid_ask_spread is not None:
            scale = scale / max(float(instrument.bid_ask_spread), 1e-6)
        if self.config.normalize_by_vega and instrument.vega is not None:
            scale = scale / max(abs(float(instrument.vega)), 1e-6)
        return scale

    def _objective_term(self, prediction: torch.Tensor, target: torch.Tensor, instrument: CalibrationInstrument) -> torch.Tensor:
        residual = prediction - target
        weighted_residual = self._weight_scale(instrument) * residual
        if self.config.objective == "mse":
            return weighted_residual**2
        if self.config.objective == "mae":
            return torch.abs(weighted_residual)
        raise ValueError(f"Unsupported objective: {self.config.objective}")

    def _instrument_prediction(self, instrument: CalibrationInstrument) -> torch.Tensor:
        prediction = instrument.pricer(self.model)
        return self._tensor(prediction)

    def _regularization_term(self) -> torch.Tensor:
        if self.config.regularization_strength <= 0 or not self.config.parameter_priors:
            return torch.zeros((), device=self.model.device, dtype=torch.float32)

        penalties = []
        for name, prior_value in self.config.parameter_priors.items():
            if name not in self.model.params:
                continue
            prior_tensor = self._tensor(prior_value)
            penalties.append((self.model.params[name] - prior_tensor) ** 2)
        if not penalties:
            return torch.zeros((), device=self.model.device, dtype=torch.float32)
        return self.config.regularization_strength * torch.stack([penalty.mean() for penalty in penalties]).sum()

    def _instrument_snapshot(self) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        residuals: dict[str, float] = {}
        predictions: dict[str, float] = {}
        targets: dict[str, float] = {}
        with torch.no_grad():
            for instrument in self.instruments:
                target = self._tensor(instrument.target)
                prediction = self._instrument_prediction(instrument)
                residuals[instrument.name] = float((prediction - target).item())
                predictions[instrument.name] = float(prediction.item())
                targets[instrument.name] = float(target.item())
        return residuals, predictions, targets

    @staticmethod
    def _residual_summary(residuals: dict[str, float]) -> dict[str, float]:
        if not residuals:
            return {"mae": 0.0, "rmse": 0.0, "max_abs": 0.0}
        values = torch.tensor(list(residuals.values()), dtype=torch.float32)
        return {
            "mae": float(values.abs().mean().item()),
            "rmse": float(torch.sqrt((values**2).mean()).item()),
            "max_abs": float(values.abs().max().item()),
        }

    def _emit_event(self, payload: dict[str, Any], *, verbose: bool) -> None:
        message = json.dumps(payload, sort_keys=True)
        LOGGER.info(message)
        if verbose:
            print(message)

    def get_calibrated_params(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, param in self.model.params.items():
            if param.dim() == 0:
                result[name] = param.item()
            else:
                result[name] = param.detach().cpu().numpy()
        return result

    def evaluate(self) -> MarketCalibrationResult:
        if self.last_result is not None:
            return self.last_result
        return MarketCalibrationResult(
            final_loss=self.loss_history[-1] if self.loss_history else float("nan"),
            loss_history=tuple(self.loss_history),
            params=self.get_calibrated_params(),
            residuals=self._instrument_snapshot()[0],
            duration_ms=0.0,
            num_epochs=0,
            objective_decomposition={},
            predictions=self._instrument_snapshot()[1],
            targets=self._instrument_snapshot()[2],
            residual_summary=self._residual_summary(self._instrument_snapshot()[0]),
        )

    def calibrate(self, num_epochs: int = 100, *, verbose: bool = True) -> MarketCalibrationResult:
        run_started_at = time.perf_counter()
        self.loss_history = []
        self._emit_event(
            {
                "event": "market_calibration_started",
                "module": __name__,
                "status": "running",
                "num_epochs": int(num_epochs),
                "num_instruments": len(self.instruments),
                "objective": self.config.objective,
            },
            verbose=verbose,
        )

        for epoch in range(int(num_epochs)):
            self.optimizer.zero_grad()
            terms = []
            for instrument in self.instruments:
                target = self._tensor(instrument.target)
                prediction = self._instrument_prediction(instrument)
                terms.append(self._objective_term(prediction, target, instrument))
            data_loss = torch.stack(terms).mean()
            regularization_loss = self._regularization_term()
            loss = data_loss + regularization_loss
            loss.backward()
            self.optimizer.step()
            self.model._apply_constraints()
            self.loss_history.append(float(loss.item()))

        duration_ms = round((time.perf_counter() - run_started_at) * 1000, 3)
        final_data_terms = []
        for instrument in self.instruments:
            target = self._tensor(instrument.target)
            prediction = self._instrument_prediction(instrument)
            final_data_terms.append(self._objective_term(prediction, target, instrument).detach())
        final_data_loss = torch.stack(final_data_terms).mean().item()
        final_regularization_loss = float(self._regularization_term().detach().item())
        residuals, predictions, targets = self._instrument_snapshot()
        self.last_result = MarketCalibrationResult(
            final_loss=self.loss_history[-1],
            loss_history=tuple(self.loss_history),
            params=self.get_calibrated_params(),
            residuals=residuals,
            duration_ms=duration_ms,
            num_epochs=int(num_epochs),
            objective_decomposition={
                "data_loss": float(final_data_loss),
                "regularization_loss": final_regularization_loss,
            },
            predictions=predictions,
            targets=targets,
            residual_summary=self._residual_summary(residuals),
        )
        self._emit_event(
            {
                "event": "market_calibration_completed",
                "module": __name__,
                "status": "success",
                "num_epochs": int(num_epochs),
                "num_instruments": len(self.instruments),
                "objective": self.config.objective,
                "final_loss": round(self.last_result.final_loss, 6),
                "duration_ms": duration_ms,
            },
            verbose=verbose,
        )
        return self.last_result

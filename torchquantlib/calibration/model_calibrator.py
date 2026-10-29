from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.optim as optim
from geomloss import SamplesLoss

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class CalibrationConfig:
    S0: float | None = None
    T: float = 1.0
    loss_type: str = "sinkhorn"
    p: int = 2
    blur: float = 0.05
    lr: float = 0.01


class ModelCalibrator:
    """Calibrate a stochastic model against observed data."""

    def __init__(
        self,
        model,
        observed_data,
        S0=None,
        T=1.0,
        loss_type="sinkhorn",
        p=2,
        blur=0.05,
        optimizer_cls=optim.Adam,
        lr=0.01,
    ):
        self.model = model
        self.config = CalibrationConfig(S0=S0, T=T, loss_type=loss_type, p=p, blur=blur, lr=lr)
        self.optimizer_cls = optimizer_cls
        self.observed_data = self._prepare_observed_data(observed_data)
        self.loss_fn = self._build_loss_function()
        self.optimizer = self._setup_optimizer()
        self.loss_history: list[float] = []
        self.last_run_summary: dict[str, Any] | None = None

    def _prepare_observed_data(self, observed_data) -> torch.Tensor:
        dtype = getattr(self.model, "dtype", torch.float32)
        return torch.as_tensor(observed_data, dtype=dtype, device=self.model.device)

    def _build_loss_function(self):
        if self.config.loss_type == "sinkhorn":
            return SamplesLoss(self.config.loss_type, p=self.config.p, blur=self.config.blur)
        if self.config.loss_type == "mse":
            return torch.nn.MSELoss()
        if self.config.loss_type == "log_likelihood":
            return self._log_likelihood_loss
        raise ValueError(f"Unsupported loss type: {self.config.loss_type}")

    def _setup_optimizer(self):
        params = [param for param in self.model.params.values() if param.requires_grad]
        return self.optimizer_cls(params, lr=self.config.lr)

    def _log_likelihood_loss(self, simulated_data, observed_data):
        mean = simulated_data.mean(dim=0)
        variance = simulated_data.var(dim=0) + 1e-6
        log_likelihood = -0.5 * torch.sum((observed_data - mean) ** 2 / variance + torch.log(variance))
        return -log_likelihood

    def _reshape_samples(self, simulated_data: torch.Tensor, observed_data: torch.Tensor):
        if simulated_data.dim() > 1:
            return simulated_data.view(simulated_data.shape[0], -1), observed_data.view(observed_data.shape[0], -1)
        return simulated_data.view(-1, 1), observed_data.view(-1, 1)

    def _format_param_value(self, value: Any):
        if isinstance(value, float):
            return round(value, 6)
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    def _build_event_payload(self, event: str, status: str, **fields: Any) -> dict[str, Any]:
        payload = {
            "event": event,
            "module": __name__,
            "function": "calibrate",
            "status": status,
            "loss_type": self.config.loss_type,
        }
        payload.update(fields)
        return payload

    def _emit_event(self, payload: dict[str, Any], *, verbose: bool) -> None:
        message = json.dumps(payload, sort_keys=True)
        LOGGER.info(message)
        if verbose:
            print(message)

    def calibrate(self, num_epochs=1000, batch_size=None, steps=100, verbose=True):
        batch_size = len(self.observed_data) if batch_size is None else int(batch_size)
        observed_tensor = self.observed_data
        progress_interval = max(1, num_epochs // 10)
        run_started_at = time.perf_counter()

        self.loss_history = []
        self._emit_event(
            self._build_event_payload(
                event="model_calibration_started",
                status="running",
                batch_size=batch_size,
                num_epochs=num_epochs,
                steps=steps,
            ),
            verbose=verbose,
        )

        try:
            for epoch in range(num_epochs):
                self.optimizer.zero_grad()
                simulated_data = self.model.simulate(
                    S0=self.config.S0,
                    T=self.config.T,
                    N=batch_size,
                    steps=steps,
                )

                simulated_flat, observed_flat = self._reshape_samples(simulated_data, observed_tensor)
                loss = self.loss_fn(simulated_flat, observed_flat)
                loss.backward()
                self.optimizer.step()
                self.model._apply_constraints()

                loss_value = float(loss.item())
                self.loss_history.append(loss_value)

                should_report = epoch % progress_interval == 0 or epoch == num_epochs - 1
                if should_report:
                    self._emit_event(
                        self._build_event_payload(
                            event="model_calibration_progress",
                            status="running",
                            epoch=epoch,
                            num_epochs=num_epochs,
                            loss=round(loss_value, 6),
                            batch_size=batch_size,
                            steps=steps,
                            params={
                                name: self._format_param_value(value)
                                for name, value in self.get_calibrated_params().items()
                            },
                        ),
                        verbose=verbose,
                    )
        except Exception as exc:
            duration_ms = round((time.perf_counter() - run_started_at) * 1000, 3)
            self._emit_event(
                self._build_event_payload(
                    event="model_calibration_failed",
                    status="error",
                    batch_size=batch_size,
                    duration_ms=duration_ms,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                ),
                verbose=verbose,
            )
            raise

        duration_ms = round((time.perf_counter() - run_started_at) * 1000, 3)
        self.last_run_summary = {
            "batch_size": batch_size,
            "duration_ms": duration_ms,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "num_epochs": num_epochs,
            "steps": steps,
        }
        self._emit_event(
            self._build_event_payload(
                event="model_calibration_completed",
                status="success",
                batch_size=batch_size,
                duration_ms=duration_ms,
                final_loss=round(self.loss_history[-1], 6) if self.loss_history else None,
                num_epochs=num_epochs,
                steps=steps,
            ),
            verbose=verbose,
        )

    def get_calibrated_params(self):
        result = {}
        for name, param in self.model.params.items():
            if param.dim() == 0:
                result[name] = param.item()
            else:
                result[name] = param.detach().cpu().numpy()
        return result

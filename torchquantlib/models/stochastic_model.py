from __future__ import annotations

from typing import Any, Mapping

import torch


class StochasticModel:
    """Shared base class for stochastic simulation models.

    The base class owns the common model state that was previously duplicated
    across several files: parameter storage, device resolution, auxiliary tensor
    registration, and parameter constraint application.
    """

    parameter_constraints: dict[str, dict[str, float]] = {}

    def __init__(
        self,
        params: Mapping[str, torch.Tensor] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = self.resolve_device(device)
        self.params: dict[str, torch.Tensor] = {}
        self._auxiliary_tensors: dict[str, torch.Tensor] = {}

        if params:
            self.params = self._normalize_params(params)

    @staticmethod
    def resolve_device(device: torch.device | str | None = None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @classmethod
    def build_parameters(
        cls,
        values: Mapping[str, Any],
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        resolved_device = cls.resolve_device(device)
        return {
            name: cls._move_tensor(
                value if isinstance(value, torch.Tensor) else torch.as_tensor(value),
                resolved_device,
                requires_grad=True,
            )
            for name, value in values.items()
        }

    @staticmethod
    def _move_tensor(
        tensor: torch.Tensor,
        device: torch.device,
        *,
        requires_grad: bool,
    ) -> torch.Tensor:
        moved = tensor.detach().to(device)
        if moved.is_floating_point() or moved.is_complex():
            moved.requires_grad_(requires_grad)
        return moved

    def _normalize_params(
        self,
        params: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        normalized: dict[str, torch.Tensor] = {}
        for name, param in params.items():
            if not isinstance(param, torch.Tensor):
                raise AttributeError(f"Parameter '{name}' must be a torch.Tensor.")
            normalized[name] = self._move_tensor(
                param,
                self.device,
                requires_grad=param.requires_grad,
            )
        return normalized

    def tensor(self, value: Any, *, requires_grad: bool = False) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return self._move_tensor(tensor, self.device, requires_grad=requires_grad)

    def register_auxiliary_tensor(
        self,
        name: str,
        value: Any,
        *,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        moved = self._move_tensor(tensor, self.device, requires_grad=requires_grad)
        self._auxiliary_tensors[name] = moved
        setattr(self, name, moved)
        return moved

    def to(self, device: torch.device | str) -> "StochasticModel":
        self.device = self.resolve_device(device)

        for name, param in list(self.params.items()):
            moved = self._move_tensor(param, self.device, requires_grad=param.requires_grad)
            self.params[name] = moved
            if hasattr(self, name):
                setattr(self, name, moved)

        for name, tensor in list(self._auxiliary_tensors.items()):
            moved = self._move_tensor(tensor, self.device, requires_grad=tensor.requires_grad)
            self._auxiliary_tensors[name] = moved
            setattr(self, name, moved)

        return self

    def constrained_param(self, name: str) -> torch.Tensor:
        param = self.params[name]
        bounds = self.parameter_constraints.get(name)
        if not bounds:
            return param
        return torch.clamp(param, min=bounds.get("min"), max=bounds.get("max"))

    def simulate(self, S0, T, N, **kwargs):
        raise NotImplementedError("simulate method must be implemented by subclasses.")

    def _apply_constraints(self) -> None:
        with torch.no_grad():
            for name, bounds in self.parameter_constraints.items():
                if name not in self.params:
                    continue
                self.params[name].clamp_(min=bounds.get("min"), max=bounds.get("max"))

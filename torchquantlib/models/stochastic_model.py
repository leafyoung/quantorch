from __future__ import annotations

from typing import Any, Mapping

import torch


class StochasticModel:
    """Shared base class for stochastic simulation models.

    Dtype handling: models default to float32 parameters (historical behavior).
    A model can opt into float64 either at construction (``StochasticModel(
    params, dtype=torch.float64)`` or a subclass ``__init__`` forwarding
    ``dtype`` into ``build_parameters``/``super().__init__``) or after
    construction via ``model.set_dtype(torch.float64)`` / ``model.to(dtype=...)``.
    Because concrete models build their parameters before calling
    ``super().__init__``, the post-construction ``set_dtype`` path is the
    supported route for existing subclasses; parameter conversion is exact
    (float32 -> float64 upcast) and should happen before an optimizer is
    attached to ``model.params``.
    """

    parameter_constraints: dict[str, dict[str, float]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        simulate = cls.__dict__.get("simulate")
        if simulate is None or getattr(simulate, "_stochastic_model_wrapped", False):
            return

        def wrapped_simulate(self, *args, **kwargs):
            result = simulate(self, *args, **kwargs)
            if isinstance(result, torch.Tensor) and self.params:
                anchor_terms = [param.sum() * 0 for param in self.params.values() if isinstance(param, torch.Tensor)]
                if anchor_terms:
                    result = result + torch.stack(anchor_terms).sum()
            return result

        wrapped_simulate._stochastic_model_wrapped = True  # type: ignore[attr-defined]
        cls.simulate = wrapped_simulate  # type: ignore[assignment]

    def __init__(
        self,
        params: Mapping[str, torch.Tensor] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        self.device = self.resolve_device(device)
        self.dtype = self.resolve_dtype(dtype)
        self.params: dict[str, torch.Tensor] = {}
        self._auxiliary_tensors: dict[str, torch.Tensor] = {}

        if params:
            self.params = self._normalize_params(params)

    @staticmethod
    def resolve_device(device: torch.device | str | None = None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    _DTYPE_ALIASES: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }

    @staticmethod
    def resolve_dtype(dtype: torch.dtype | str | None = None) -> torch.dtype:
        if dtype is None:
            return torch.float32
        if isinstance(dtype, str):
            key = dtype.lower()
            if key not in StochasticModel._DTYPE_ALIASES:
                raise ValueError(f"Unsupported dtype string: {dtype!r}.")
            resolved = StochasticModel._DTYPE_ALIASES[key]
        elif isinstance(dtype, torch.dtype):
            resolved = dtype
        else:
            raise TypeError(f"dtype must be a torch.dtype or str, got {type(dtype).__name__}.")
        if not resolved.is_floating_point:
            raise ValueError(f"dtype must be a floating-point type, got {resolved}.")
        return resolved

    @classmethod
    def build_parameters(
        cls,
        values: Mapping[str, Any],
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> dict[str, torch.Tensor]:
        resolved_device = cls.resolve_device(device)
        resolved_dtype = cls.resolve_dtype(dtype)
        return {
            name: cls._move_tensor(
                value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=resolved_dtype),
                resolved_device,
                requires_grad=True,
                dtype=resolved_dtype,
            )
            for name, value in values.items()
        }

    @staticmethod
    def _move_tensor(
        tensor: torch.Tensor,
        device: torch.device,
        *,
        requires_grad: bool,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        moved = tensor.detach().to(device)
        if moved.is_floating_point() or moved.is_complex():
            moved = moved.to(dtype=dtype)
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
                dtype=self.dtype,
            )
        return normalized

    def tensor(self, value: Any, *, requires_grad: bool = False) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=self.dtype)
        return self._move_tensor(tensor, self.device, requires_grad=requires_grad, dtype=self.dtype)

    def register_auxiliary_tensor(
        self,
        name: str,
        value: Any,
        *,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=self.dtype)
        moved = self._move_tensor(tensor, self.device, requires_grad=requires_grad, dtype=self.dtype)
        self._auxiliary_tensors[name] = moved
        setattr(self, name, moved)
        return moved

    def _retarget(
        self,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> "StochasticModel":
        for name, param in list(self.params.items()):
            moved = self._move_tensor(
                param,
                self.device,
                requires_grad=param.requires_grad,
                dtype=self.dtype if dtype is None else dtype,
            )
            self.params[name] = moved
            if hasattr(self, name):
                setattr(self, name, moved)

        for name, tensor in list(self._auxiliary_tensors.items()):
            moved = self._move_tensor(
                tensor,
                self.device,
                requires_grad=tensor.requires_grad,
                dtype=self.dtype if dtype is None else dtype,
            )
            self._auxiliary_tensors[name] = moved
            setattr(self, name, moved)

        return self

    def set_dtype(self, dtype: torch.dtype | str) -> "StochasticModel":
        """Convert all parameters and auxiliary tensors to ``dtype`` in place.

        Intended to be called before attaching an optimizer to ``model.params``;
        optimizer state created beforehand is not converted.
        """
        resolved = self.resolve_dtype(dtype)
        if resolved == self.dtype:
            return self
        self.dtype = resolved
        return self._retarget(device=None, dtype=resolved)

    def to(
        self,
        device: torch.device | str | torch.dtype | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> "StochasticModel":
        if isinstance(device, torch.dtype):
            # nn.Module-style convenience: model.to(torch.float64)
            device, dtype = None, device
        if device is not None:
            self.device = self.resolve_device(device)
        if dtype is not None:
            self.dtype = self.resolve_dtype(dtype)

        if device is None and dtype is None:
            return self

        return self._retarget(
            device=self.device if device is not None else None,
            dtype=self.dtype if dtype is not None else None,
        )

    def _safe_lower_bound(self, min_value: float | None, reference: torch.Tensor) -> float | None:
        if min_value is None:
            return None
        return torch.nextafter(
            torch.tensor(min_value, dtype=reference.dtype, device=reference.device),
            torch.tensor(float("inf"), dtype=reference.dtype, device=reference.device),
        ).item()

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
                min_value = self._safe_lower_bound(bounds.get("min"), self.params[name])
                self.params[name].clamp_(min=min_value, max=bounds.get("max"))

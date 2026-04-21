from __future__ import annotations

import math

import torch

from torchquantlib.models.stochastic_model import StochasticModel

EPSILON = 1e-8
PI = math.pi


class Heston(StochasticModel):
    """Heston stochastic volatility model.

    ``simulate`` returns terminal asset prices from a full-truncation Euler
    discretization of the Heston dynamics under the model's ``mu`` parameter.
    ``option_price`` supports either Monte Carlo pricing or characteristic-
    function pricing. Characteristic-function pricing is the default because it
    is faster and more numerically stable for European options.
    """

    parameter_constraints = {
        "kappa": {"min": 1.0001e-6},
        "theta": {"min": 1.0001e-6},
        "sigma_v": {"min": 1.0001e-6},
        "rho": {"min": -0.9989, "max": 0.9989},
        "v0": {"min": 1.0001e-6},
    }

    def __init__(
        self,
        kappa_init=1.0,
        theta_init=0.04,
        sigma_v_init=0.5,
        rho_init=-0.5,
        v0_init=0.04,
        mu_init=0.0,
    ):
        super().__init__(
            self.build_parameters(
                {
                    "kappa": kappa_init,
                    "theta": theta_init,
                    "sigma_v": sigma_v_init,
                    "rho": rho_init,
                    "v0": v0_init,
                    "mu": mu_init,
                }
            )
        )

    def _tensor64(self, value) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float64)
        return tensor.to(device=self.device, dtype=torch.float64)

    def _simulate_terminal_paths(
        self,
        S0,
        T,
        N,
        *,
        steps=100,
        drift=None,
        antithetic: bool = False,
        seed: int | None = None,
        normalize_martingale: bool = False,
    ) -> torch.Tensor:
        spot = torch.clamp(self.tensor(S0), min=EPSILON)
        expiry = self.tensor(T)
        if expiry.item() <= 0:
            return spot.expand(int(N)).clone()

        num_paths = int(N)
        steps = max(1, int(steps))
        dt = expiry / steps
        sqrt_dt = torch.sqrt(dt)

        kappa = self.constrained_param("kappa")
        theta = self.constrained_param("theta")
        sigma_v = self.constrained_param("sigma_v")
        rho = self.constrained_param("rho")
        model_drift = self.params["mu"] if drift is None else self.tensor(drift)

        base_count = (num_paths + 1) // 2 if antithetic else num_paths
        prices = torch.ones(base_count, device=self.device, dtype=spot.dtype) * spot
        variance = torch.ones_like(prices) * self.constrained_param("v0")
        rho_residual = torch.sqrt(torch.clamp(1.0 - rho**2, min=EPSILON))

        def _draw_normals() -> tuple[torch.Tensor, torch.Tensor]:
            z1 = torch.randn(base_count, device=self.device, dtype=spot.dtype)
            z2 = torch.randn(base_count, device=self.device, dtype=spot.dtype)
            if antithetic:
                z1 = torch.cat([z1, -z1], dim=0)
                z2 = torch.cat([z2, -z2], dim=0)
            return z1[:num_paths], z2[:num_paths]

        def _run_simulation() -> torch.Tensor:
            simulated_prices = prices.repeat(2)[:num_paths].clone() if antithetic else prices.clone()
            simulated_variance = variance.repeat(2)[:num_paths].clone() if antithetic else variance.clone()

            for _ in range(steps):
                z_spot, z_aux = _draw_normals()
                z_var = rho * z_spot + rho_residual * z_aux

                variance_pos = torch.clamp(simulated_variance, min=0.0)
                simulated_variance = torch.clamp(
                    simulated_variance
                    + kappa * (theta - variance_pos) * dt
                    + sigma_v * torch.sqrt(torch.clamp(variance_pos, min=EPSILON)) * sqrt_dt * z_var,
                    min=0.0,
                )
                simulated_prices = simulated_prices * torch.exp(
                    (model_drift - 0.5 * variance_pos) * dt
                    + torch.sqrt(torch.clamp(variance_pos, min=EPSILON)) * sqrt_dt * z_spot
                )

            if normalize_martingale:
                target_forward = spot * torch.exp(model_drift * expiry)
                simulated_prices = simulated_prices * (target_forward / torch.clamp(simulated_prices.mean(), min=EPSILON))

            return simulated_prices

        if seed is None:
            return _run_simulation()

        fork_devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            return _run_simulation()

    def _log_price_characteristic_function(self, u: torch.Tensor, spot: torch.Tensor, expiry: torch.Tensor, rate: torch.Tensor, dividend: torch.Tensor) -> torch.Tensor:
        kappa = self.constrained_param("kappa").to(dtype=torch.float64)
        theta = self.constrained_param("theta").to(dtype=torch.float64)
        sigma_v = self.constrained_param("sigma_v").to(dtype=torch.float64)
        rho = self.constrained_param("rho").to(dtype=torch.float64)
        v0 = self.constrained_param("v0").to(dtype=torch.float64)

        a = kappa * theta
        iu = 1j * u
        d = torch.sqrt((rho * sigma_v * iu - kappa) ** 2 + sigma_v**2 * (u**2 + iu))
        g = (kappa - rho * sigma_v * iu - d) / (kappa - rho * sigma_v * iu + d)
        exp_neg_d_t = torch.exp(-d * expiry)

        c_term = (rate - dividend) * iu * expiry + (a / sigma_v**2) * (
            (kappa - rho * sigma_v * iu - d) * expiry
            - 2.0 * torch.log((1.0 - g * exp_neg_d_t) / (1.0 - g))
        )
        d_term = ((kappa - rho * sigma_v * iu - d) / sigma_v**2) * ((1.0 - exp_neg_d_t) / (1.0 - g * exp_neg_d_t))
        return torch.exp(c_term + d_term * v0 + iu * torch.log(spot))

    def characteristic_function_price(
        self,
        S0,
        K,
        T,
        r,
        *,
        option_type: str = "call",
        q=0.0,
        integration_limit: float = 100.0,
        integration_points: int = 2000,
    ) -> torch.Tensor:
        option_style = option_type.lower()
        if option_style not in {"call", "put"}:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        spot = torch.clamp(self._tensor64(S0), min=EPSILON)
        strike = torch.clamp(self._tensor64(K), min=EPSILON)
        expiry = self._tensor64(T)
        rate = self._tensor64(r)
        dividend = self._tensor64(q)

        if expiry.item() <= 0:
            intrinsic = torch.relu(spot - strike) if option_style == "call" else torch.relu(strike - spot)
            return intrinsic.to(dtype=self.tensor(0.0).dtype)

        u = torch.linspace(1e-5, integration_limit, max(256, int(integration_points)), device=self.device, dtype=torch.float64)
        log_strike = torch.log(strike)
        kernel = torch.exp(-1j * u * log_strike) / (1j * u)
        phi_u = self._log_price_characteristic_function(u, spot, expiry, rate, dividend)
        phi_u_minus_i = self._log_price_characteristic_function(u - 1j, spot, expiry, rate, dividend)
        phi_minus_i = self._log_price_characteristic_function(
            torch.tensor([complex(0.0, -1.0)], device=self.device, dtype=torch.complex128),
            spot,
            expiry,
            rate,
            dividend,
        )[0]

        p1_integrand = torch.real(kernel * phi_u_minus_i / phi_minus_i)
        p2_integrand = torch.real(kernel * phi_u)

        p1 = 0.5 + torch.trapz(p1_integrand, u) / PI
        p2 = 0.5 + torch.trapz(p2_integrand, u) / PI
        discount_dividend = torch.exp(-dividend * expiry)
        discount_rate = torch.exp(-rate * expiry)
        call_price = spot * discount_dividend * p1 - strike * discount_rate * p2

        if option_style == "call":
            return torch.clamp(call_price, min=0.0).to(dtype=self.tensor(0.0).dtype)

        put_price = call_price - spot * discount_dividend + strike * discount_rate
        return torch.clamp(put_price, min=0.0).to(dtype=self.tensor(0.0).dtype)

    def monte_carlo_option_price(self, S0, K, T, r, *, option_type="call", q=0.0, N=10000, steps=100, seed: int = 1234) -> torch.Tensor:
        option_style = option_type.lower()
        if option_style not in {"call", "put"}:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        strike = self.tensor(K)
        rate = self.tensor(r)
        dividend = self.tensor(q)
        expiry = self.tensor(T)
        terminal_prices = self._simulate_terminal_paths(
            S0=S0,
            T=expiry,
            N=N,
            steps=steps,
            drift=rate - dividend,
            antithetic=True,
            seed=seed,
            normalize_martingale=True,
        )
        payoffs = torch.relu(terminal_prices - strike) if option_style == "call" else torch.relu(strike - terminal_prices)
        return torch.exp(-rate * expiry) * payoffs.mean()

    def simulate(self, S0, T, N, steps=100):
        return self._simulate_terminal_paths(S0, T, N, steps=steps)

    def option_price(
        self,
        S0,
        K,
        T,
        r,
        option_type="call",
        N=10000,
        steps=100,
        *,
        method: str = "characteristic_function",
        q=0.0,
        integration_limit: float = 100.0,
        integration_points: int = 2000,
        seed: int = 1234,
    ):
        method_name = method.lower()
        if method_name == "characteristic_function":
            price = self.characteristic_function_price(
                S0,
                K,
                T,
                r,
                option_type=option_type,
                q=q,
                integration_limit=integration_limit,
                integration_points=integration_points,
            )
        elif method_name == "monte_carlo":
            price = self.monte_carlo_option_price(S0, K, T, r, option_type=option_type, q=q, N=N, steps=steps, seed=seed)
        else:
            raise ValueError("method must be 'characteristic_function' or 'monte_carlo'.")
        return float(price.item())

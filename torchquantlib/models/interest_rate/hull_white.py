import torch

from torchquantlib.models.stochastic_model import StochasticModel


def _linear_interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    if xp.numel() == 1:
        return fp.reshape(())
    clamped_x = torch.clamp(x, min=xp[0], max=xp[-1])
    upper = torch.searchsorted(xp, clamped_x, right=False)
    upper = torch.clamp(upper, min=1, max=xp.numel() - 1)
    lower = upper - 1
    x0 = xp[lower]
    x1 = xp[upper]
    y0 = fp[lower]
    y1 = fp[upper]
    weight = (clamped_x - x0) / torch.clamp(x1 - x0, min=1e-12)
    return y0 + weight * (y1 - y0)


class HullWhiteModel(StochasticModel):
    """Hull-White short-rate model."""

    def fit_initial_discount_curve(self, maturities, discount_factors):
        curve_maturities = self.register_auxiliary_tensor("curve_maturities", maturities)
        curve_discount_factors = self.register_auxiliary_tensor("curve_discount_factors", discount_factors)
        if curve_maturities.dim() != 1 or curve_discount_factors.dim() != 1 or curve_maturities.shape != curve_discount_factors.shape:
            raise ValueError("maturities and discount_factors must be one-dimensional tensors of the same shape.")
        return self

    def initial_discount_factor(self, maturity):
        maturity_tensor = self.tensor(maturity)
        if not hasattr(self, "curve_maturities") or not hasattr(self, "curve_discount_factors"):
            return torch.exp(-self.params["r0"] * maturity_tensor)
        return _linear_interpolate(maturity_tensor, self.curve_maturities, self.curve_discount_factors)

    def instantaneous_forward_rate(self, maturity):
        maturity_tensor = self.tensor(maturity)
        bump = torch.tensor(1e-4, device=self.device, dtype=maturity_tensor.dtype)
        t_up = maturity_tensor + bump
        t_down = torch.clamp(maturity_tensor - bump, min=0.0)
        log_discount_up = -torch.log(torch.clamp(self.initial_discount_factor(t_up), min=1e-12))
        log_discount_down = -torch.log(torch.clamp(self.initial_discount_factor(t_down), min=1e-12))
        denominator = torch.clamp(t_up - t_down, min=1e-12)
        return (log_discount_up - log_discount_down) / denominator

    def theta(self, maturity):
        maturity_tensor = self.tensor(maturity)
        a = torch.clamp(self.params["a"], min=1e-12)
        sigma = torch.clamp(self.params["sigma"], min=0.0)
        bump = torch.tensor(1e-4, device=self.device, dtype=maturity_tensor.dtype)
        fwd_up = self.instantaneous_forward_rate(maturity_tensor + bump)
        fwd_down = self.instantaneous_forward_rate(torch.clamp(maturity_tensor - bump, min=0.0))
        dfdt = (fwd_up - fwd_down) / torch.clamp(2.0 * bump, min=1e-12)
        forward = self.instantaneous_forward_rate(maturity_tensor)
        return dfdt + a * forward + (sigma**2 / (2.0 * a)) * (1.0 - torch.exp(-2.0 * a * maturity_tensor))

    def zero_coupon_bond_price(self, maturity, r_t=None):
        """Curve-consistent Hull-White ZCB price (Brigo-Mercurio (3.39)) at t=0.

        P(0,T) = A(0,T) * exp(-B(0,T) * r_t) with
            B(0,T) = (1 - exp(-a*T)) / a
            ln A(0,T) = ln P^M(0,T) + B(0,T) * f^M(0,0)
        (the sigma^2 term of A vanishes at t=0). P^M is the fitted initial
        discount curve — or the flat exp(-r0*t) default — and f^M(0,0) the
        initial instantaneous forward rate.

        Replaces the previous zero-drift approximation, which ignored the
        market-curve drift and did not reproduce the initial curve (e.g. with
        the flat default curve it priced the 1y ZCB at 0.97187 instead of
        exp(-r0) = 0.97045 for r0=3%, a=0.1, sigma=0.01).

        The formula is evaluated in float64 (Heston's pattern): f^M(0,0) is a
        one-sided finite difference of log discount factors at a 1e-4 bump,
        which cancels catastrophically in float32; the result is cast back to
        the model dtype.
        """
        maturity_tensor = self.tensor(maturity)
        if maturity_tensor.item() <= 0:
            return torch.ones((), device=self.device, dtype=self.params["r0"].dtype)

        if r_t is None and hasattr(self, "curve_maturities") and hasattr(self, "curve_discount_factors"):
            return self.initial_discount_factor(maturity_tensor)

        a = self.params["a"].double().clamp(min=1e-12)
        short_rate = self.params["r0"].double() if r_t is None else self.tensor(r_t).double()
        maturity64 = maturity_tensor.double()

        def discount_factor_f64(t):
            if hasattr(self, "curve_maturities") and hasattr(self, "curve_discount_factors"):
                return _linear_interpolate(
                    t, self.curve_maturities.double(), self.curve_discount_factors.double())
            return torch.exp(-self.params["r0"].double() * t)

        b_term = (1 - torch.exp(-a * maturity64)) / a
        market_price = discount_factor_f64(maturity64)
        bump = torch.tensor(1e-4, device=self.device, dtype=maturity64.dtype)
        f0 = (-torch.log(torch.clamp(discount_factor_f64(bump), min=1e-12))
              + torch.log(torch.clamp(discount_factor_f64(torch.zeros((), device=self.device, dtype=maturity64.dtype)), min=1e-12))) / bump
        a_term = market_price * torch.exp(b_term * f0)
        return (a_term * torch.exp(-b_term * short_rate)).to(dtype=self.tensor(0.0).dtype)

    def zero_coupon_yield(self, maturity, r_t=None):
        maturity_tensor = self.tensor(maturity)
        price = self.zero_coupon_bond_price(maturity_tensor, r_t=r_t)
        return -torch.log(torch.clamp(price, min=1e-12)) / torch.clamp(maturity_tensor, min=1e-12)

    parameter_constraints = {
        "a": {"min": 1e-6},
        "sigma": {"min": 1e-6},
    }

    def __init__(self, a_init=0.1, sigma_init=0.01, r0_init=0.03):
        super().__init__(
            self.build_parameters(
                {
                    "a": a_init,
                    "sigma": sigma_init,
                    "r0": r0_init,
                }
            )
        )

    def simulate(self, S0, T, N, steps=100):
        del S0, steps
        horizon = self.tensor(T)
        a = torch.clamp(self.params["a"], min=0)
        sigma = torch.clamp(self.params["sigma"], min=0)
        r0 = self.params["r0"]

        N = int(N)
        mean = r0 * torch.exp(-a * horizon)

        if sigma.item() == 0:
            return mean.expand(N)

        variance = sigma**2 * (1 - torch.exp(-2 * a * horizon)) / torch.clamp(2 * a, min=1e-12)
        noise = torch.randn(N, device=self.device, dtype=mean.dtype)
        return mean + torch.sqrt(torch.clamp(variance, min=0)) * noise

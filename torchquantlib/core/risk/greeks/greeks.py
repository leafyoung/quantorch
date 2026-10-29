import math

import torch

class MalliavinGreeks:
    """
    A class for calculating option Greeks using Malliavin calculus.
    This approach is particularly useful for complex options where closed-form solutions are not available.

    Estimator-correctness note (2026-10-29): ``european_option_greeks`` and
    ``digital_option_greeks`` were re-derived with proper likelihood-ratio
    (integration-by-parts) / pathwise weights and verified against QuantLib
    1.44 ``AnalyticEuropeanEngine`` greeks at 500k paths over three parameter
    sets (relative errors: delta <= 1.4e-3, gamma <= 1.0e-2, vega <= 1.5e-2,
    theta <= 2.8e-3, rho <= 2.9e-3). The previous implementation multiplied the
    payoff by pathwise derivatives of S_T (e.g. ``S_T*Z/sqrt(T)``) where LR
    weights are required, biasing every European greek by 26x-386x (see
    cross-check results/risk_golden.json). ``barrier_option_greeks``,
    ``lookback_option_delta``, ``basket_option_greeks`` and
    ``asian_option_greeks`` still use the OLD weight construction and are
    BIASED / UNDER REVIEW (same extra-S_T-factor defect; left unchanged).
    """

    def __init__(self, device=None):
        """
        Initialize the MalliavinGreeks class.

        Args:
            device (torch.device, optional): The device to run calculations on. Defaults to GPU if available, else CPU.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def european_option_greeks(self, S0, K, T, r, sigma, num_paths=100000, seed=42):
        """
        Calculate Greeks for European options under GBM (single-step exact
        terminal simulation, S_T = S0*exp((r - sigma^2/2)T + sigma*sqrt(T)*Z)).

        Estimators (verified vs QuantLib 1.44 AnalyticEuropeanEngine; 500k
        paths, three parameter sets, rel err delta <= 1.4e-3 / gamma <= 1.0e-2 /
        vega <= 1.5e-2 / theta <= 2.8e-3 / rho <= 2.9e-3):
          delta (likelihood-ratio): disc * E[payoff * Z] / (sigma*S0*sqrt(T))
          gamma (2nd-order LR):     [disc*E[payoff*(Z^2-1)]
                                     - sigma*sqrt(T)*disc*E[payoff*Z]]
                                    / (sigma^2 * T * S0^2)
          vega (pathwise):          disc * E[1{S_T>K} * S_T * (sqrt(T)*Z - sigma*T)]
          theta (BS PDE identity):  r*price - r*S0*delta - sigma^2*S0^2*gamma/2
          rho   (identity, q=0):    T * (S0*delta - price)

        Args:
            S0 (float): Initial stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            num_paths (int): Number of Monte Carlo paths
            seed (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing Delta, Gamma, Vega, Theta, and Rho
        """
        torch.manual_seed(seed)

        dt = T
        sqrt_dt = math.sqrt(dt)
        z = torch.randn(num_paths, device=self.device)

        # Simulate stock price at maturity
        ST = S0 * torch.exp((r - 0.5 * sigma**2) * T + sigma * sqrt_dt * z)

        # Calculate option payoff
        payoff = torch.relu(ST - K)
        discount_factor = math.exp(-r * T)

        # Likelihood-ratio / pathwise weights (corrected 2026-10-29; the old
        # code used S_T-scaled pathwise derivatives of S_T here, biasing all
        # greeks by 26x-386x -- see class docstring).
        lr1 = discount_factor * payoff * z                       # first-order LR weight
        lr2 = discount_factor * payoff * (z * z - 1.0)           # second-order LR weight
        st_is_k = (ST > K).to(payoff.dtype)
        dST_dsigma = ST * (sqrt_dt * z - sigma * T)              # pathwise dS_T/dsigma

        # Calculate Greeks
        price = torch.mean(discount_factor * payoff)
        Delta = torch.mean(lr1) / (sigma * S0 * sqrt_dt)
        Gamma = (torch.mean(lr2) - sigma * sqrt_dt * torch.mean(lr1)) / (sigma**2 * T * S0**2)
        Vega = torch.mean(discount_factor * st_is_k * dST_dsigma)
        Theta = r * price - r * S0 * Delta - 0.5 * sigma**2 * S0**2 * Gamma
        Rho = T * (S0 * Delta - price)

        return {
            'Delta': Delta.item(),
            'Gamma': Gamma.item(),
            'Vega': Vega.item(),
            'Theta': Theta.item(),
            'Rho': Rho.item()
        }

    def digital_option_greeks(self, S0, K, T, r, sigma, Q=10.0, num_paths=100000, epsilon=1e-4, seed=42):
        """
        Calculate Greeks for Digital (cash-or-nothing) options under GBM,
        with the Dirac delta approximated by a Gaussian kernel of width
        ``epsilon`` (smoothed payoff h(S_T) = Q*Phi((S_T-K)/epsilon), so
        h' = K_eps, h'' = K_eps').

        Pathwise-kernel estimators (corrected 2026-10-29; kernel bias measured
        by exact quadrature of the estimator expectation vs QuantLib 1.44
        ``AnalyticEuropeanEngine`` ``CashOrNothingPayoff`` at eps=2.0:
        delta -0.50%, gamma -0.65%, vega +0.63%; at eps=0.5 all <= 0.06%,
        confirming convergence to the analytic digital greeks as eps -> 0):
          delta = disc * Q * E[K_eps(S_T-K) * S_T / S0]
          gamma = disc * Q * E[dK_eps/ds (S_T-K) * (S_T/S0)^2],
                  dK_eps/ds = -(x/epsilon) * K_eps(x), x = (S_T-K)/epsilon
                  (this is h''(s)/Q for h(s) = Phi((s-K)/eps); note the tempting
                  -(x/epsilon^2) form carries a spurious extra 1/epsilon)
          vega  = disc * Q * E[K_eps(S_T-K) * S_T * (sqrt(T)*Z - sigma*T)]
        The old code used S_T-scaled pathwise derivatives of S_T as weights
        (wrong sign / magnitude for all three greeks).

        Note: epsilon=1e-4 keeps the kernel estimators high-variance at
        moderate path counts (tests use epsilon=2.0, the documented family
        quirk).

        Args:
            S0 (float): Initial stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            Q (float): Payout amount
            num_paths (int): Number of Monte Carlo paths
            epsilon (float): Width of the Gaussian kernel approximating the Dirac delta
            seed (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing Delta, Gamma, and Vega
        """
        torch.manual_seed(seed)

        dt = T
        sqrt_dt = math.sqrt(dt)
        z = torch.randn(num_paths, device=self.device)

        # Simulate stock price at maturity
        ST = S0 * torch.exp((r - 0.5 * sigma**2) * T + sigma * sqrt_dt * z)

        # Gaussian kernel K_eps approximating the Dirac delta at K, and its
        # derivative K_eps'(x) = -(x / eps^2) * K_eps(x).
        normalizer = epsilon * torch.sqrt(torch.as_tensor(2.0 * torch.pi, device=self.device))
        x = (ST - K) / epsilon
        delta_approx = (1 / normalizer) * torch.exp(-0.5 * x**2)
        # d/ds of K_eps((s-K)/eps) = -(x/eps) * K_eps (NOT -(x/eps^2), which
        # would carry a spurious extra 1/eps and halve gamma at eps=2).
        delta_approx_prime = -(x / epsilon) * delta_approx
        discount_factor = math.exp(-r * T)

        # Pathwise quantities
        dST_dS0 = ST / S0                        # dS_T/dS0 (S0-scaling of GBM)
        dST_dsigma = ST * (sqrt_dt * z - sigma * T)

        # Calculate Greeks
        Delta = discount_factor * Q * torch.mean(delta_approx * dST_dS0)
        Gamma = discount_factor * Q * torch.mean(delta_approx_prime * dST_dS0**2)
        Vega = discount_factor * Q * torch.mean(delta_approx * dST_dsigma)

        return {
            'Delta': Delta.item(),
            'Gamma': Gamma.item(),
            'Vega': Vega.item()
        }

    def barrier_option_greeks(self, S0, K, H, T, r, sigma, option_type='up-and-out', num_paths=100000, num_steps=50, seed=42):
        """
        Calculate Greeks for Barrier options using Malliavin calculus.

        .. warning::
            BIASED / UNDER REVIEW (2026-10-29): this estimator still uses the
            old weight construction (payoff * S_T-scaled pathwise derivatives
            of S_T where likelihood-ratio weights are required), the same
            defect that biased ``european_option_greeks`` by 26x-386x. Values
            are NOT trustworthy; left unchanged pending a QuantLib-verified
            re-derivation.

        Args:
            S0 (float): Initial stock price
            K (float): Strike price
            H (float): Barrier level
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            option_type (str): Type of barrier option ('up-and-out' or 'down-and-out')
            num_paths (int): Number of Monte Carlo paths
            num_steps (int): Number of time steps in the simulation
            seed (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing Delta, Gamma, and Vega
        """
        torch.manual_seed(seed)

        dt = T / num_steps
        S = torch.zeros((num_paths, num_steps+1), device=self.device)
        S[:, 0] = S0
        z = torch.randn((num_paths, num_steps), device=self.device)

        # Simulate stock price paths
        for i in range(num_steps):
            S[:, i+1] = S[:, i] * torch.exp((r - 0.5 * sigma**2) * dt + sigma * torch.sqrt(dt) * z[:, i])

        # Check if barrier is hit
        if option_type == 'up-and-out':
            hit_barrier = torch.any(S[:, 1:] >= H, dim=1)
        elif option_type == 'down-and-out':
            hit_barrier = torch.any(S[:, 1:] <= H, dim=1)
        else:
            raise ValueError("Invalid option_type. Use 'up-and-out' or 'down-and-out'.")

        # Calculate payoff
        payoff = torch.relu(S[:, -1] - K)
        payoff = payoff * (~hit_barrier)
        discount_factor = torch.exp(-r * T)

        # Malliavin derivatives
        Dt_ST = S[:, -1] * torch.sum(z, dim=1) / torch.sqrt(dt)
        Dt_ST = Dt_ST * (~hit_barrier)
        D2t_ST = Dt_ST * (torch.sum(z, dim=1) / torch.sqrt(dt) - sigma * torch.sqrt(dt))
        D2t_ST = D2t_ST * (~hit_barrier)
        D_sigma_ST = S[:, -1] * (torch.sum(z**2 - 1, dim=1) * torch.sqrt(dt))
        D_sigma_ST = D_sigma_ST * (~hit_barrier)

        # Calculate Greeks
        Delta = discount_factor * torch.mean(payoff * Dt_ST / (sigma * S0))
        Gamma = discount_factor * torch.mean(payoff * D2t_ST / (sigma**2 * S0**2))
        Vega = discount_factor * torch.mean(payoff * D_sigma_ST / sigma)

        return {
            'Delta': Delta.item(),
            'Gamma': Gamma.item(),
            'Vega': Vega.item()
        }

    def lookback_option_delta(self, S0, T, r, sigma, num_paths=100000, num_steps=50, seed=42, **kwargs):
        """
        Calculate Delta for Lookback options using Malliavin calculus.

        .. warning::
            BIASED / UNDER REVIEW (2026-10-29): old weight construction
            (see the class docstring); left unchanged pending a
            QuantLib-verified re-derivation.

        Args:
            S0 (float): Initial stock price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            num_paths (int): Number of Monte Carlo paths
            num_steps (int): Number of time steps in the simulation
            seed (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing Delta
        """
        torch.manual_seed(seed)

        dt = T / num_steps
        S = torch.zeros((num_paths, num_steps+1), device=self.device)
        S[:, 0] = S0
        z = torch.randn((num_paths, num_steps), device=self.device)

        # Simulate stock price paths
        for i in range(num_steps):
            S[:, i+1] = S[:, i] * torch.exp((r - 0.5 * sigma**2) * dt + sigma * torch.sqrt(dt) * z[:, i])

        # Calculate minimum price and payoff
        S_min, _ = torch.min(S[:, :-1], dim=1)
        payoff = torch.relu(S[:, -1] - S_min)
        discount_factor = torch.exp(-r * T)

        # Malliavin derivatives
        Dt_ST = S[:, -1] * torch.sum(z, dim=1) / torch.sqrt(dt)
        indicator = (S_min.unsqueeze(1) == S[:, :-1]).float()
        Dt_Smin = S[:, :-1] * z / torch.sqrt(dt)
        Dt_Smin = torch.sum(Dt_Smin * indicator, dim=1) / torch.sum(indicator, dim=1)
        Dt_Smin[torch.isnan(Dt_Smin)] = 0.0

        # Calculate Delta
        Delta = discount_factor * torch.mean((Dt_ST - Dt_Smin) / (sigma * S0) * (payoff > 0))

        return {'Delta': Delta.item()}

    def basket_option_greeks(self, S0, K, T, r, sigma, rho, weights, num_paths=100000, seed=42):
        """
        Calculate Greeks for Basket options using Malliavin calculus.

        .. warning::
            BIASED / UNDER REVIEW (2026-10-29): old weight construction
            (see the class docstring); left unchanged pending an independent
            (e.g. common-random-number finite-difference) verification.

        Args:
            S0 (list): Initial stock prices
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (list): Volatilities
            rho (list): Correlation matrix
            weights (list): Weights of assets in the basket
            num_paths (int): Number of Monte Carlo paths
            seed (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing Delta, Gamma, and Vega for each asset
        """
        torch.manual_seed(seed)

        num_assets = len(S0)
        S0 = torch.as_tensor(S0, device=self.device)
        sigma = torch.as_tensor(sigma, device=self.device)
        rho = torch.as_tensor(rho, device=self.device)
        weights = torch.as_tensor(weights, device=self.device)

        # Cholesky decomposition for correlated random numbers
        L = torch.linalg.cholesky(rho)
        z = torch.randn((num_paths, num_assets), device=self.device)
        z = z @ L.T
        dt = T

        # Simulate stock prices at maturity
        ST = S0 * torch.exp((r - 0.5 * sigma**2) * T + sigma * torch.sqrt(dt) * z)
        basket_price = torch.sum(weights * ST, dim=1)
        payoff = torch.relu(basket_price - K)
        discount_factor = torch.exp(-r * T)

        # Malliavin derivatives
        Dt_ST = ST * z / torch.sqrt(dt)

        # Calculate Greeks for each asset
        Delta = []
        Gamma = []
        Vega = []
        for i in range(num_assets):
            Delta_i = discount_factor * torch.mean(payoff * weights[i] * Dt_ST[:, i] / (sigma[i] * S0[i]))
            Delta.append(Delta_i.item())

            D2t_ST = Dt_ST[:, i] * (z[:, i] / torch.sqrt(dt) - sigma[i] * torch.sqrt(dt))
            Gamma_i = discount_factor * torch.mean(payoff * weights[i] * D2t_ST / (sigma[i]**2 * S0[i]**2))
            Gamma.append(Gamma_i.item())

            D_sigma_ST = ST[:, i] * (z[:, i] * torch.sqrt(dt) - sigma[i] * dt)
            Vega_i = discount_factor * torch.mean(payoff * weights[i] * D_sigma_ST / sigma[i])
            Vega.append(Vega_i.item())

        return {
            'Delta': Delta,
            'Gamma': Gamma,
            'Vega': Vega
        }

    def asian_option_greeks(self, S0, K, T, r, sigma, num_paths=100000, num_steps=50, seed=42):
        """
        Calculate Greeks for Asian options using Malliavin calculus.

        .. warning::
            BIASED / UNDER REVIEW (2026-10-29): old weight construction
            (see the class docstring); left unchanged pending an independent
            verification.

        Args:
            S0 (float): Initial stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            num_paths (int): Number of Monte Carlo paths
            num_steps (int): Number of time steps in the simulation
            seed (int): Random seed for reproducibility

        Returns:
            dict: Dictionary containing Delta, Gamma, and Vega
        """
        torch.manual_seed(seed)

        dt = T / num_steps
        S = torch.zeros((num_paths, num_steps+1), device=self.device)
        S[:, 0] = S0
        z = torch.randn((num_paths, num_steps), device=self.device)

        # Simulate stock price paths
        for i in range(num_steps):
            S[:, i+1] = S[:, i] * torch.exp((r - 0.5 * sigma**2) * dt + sigma * torch.sqrt(dt) * z[:, i])

        # Calculate average price and payoff
        S_mean = torch.mean(S[:, 1:], dim=1)
        payoff = torch.relu(S_mean - K)
        discount_factor = torch.exp(-r * T)

        # Malliavin derivatives
        Dt_S = S[:, 1:] * z / torch.sqrt(dt)
        Dt_S_mean = torch.mean(Dt_S, dim=1)
        D2t_S = Dt_S * (z / torch.sqrt(dt) - sigma * torch.sqrt(dt))
        D2t_S_mean = torch.mean(D2t_S, dim=1)
        D_sigma_S = S[:, 1:] * (z * torch.sqrt(dt) - sigma * dt)
        D_sigma_S_mean = torch.mean(D_sigma_S, dim=1)

        # Calculate Greeks
        Delta = discount_factor * torch.mean(payoff * Dt_S_mean / (sigma * S0))
        Gamma = discount_factor * torch.mean(payoff * D2t_S_mean / (sigma**2 * S0**2))
        Vega = discount_factor * torch.mean(payoff * D_sigma_S_mean / sigma)

        return {
            'Delta': Delta.item(),
            'Gamma': Gamma.item(),
            'Vega': Vega.item()
        }


Greeks = MalliavinGreeks
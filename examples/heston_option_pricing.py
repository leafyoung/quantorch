from torchquantlib import Heston


heston_model = Heston(
    kappa_init=2.0,
    theta_init=0.04,
    sigma_v_init=0.3,
    rho_init=-0.7,
    v0_init=0.04,
    mu_init=0.05,
)

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
option_type = "call"

option_price = heston_model.option_price(S0, K, T, r, option_type, N=100000, steps=200)
print(f"The {option_type} option price under the Heston model is: {option_price:.4f}")

import math

import torch

from torchquantlib.core.asset_pricing.option.black_scholes_merton import black_scholes_merton
from torchquantlib.core.asset_pricing.option.vanilla_analytics import barrier_price_analytic, binary_price

# Golden values verified against QuantLib (AnalyticBarrierEngine on
# BarrierOption; CashOrNothingPayoff + AnalyticEuropeanEngine) and tff
# float64 (cross-check harness, 2026-10).
R, Q = 0.05, 0.02


def test_barrier_golden_values():
    # down-out B=85 K=100 call vol=20% T=1, S=100
    v = barrier_price_analytic(0.20, 100.0, 1.0, 100.0, 85.0, 0.0, R, Q, True, True, True)
    assert abs(v - 8.759460458735848) < 1e-10
    # up-out B=125 K=100 put vol=25% T=1
    v = barrier_price_analytic(0.25, 100.0, 1.0, 100.0, 125.0, 0.0, R, Q, False, True, False)
    assert abs(v - 7.902600854338467) < 1e-10
    # down-in B=85 K=110 call vol=25% T=1
    v = barrier_price_analytic(0.25, 110.0, 1.0, 100.0, 85.0, 0.0, R, Q, True, False, True)
    assert abs(v - 0.5479573225762513) < 1e-10
    # up-in B=125 K=90 put vol=20% T=1
    v = barrier_price_analytic(0.20, 90.0, 1.0, 100.0, 125.0, 0.0, R, Q, False, False, False)
    assert abs(v - 0.013422992543647733) < 1e-12


def test_barrier_extra_ql_grid():
    # down-out call, K > B, vol=30%
    v = barrier_price_analytic(0.30, 95.0, 1.0, 100.0, 90.0, 0.0, R, Q, True, True, True)
    assert abs(v - 9.711137714182236) < 1e-10
    # up-in call, deep OTM K=140
    v = barrier_price_analytic(0.25, 140.0, 1.0, 100.0, 120.0, 0.0, R, Q, False, False, True)
    assert abs(v - 1.5111745977832367) < 1e-11
    # up-out put, K > B
    v = barrier_price_analytic(0.25, 130.0, 1.0, 100.0, 125.0, 0.0, R, Q, False, True, False)
    assert abs(v - 24.5605437578551) < 1e-10
    # down-in put, K < B, T=2, vol=30%
    v = barrier_price_analytic(0.30, 75.0, 2.0, 100.0, 85.0, 0.0, R, Q, True, False, False)
    assert abs(v - 3.925672050043495) < 1e-11
    # down-out put, K < B: barrier below both spot and strike region -> 0
    v = barrier_price_analytic(0.25, 80.0, 1.0, 100.0, 85.0, 0.0, R, Q, True, True, False)
    assert abs(v) < 1e-12
    # up-in call, S=90, T=2, vol=40%
    v = barrier_price_analytic(0.40, 110.0, 2.0, 90.0, 105.0, 0.0, R, Q, False, False, True)
    assert abs(v - 14.800183620042578) < 1e-11


def test_barrier_in_plus_out_equals_vanilla():
    # down barrier, call: knock-in + knock-out = BSM vanilla (rebate = 0)
    pin = barrier_price_analytic(0.20, 100.0, 1.0, 100.0, 80.0, 0.0, R, Q, True, False, True)
    pout = barrier_price_analytic(0.20, 100.0, 1.0, 100.0, 80.0, 0.0, R, Q, True, True, True)
    vanilla = float(
        black_scholes_merton(
            "call", "european",
            torch.tensor(100.0, dtype=torch.float64),
            torch.tensor(100.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(0.20, dtype=torch.float64),
            torch.tensor(R, dtype=torch.float64),
            torch.tensor(Q, dtype=torch.float64),
        )
    )
    assert abs(pin + pout - vanilla) < 1e-10

    # up barrier, put, T=2
    pin = barrier_price_analytic(0.25, 100.0, 2.0, 100.0, 120.0, 0.0, R, Q, False, False, False)
    pout = barrier_price_analytic(0.25, 100.0, 2.0, 100.0, 120.0, 0.0, R, Q, False, True, False)
    vanilla = float(
        black_scholes_merton(
            "put", "european",
            torch.tensor(100.0, dtype=torch.float64),
            torch.tensor(100.0, dtype=torch.float64),
            torch.tensor(2.0, dtype=torch.float64),
            torch.tensor(0.25, dtype=torch.float64),
            torch.tensor(R, dtype=torch.float64),
            torch.tensor(Q, dtype=torch.float64),
        )
    )
    assert abs(pin + pout - vanilla) < 1e-10


def test_barrier_with_rebate():
    # down-out call, rebate 3 paid at hit (tff semantics)
    v = barrier_price_analytic(0.25, 100.0, 1.0, 100.0, 85.0, 3.0, R, Q, True, True, True)
    assert abs(v - 11.475189709298022) < 1e-10
    # down-in call, rebate 1.5 paid at expiry if never knocked in
    # (also QuantLib-exact)
    v = barrier_price_analytic(0.25, 100.0, 1.0, 100.0, 85.0, 1.5, R, Q, True, False, True)
    assert abs(v - 1.8890581732854028) < 1e-11


def test_binary_golden_values():
    # ATM call, K=100, T=1, vol=20%
    v = binary_price(0.20, 100.0, 1.0, 100.0, R, Q, True)
    assert abs(v - 0.49458109105322323) < 1e-12
    # put, K=110, T=2, vol=30%
    v = binary_price(0.30, 110.0, 2.0, 100.0, R, Q, False)
    assert abs(v - 0.557506659561213) < 1e-12


def test_binary_extra_ql_grid():
    # deep ITM call, K=70
    v = binary_price(0.20, 70.0, 1.0, 100.0, R, Q, True)
    assert abs(v - 0.9194836337641996) < 1e-12
    # deep OTM call, K=140
    v = binary_price(0.20, 140.0, 1.0, 100.0, R, Q, True)
    assert abs(v - 0.04879969669114354) < 1e-12
    # put, K=70, T=2, vol=30%
    v = binary_price(0.30, 70.0, 2.0, 100.0, R, Q, False)
    assert abs(v - 0.1996576342569846) < 1e-12
    # ATM call, vol=50%
    v = binary_price(0.50, 100.0, 1.0, 100.0, R, Q, True)
    assert abs(v - 0.4039439177288213) < 1e-12


def test_binary_put_call_sum_and_zero_vol():
    c = binary_price(0.25, 105.0, 1.5, 100.0, R, Q, True)
    p = binary_price(0.25, 105.0, 1.5, 100.0, R, Q, False)
    assert abs(c + p - math.exp(-R * 1.5)) < 1e-14
    # forward 100*e^{0.03} > 90 -> call pays DF; < 110 -> put pays DF
    assert abs(binary_price(0.0, 90.0, 1.0, 100.0, R, Q, True) - math.exp(-R)) < 1e-14
    assert abs(binary_price(0.0, 110.0, 1.0, 100.0, R, Q, False) - math.exp(-R)) < 1e-14
    assert abs(binary_price(0.0, 110.0, 1.0, 100.0, R, Q, True)) < 1e-14


def test_accepts_zero_dim_tensors():
    # The module mirrors the tensor-pricer call style: 0-dim tensors are
    # converted via float() and the math runs in f64 regardless of the
    # incoming tensor dtype.
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    v = barrier_price_analytic(
        t(0.20), t(100.0), t(1.0), t(100.0), t(85.0), t(0.0), t(R), t(Q), True, True, True
    )
    assert abs(v - 8.759460458735848) < 1e-12
    b = binary_price(t(0.20), t(100.0), t(1.0), t(100.0), t(R), t(Q), True)
    assert abs(b - 0.49458109105322323) < 1e-12
    # float32 tensors: inputs quantize to f32 (~1e-7 relative), math stays f64
    t32 = lambda x: torch.tensor(x)
    v32 = barrier_price_analytic(
        t32(0.20), t32(100.0), t32(1.0), t32(100.0), t32(85.0), t32(0.0), t32(R), t32(Q),
        True, True, True,
    )
    assert abs(v32 - 8.759460458735848) < 1e-6

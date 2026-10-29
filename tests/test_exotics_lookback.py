import pytest
import torch
from torchquantlib.core.asset_pricing.option.exotics import lookback_option

# QuantLib 1.44 goldens: ContinuousFixedLookbackOption +
# AnalyticContinuousFixedLookbackEngine (Act365, yearFraction exactly 1.0),
# S=100 K=100 r=5% q=0 vol=20% T=1.
QL_CONTINUOUS_FIXED_CALL = 19.16762525733232
QL_CONTINUOUS_FIXED_PUT = 12.339744687432269


@pytest.fixture
def setup_tensors():
    return {
        'spot': torch.tensor(100.0),
        'strike': torch.tensor(100.0),
        'expiry': torch.tensor(1.0),
        'volatility': torch.tensor(0.2),
        'rate': torch.tensor(0.05),
    }


def test_lookback_continuous_call_matches_ql_golden(setup_tensors):
    # 200k paths, 252 daily steps, seed documented: torch.manual_seed(1244).
    # The BGK-corrected continuous price lands within ~1e-2 relative of the
    # golden at this seed (MC noise at 200k paths is of the same order);
    # band 3e-2. The uncorrected discrete estimator sits ~4.6% low, i.e. the
    # Broadie-Glasserman-Kou discrete-monitoring bias.
    torch.manual_seed(1244)
    price = lookback_option('call', num_paths=200_000, strike_type='fixed',
                            monitoring='continuous', **setup_tensors)
    assert abs(price.item() - QL_CONTINUOUS_FIXED_CALL) / QL_CONTINUOUS_FIXED_CALL <= 3e-2


def test_lookback_continuous_put_matches_ql_golden(setup_tensors):
    # 200k paths, seed documented: torch.manual_seed(1245). Verifies the
    # minimum-deflation direction of the BGK correction
    # (min_cont ~ min_disc * e^{-beta*sigma*sqrt(dt)}).
    torch.manual_seed(1245)
    price = lookback_option('put', num_paths=200_000, strike_type='fixed',
                            monitoring='continuous', **setup_tensors)
    assert abs(price.item() - QL_CONTINUOUS_FIXED_PUT) / QL_CONTINUOUS_FIXED_PUT <= 3e-2


def test_lookback_continuous_exceeds_discrete(setup_tensors):
    # Same seed: the BGK-adjusted (continuous) price must sit above the
    # discrete price by ~4.6% for these parameters — the extremum factor
    # e^{beta*sigma*sqrt(dt)} ~ 1.0074 is leveraged through the payoff,
    # (f*E[M] - K) / (E[M] - K) with E[M] ~ 119, K = 100.
    torch.manual_seed(1244)
    discrete = lookback_option('call', num_paths=50_000, strike_type='fixed', **setup_tensors)
    torch.manual_seed(1244)
    continuous = lookback_option('call', num_paths=50_000, strike_type='fixed',
                                 monitoring='continuous', **setup_tensors)
    ratio = continuous.item() / discrete.item()
    assert 1.03 < ratio < 1.06


def test_lookback_invalid_monitoring(setup_tensors):
    with pytest.raises(ValueError):
        lookback_option('call', num_paths=100, strike_type='fixed',
                        monitoring='weekly', **setup_tensors)

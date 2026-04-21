import pytest
import torch
from torchquantlib.core.asset_pricing.option.asian_option import asian_option


@pytest.fixture
def setup_tensors():
    return {
        'spot': torch.tensor(100.0),
        'strike': torch.tensor(100.0),
        'expiry': torch.tensor(1.0),
        'volatility': torch.tensor(0.2),
        'rate': torch.tensor(0.05),
        'steps': 100
    }


def _expected_discrete_average(spot: torch.Tensor, expiry: torch.Tensor, rate: torch.Tensor, steps: int) -> torch.Tensor:
    times = expiry * torch.arange(1, steps + 1, dtype=spot.dtype) / steps
    return spot * torch.exp(rate * times).mean()


def test_asian_call_option(setup_tensors):
    price = asian_option('call', **setup_tensors)
    assert price > 0
    assert price <= setup_tensors['spot']


def test_asian_put_option(setup_tensors):
    price = asian_option('put', **setup_tensors)
    assert price > 0
    assert price <= setup_tensors['strike']


def test_asian_call_vs_put(setup_tensors):
    call_price = asian_option('call', **setup_tensors)
    put_price = asian_option('put', **setup_tensors)
    assert call_price != put_price


def test_asian_option_increase_volatility(setup_tensors):
    low_vol_price = asian_option('call', **setup_tensors)
    setup_tensors['volatility'] *= 2
    high_vol_price = asian_option('call', **setup_tensors)
    assert high_vol_price > low_vol_price


def test_asian_option_increase_expiry(setup_tensors):
    short_expiry_price = asian_option('call', **setup_tensors)
    setup_tensors['expiry'] *= 2
    long_expiry_price = asian_option('call', **setup_tensors)
    assert long_expiry_price != short_expiry_price


def test_asian_option_increase_steps(setup_tensors):
    low_steps_price = asian_option('call', **setup_tensors)
    setup_tensors['steps'] *= 2
    high_steps_price = asian_option('call', **setup_tensors)
    assert torch.isclose(low_steps_price, high_steps_price, rtol=5e-2)


def test_asian_option_zero_volatility(setup_tensors):
    setup_tensors['volatility'] = torch.tensor(0.0)
    price = asian_option('call', **setup_tensors)
    assert price >= 0


def test_asian_option_invalid_type():
    with pytest.raises(ValueError):
        asian_option('invalid', spot=torch.tensor(100.0), strike=torch.tensor(100.0),
                     expiry=torch.tensor(1.0), volatility=torch.tensor(0.2),
                     rate=torch.tensor(0.05), steps=100)


def test_asian_option_put_call_parity_on_average(setup_tensors):
    call_price = asian_option('call', **setup_tensors)
    put_price = asian_option('put', **setup_tensors)
    expected_average = _expected_discrete_average(
        setup_tensors['spot'],
        setup_tensors['expiry'],
        setup_tensors['rate'],
        setup_tensors['steps'],
    )
    rhs = torch.exp(-setup_tensors['rate'] * setup_tensors['expiry']) * (expected_average - setup_tensors['strike'])
    assert torch.isclose(call_price - put_price, rhs, atol=5e-3, rtol=1e-3)


def test_geometric_asian_call_is_not_more_expensive_than_arithmetic(setup_tensors):
    arithmetic_price = asian_option('call', **setup_tensors, average_type='arithmetic')
    geometric_price = asian_option('call', **setup_tensors, average_type='geometric')
    assert geometric_price <= arithmetic_price

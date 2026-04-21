import pytest
import torch
from torchquantlib.core.asset_pricing.bond.bond_pricer import (
    zero_coupon_bond,
    coupon_bond,
    callable_bond,
    callable_bond_lattice,
    callable_convertible_bond_lattice,
    putable_bond,
    putable_convertible_bond_lattice,
    putable_bond_lattice,
    convertible_bond,
    convertible_bond_lattice,
    stochastic_rate_bond
)


@pytest.fixture
def setup_tensors():
    return {
        'face_value': torch.tensor(1000.0),
        'rate': torch.tensor(0.05),
        'maturity': torch.tensor(5.0),
        'coupon_rate': torch.tensor(0.04),
        'periods': torch.tensor(5),
        'call_price': torch.tensor(1050.0),
        'call_period': torch.tensor(3),
        'put_price': torch.tensor(950.0),
        'put_period': torch.tensor(3),
        'conversion_ratio': torch.tensor(10.0),
        'conversion_price': torch.tensor(110.0),
        'stochastic_rates': torch.tensor([0.04, 0.045, 0.05, 0.055, 0.06])
    }


def test_zero_coupon_bond(setup_tensors):
    price = zero_coupon_bond(setup_tensors['face_value'], setup_tensors['rate'], setup_tensors['maturity'])
    expected_price = 1000 / (1 + 0.05)**5
    assert torch.isclose(price, torch.tensor(expected_price), atol=1e-4)


def test_coupon_bond(setup_tensors):
    price = coupon_bond(
        setup_tensors['face_value'],
        setup_tensors['coupon_rate'],
        setup_tensors['rate'],
        setup_tensors['periods'],
    )
    expected_price = sum(40 / (1.05) ** t for t in range(1, 6)) + 1000 / (1.05) ** 5
    assert torch.isclose(price, torch.tensor(expected_price), atol=1e-4)
    assert price < setup_tensors['face_value']


def test_callable_bond(setup_tensors):
    straight_price = coupon_bond(
        setup_tensors['face_value'],
        setup_tensors['coupon_rate'],
        setup_tensors['rate'],
        setup_tensors['periods'],
    )
    price = callable_bond(
        setup_tensors['face_value'],
        setup_tensors['coupon_rate'],
        setup_tensors['rate'],
        setup_tensors['periods'],
        setup_tensors['call_price'],
        setup_tensors['call_period'],
    )
    assert price <= straight_price


def test_putable_bond(setup_tensors):
    straight_price = coupon_bond(
        setup_tensors['face_value'],
        setup_tensors['coupon_rate'],
        setup_tensors['rate'],
        setup_tensors['periods'],
    )
    price = putable_bond(
        setup_tensors['face_value'],
        setup_tensors['coupon_rate'],
        setup_tensors['rate'],
        setup_tensors['periods'],
        setup_tensors['put_price'],
        setup_tensors['put_period'],
    )
    assert price >= straight_price


def test_convertible_bond(setup_tensors):
    straight_price = coupon_bond(
        setup_tensors['face_value'],
        setup_tensors['coupon_rate'],
        setup_tensors['rate'],
        setup_tensors['periods'],
    )
    price = convertible_bond(
        setup_tensors['face_value'],
        setup_tensors['coupon_rate'],
        setup_tensors['rate'],
        setup_tensors['periods'],
        setup_tensors['conversion_ratio'],
        setup_tensors['conversion_price'],
    )
    conversion_value = setup_tensors['conversion_ratio'] * setup_tensors['conversion_price']
    assert price >= straight_price
    assert price >= conversion_value


def test_stochastic_rate_bond(setup_tensors):
    price = stochastic_rate_bond(
        setup_tensors['face_value'],
        setup_tensors['coupon_rate'],
        setup_tensors['stochastic_rates'],
        setup_tensors['periods'],
    )
    assert price > 0
    assert price < setup_tensors['face_value'] * (1 + setup_tensors['coupon_rate'] * setup_tensors['periods'])


def test_callable_bond_lattice_matches_zero_vol_wrapper(setup_tensors):
    lattice_price = callable_bond_lattice(
        setup_tensors['face_value'],
        setup_tensors['coupon_rate'],
        setup_tensors['rate'],
        setup_tensors['periods'],
        setup_tensors['call_price'],
        setup_tensors['call_period'],
        rate_volatility=torch.tensor(0.0),
    )
    wrapper_price = callable_bond(
        setup_tensors['face_value'],
        setup_tensors['coupon_rate'],
        setup_tensors['rate'],
        setup_tensors['periods'],
        setup_tensors['call_price'],
        setup_tensors['call_period'],
    )
    assert torch.isclose(lattice_price, wrapper_price, atol=1e-6)


def test_putable_bond_lattice_matches_zero_vol_wrapper(setup_tensors):
    lattice_price = putable_bond_lattice(
        setup_tensors['face_value'],
        setup_tensors['coupon_rate'],
        setup_tensors['rate'],
        setup_tensors['periods'],
        setup_tensors['put_price'],
        setup_tensors['put_period'],
        rate_volatility=torch.tensor(0.0),
    )
    wrapper_price = putable_bond(
        setup_tensors['face_value'],
        setup_tensors['coupon_rate'],
        setup_tensors['rate'],
        setup_tensors['periods'],
        setup_tensors['put_price'],
        setup_tensors['put_period'],
    )
    assert torch.isclose(lattice_price, wrapper_price, atol=1e-6)


def test_convertible_bond_lattice_is_at_least_conversion_value():
    price = convertible_bond_lattice(
        face_value=torch.tensor(1000.0),
        coupon_rate=torch.tensor(0.03),
        rate=torch.tensor(0.04),
        periods=torch.tensor(5),
        conversion_ratio=torch.tensor(8.0),
        stock_price=torch.tensor(120.0),
        stock_volatility=torch.tensor(0.25),
    )
    assert price >= torch.tensor(8.0 * 120.0)


def test_callable_convertible_bond_lattice_is_not_more_valuable_than_noncallable_version():
    callable_price = callable_convertible_bond_lattice(
        face_value=torch.tensor(1000.0),
        coupon_rate=torch.tensor(0.03),
        rate=torch.tensor(0.04),
        periods=torch.tensor(5),
        conversion_ratio=torch.tensor(8.0),
        stock_price=torch.tensor(120.0),
        stock_volatility=torch.tensor(0.25),
        call_price=torch.tensor(1100.0),
        call_period=torch.tensor(3),
    )
    noncallable_price = convertible_bond_lattice(
        face_value=torch.tensor(1000.0),
        coupon_rate=torch.tensor(0.03),
        rate=torch.tensor(0.04),
        periods=torch.tensor(5),
        conversion_ratio=torch.tensor(8.0),
        stock_price=torch.tensor(120.0),
        stock_volatility=torch.tensor(0.25),
    )
    assert callable_price <= noncallable_price


def test_putable_convertible_bond_lattice_is_not_less_valuable_than_nonputable_version():
    putable_price = putable_convertible_bond_lattice(
        face_value=torch.tensor(1000.0),
        coupon_rate=torch.tensor(0.03),
        rate=torch.tensor(0.04),
        periods=torch.tensor(5),
        conversion_ratio=torch.tensor(8.0),
        stock_price=torch.tensor(120.0),
        stock_volatility=torch.tensor(0.25),
        put_price=torch.tensor(1050.0),
        put_period=torch.tensor(3),
    )
    nonputable_price = convertible_bond_lattice(
        face_value=torch.tensor(1000.0),
        coupon_rate=torch.tensor(0.03),
        rate=torch.tensor(0.04),
        periods=torch.tensor(5),
        conversion_ratio=torch.tensor(8.0),
        stock_price=torch.tensor(120.0),
        stock_volatility=torch.tensor(0.25),
    )
    assert putable_price >= nonputable_price

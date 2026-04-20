import torch

from torchquantlib import callable_bond, convertible_bond, coupon_bond, stochastic_rate_bond, zero_coupon_bond


face_value = torch.tensor(1000.0)
rate = torch.tensor(0.05)
maturity = torch.tensor(5)
coupon_rate = torch.tensor(0.06)
periods = torch.tensor(10)

zcb_price = zero_coupon_bond(face_value, rate, maturity)
print(f"Zero Coupon Bond Price: {zcb_price}")

cb_price = coupon_bond(face_value, coupon_rate, rate, periods)
print(f"Coupon Bond Price: {cb_price}")

rate_tensor = torch.tensor([0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095])
srb_price = stochastic_rate_bond(face_value, coupon_rate, rate_tensor, periods)
print(f"Stochastic Rate Bond Price: {srb_price}")

call_price = torch.tensor(1050.0)
call_period = torch.tensor(5)
callable_price = callable_bond(face_value, coupon_rate, rate, periods, call_price, call_period)
print(f"Callable Bond Price: {callable_price}")

conversion_ratio = torch.tensor(1.1)
conversion_price = torch.tensor(100.0)
convertible_price = convertible_bond(face_value, coupon_rate, rate, periods, conversion_ratio, conversion_price)
print(f"Convertible Bond Price: {convertible_price}")

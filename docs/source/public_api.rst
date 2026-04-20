Public API
==========

TorchQuant's supported import surface is intentionally explicit. Prefer
package-level imports from ``torchquantlib`` for stable public symbols.
Deep imports remain available for implementation work and module-level docs, but
are discouraged in downstream code when a symbol is already re-exported.

Preferred pattern
-----------------

.. code-block:: python

   from torchquantlib import (
       AsianOption,
       CalibrationConfig,
       Heston,
       MalliavinGreeks,
       ModelCalibrator,
       american_option,
       asian_option,
       bermudan_option,
       black_scholes_merton,
       bootstrap_yield_curve,
       calculate_es,
       calculate_var,
       future_pricer,
       implied_volatility,
       merton_model,
       reduced_form_model,
   )

Available public groups
-----------------------

Pricing
^^^^^^^

- ``american_option(option_type, spot, strike, expiry, volatility, rate, steps)``
- ``AsianOption(...)``
- ``asian_option(option_type, spot, strike, expiry, volatility, rate, steps, ...)``
- ``bermudan_option(option_type, spot, strike, expiry, volatility, rate, steps, exercise_dates)``
- ``black_scholes_merton(option_type, option_style, spot, strike, expiry, volatility, rate, dividend)``
- ``future_pricer(spot, domestic_rate, foreign_rate, expiry)``
- ``fixed_income_forward(face_value, rate, time_to_maturity, forward_rate)``
- bond pricing helpers such as ``zero_coupon_bond`` and ``coupon_bond``
- ``implied_volatility(option_price, spot, strike, expiry, rate, is_call=True)``

Risk
^^^^

- ``MalliavinGreeks``
- ``calculate_var``
- ``calculate_es``
- ``scenario_analysis``
- ``stress_test``
- ``merton_model``
- ``reduced_form_model``
- ``calculate_cva`` / ``calculate_dva`` / ``calculate_fva`` / ``calculate_mva``

Models and utilities
^^^^^^^^^^^^^^^^^^^^

- ``Heston``, ``SABR``, ``Vasicek``, ``CIR``, ``BlackKarasinski``, ``HullWhiteModel``
- ``DupireLocalVol``
- ``binomial_tree``
- ``ModelCalibrator`` and ``CalibrationConfig``
- ``Seq2SeqPDESolver``
- ``bootstrap_yield_curve`` and ``nelson_siegel_yield_curve``

Guidance
--------

- Prefer ``from torchquantlib import ...`` for public symbols.
- Use deeper imports only when the symbol is not intentionally re-exported.
- Treat deep module paths as implementation-oriented, not the primary public API.

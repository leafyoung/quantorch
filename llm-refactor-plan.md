# LLM Refactor Plan

This repo already has a useful feature split (`core`, `models`, `calibration`, `utils`, `examples`, `tests`). The main refactor need is not a total rewrite. The biggest gains come from making the package surface consistent, centralizing repeated state handling, and replacing implicit behavior with explicit, discoverable APIs.

## Status

Completed in this pass:
- created this audit document
- refactored the shared stochastic model infrastructure
- added explicit model/calibration package exports
- refactored `ModelCalibrator` around explicit config and structured progress events
- added compatibility modules for missing Asian option and Malliavin Greeks API paths
- moved embedded demo entrypoints out of library modules into `examples/`
- reconciled option API signature drift so American option examples/tests match the public function signature
- added `pyproject.toml` packaging metadata and aligned install docs around editable installs and extras
- added a dedicated public API docs page and cleaned major README/Sphinx usage examples

Still outstanding:
- full runtime verification remains blocked in this environment because `torch` is not installed
- some broader non-option documentation can still be tightened further, but the main public API and packaging drift have been reconciled

## High

### 1. Make the package surface internally consistent
Several parts of the repo disagree about what the public API actually is.

- Multiple model files import `from models.stochastic_model import StochasticModel`, which is not a package-safe import path.
- `tests/`, `examples/`, and `README.md` reference modules that do not exist, including:
  - `torchquantlib.core.asset_pricing.option.asian_option`
  - `torchquantlib.core.risk.greeks.malliavin`
- Many `__init__.py` files are effectively empty, so users and future agents have no clear supported import surface.

**Why this matters:** future LLMs cannot reliably reason about the package if docs, tests, and imports point to different APIs.

**Refactor direction:**
- Fix package-relative imports.
- Add explicit re-exports in package `__init__.py` files for the supported surface.
- Either add the missing modules or update tests/docs/examples so they all target the same API.

### 2. Centralize model parameter and device handling
Device selection, parameter creation, tensor movement, and constraint application are repeated across stochastic models, and one model (`LMM`) uses `self.device` before base initialization.

There is also a duplicate mini `StochasticModel` inside `torchquantlib/models/local_volatility/dupire_local_volatility.py`.

**Why this matters:** this is the most important maintainability hotspot. The same low-level state logic is duplicated in several files, which makes future fixes error-prone.

**Refactor direction:**
- Expand `torchquantlib/models/stochastic_model.py` into the single source of truth for:
  - device resolution
  - moving model parameters between devices
  - parameter constraint application
  - tensor creation helpers for subclasses
- Remove the duplicate base class from `dupire_local_volatility.py`.
- Keep subclass code focused on simulation logic only.

### 3. Replace print-driven workflows with explicit calibration state and structured logs
`ModelCalibrator` mixes configuration, mutable runtime state, and progress printing in one class.

**Why this matters:** calibration is a workflow boundary. It should be easy to inspect, rerun, and extend without guessing which fields are configuration and which fields are runtime state.

**Refactor direction:**
- Introduce an explicit config object for calibration inputs.
- Emit structured progress events instead of ad hoc print lines.
- Keep the top-level calibration loop linear and easy to scan.

## Medium

### 4. Move executable examples out of library modules
Some library modules contain demo code or training entrypoints (for example `letsberational.py` and `models/stochastic_volatility/gans.py`).

**Why this matters:** library files should define reusable behavior. Example execution belongs in `examples/` or dedicated scripts.

**Refactor direction:**
- Remove or relocate demo code from importable library modules.
- Keep `examples/` as the only place for runnable scripts.

### 5. Normalize naming and file consistency
There are avoidable naming mismatches that make searching harder.

Examples:
- `tests/test_binomial_tree.py` and `tests/test_expected_shortfall.py` were previously misspelled and have now been normalized.
- `HullWhiteModel` is the odd naming pattern among otherwise short model names.

**Why this matters:** small inconsistencies create friction for code search, scripted refactors, and future agent reasoning.

**Refactor direction:**
- Rename misspelled files.
- Add compatibility aliases where needed before renaming public classes.

### 6. Separate pure library code from repo packaging and docs drift
The repo still uses `setup.py` + `requirements.txt`, while examples and README contain outdated imports and snippets.

**Why this matters:** packaging and documentation should point to the same reality as the code.

**Refactor direction:**
- Consider migrating to `pyproject.toml`.
- Update README/examples only after the public API is stabilized.

## Lower

### 7. Add explicit public exports
Use `__all__` and package re-exports for the stable API.

**Why this matters:** empty package files force readers to inspect the full tree instead of learning the intended entrypoints quickly.

### 8. Trim duplicate or low-value files after API stabilization
Possible cleanup candidates:
- duplicate base-class logic in `dupire_local_volatility.py`
- executable demo code embedded in library modules
- any docs pages that only repeat stale example code after the API is cleaned up

Only delete files after the replacement API or example path is clear.

### 9. Add repo-wide verification guidance
There is no existing repo instruction file explaining the expected validation flow.

**Refactor direction:**
- Once dependencies are stable, document the default verification steps for this repo (for example: install deps, run targeted tests, then full test suite).

## Recommended order

1. Stabilize the package surface and shared model base.
2. Refactor `ModelCalibrator` around explicit config and structured progress events.
3. Reconcile docs/examples/tests with the actual public API.
4. Clean up naming drift and move executable demos out of library modules.

## Good candidates for deletion or consolidation

- Delete the duplicate `StochasticModel` defined inside `torchquantlib/models/local_volatility/dupire_local_volatility.py` after switching it to the shared base class.
- Move demo code out of `torchquantlib/core/asset_pricing/letsberational.py` and `torchquantlib/models/stochastic_volatility/gans.py`.
- Remove stale doc/example references to missing modules once the intended API is chosen.

## First discrete refactor unit

The best first implementation unit is:

**Unify stochastic model infrastructure**

That means:
- strengthen `torchquantlib/models/stochastic_model.py`
- switch all model subclasses to the shared infrastructure
- fix package-safe imports
- remove duplicated device/base-class logic
- add explicit package exports for the refactored model surface

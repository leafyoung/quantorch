import torch

from torchquantlib.calibration import CalibrationInstrument, MarketCalibrator


class MockSurfaceModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.params = {
            "level": torch.tensor(0.5, requires_grad=True),
            "slope": torch.tensor(0.0, requires_grad=True),
        }

    def quote(self, strike: float) -> torch.Tensor:
        strike_tensor = torch.tensor(strike, dtype=torch.float32)
        return self.params["level"] + self.params["slope"] * strike_tensor

    def _apply_constraints(self):
        pass


def test_market_calibrator_moves_parameters_toward_targets():
    model = MockSurfaceModel()
    instruments = [
        CalibrationInstrument(name="k1", target=1.0, pricer=lambda m: m.quote(1.0), weight=1.0),
        CalibrationInstrument(name="k2", target=1.5, pricer=lambda m: m.quote(2.0), weight=1.0),
    ]
    calibrator = MarketCalibrator(model, instruments, lr=0.1)

    initial_level = model.params["level"].item()
    result = calibrator.calibrate(num_epochs=200, verbose=False)

    assert result.final_loss < 1e-4
    assert model.params["level"].item() != initial_level
    assert abs(model.params["level"].item() - 0.5) < 0.05
    assert abs(model.params["slope"].item() - 0.5) < 0.05


def test_market_calibrator_returns_residual_snapshot():
    model = MockSurfaceModel()
    instruments = [
        CalibrationInstrument(name="k1", target=1.0, pricer=lambda m: m.quote(1.0), weight=1.0, bid_ask_spread=0.1),
    ]
    calibrator = MarketCalibrator(model, instruments, lr=0.05, normalize_by_spread=True)
    result = calibrator.calibrate(num_epochs=50, verbose=False)

    assert "k1" in result.residuals
    assert isinstance(result.residuals["k1"], float)
    assert result.num_epochs == 50


def test_market_calibrator_regularization_tracks_objective_decomposition():
    model = MockSurfaceModel()
    instruments = [
        CalibrationInstrument(name="k1", target=1.0, pricer=lambda m: m.quote(1.0), weight=1.0),
    ]
    calibrator = MarketCalibrator(
        model,
        instruments,
        lr=0.05,
        regularization_strength=10.0,
        parameter_priors={"level": 0.5, "slope": 0.0},
    )
    result = calibrator.calibrate(num_epochs=25, verbose=False)

    assert "data_loss" in result.objective_decomposition
    assert "regularization_loss" in result.objective_decomposition
    assert result.objective_decomposition["regularization_loss"] >= 0

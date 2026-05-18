import pytest
from utils.interpretation import (
    CalibrationTier,
    _uninformative_brier,
    calibration_tier,
    calibration_tiers_from_results,
    likelihood_label
)

# likelihood_label
class TestLikelihoodLabel:
    def test_very_unlikely_at_zero(self):
        assert likelihood_label(0.0) == "Very unlikely"

    def test_very_unlikely_just_below_threshold(self):
        assert likelihood_label(0.09) == "Very unlikely"

    def test_unlikely_at_threshold(self):
        assert likelihood_label(0.10) == "Unlikely"

    def test_about_as_likely_as_not_at_threshold(self):
        assert likelihood_label(0.33) == "About as likely as not"

    def test_likely_at_threshold(self):
        assert likelihood_label(0.66) == "Likely"

    def test_very_likely_at_threshold(self):
        assert likelihood_label(0.90) == "Very likely"

    def test_very_likely_at_one(self):
        assert likelihood_label(1.0) == "Very likely"

    def test_rejects_negative_probability(self):
        with pytest.raises(ValueError):
            likelihood_label(-0.1)

    def test_rejects_probability_above_one(self):
        with pytest.raises(ValueError):
            likelihood_label(1.5)


# _uninformative_brier
class TestUninformativeBrier:
    def test_maximum_at_half(self):
        assert _uninformative_brier(0.5) == 0.25

    def test_zero_at_endpoints(self):
        assert _uninformative_brier(0.0) == 0.0
        assert _uninformative_brier(1.0) == 0.0

    def test_realistic_class_baselines(self):
        # Informal Action Taken at ~83% base rate
        assert _uninformative_brier(0.83) == pytest.approx(0.1411, abs=1e-4)
        # Investigation Pursued at ~6% base rate
        assert _uninformative_brier(0.06) == pytest.approx(0.0564, abs=1e-4)


# calibration_tier
class TestCalibrationTier:
    def test_high_reliability_for_brier_well_below_baseline(self):
        # baseline = 0.83 * 0.17 = 0.1411
        # 50% of baseline = 0.0705
        tier = calibration_tier("Informal Action Taken", brier=0.05, base_rate=0.83)
        assert tier.tier == "high"
        assert tier.label == "High reliability"

    def test_moderate_reliability_between_half_and_full_baseline(self):
        # baseline = 0.83 * 0.17 = 0.1411
        # half baseline = 0.0705
        tier = calibration_tier("Informal Action Taken", brier=0.10, base_rate=0.83)
        assert tier.tier == "moderate"
        assert tier.label == "Moderate reliability"

    def test_lower_reliability_above_baseline(self):
        # baseline = 0.83 * 0.17 = 0.1411
        tier = calibration_tier("Informal Action Taken", brier=0.20, base_rate=0.83)
        assert tier.tier == "low"
        assert tier.label == "Lower reliability"

    def test_limited_reliability_override_for_investigation_pursued(self):
        tier = calibration_tier(
            "Investigation Pursued", brier=0.001, base_rate=0.06
        )
        assert tier.tier == "limited"
        assert tier.label == "Limited reliability"
        assert "rarely predicts" in tier.explanation.lower()

    def test_limited_override_applies_even_with_excellent_brier(self):
        very_low_brier_tier = calibration_tier(
            "Investigation Pursued", brier=0.0001, base_rate=0.06
        )
        assert very_low_brier_tier.tier == "limited"

    def test_rejects_invalid_brier(self):
        with pytest.raises(ValueError):
            calibration_tier("No Further Action", brier=1.5, base_rate=0.11)


# calibration_tiers_from_results
class TestCalibrationTiersFromResults:
    def test_returns_a_tier_per_class(self):
        classifier_results = {
            "brier_Informal Action Taken": 0.10,
            "brier_Investigation Pursued": 0.05,
            "brier_No Further Action": 0.08
        }
        base_rates = {
            "Informal Action Taken": 0.83,
            "Investigation Pursued": 0.06,
            "No Further Action": 0.11
        }
        tiers = calibration_tiers_from_results(
            classifier_results=classifier_results,
            base_rates=base_rates,
            class_labels=[
                "Informal Action Taken",
                "Investigation Pursued",
                "No Further Action"
            ],
        )
        assert set(tiers.keys()) == {
            "Informal Action Taken",
            "Investigation Pursued",
            "No Further Action"
        }
        assert all(isinstance(t, CalibrationTier) for t in tiers.values())

    def test_handles_missing_brier(self):
        classifier_results = {}
        base_rates = {"Informal Action Taken": 0.83}
        tiers = calibration_tiers_from_results(
            classifier_results=classifier_results,
            base_rates=base_rates,
            class_labels=["Informal Action Taken"]
        )
        assert tiers["Informal Action Taken"].label == "Reliability unknown"
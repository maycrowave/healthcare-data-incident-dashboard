from typing import Dict, NamedTuple

# Probability to plain text likelihood
_LIKELIHOODS = [
    (0.00, 0.10, "Very unlikely"),
    (0.10, 0.33, "Unlikely"),
    (0.33, 0.66, "About as likely as not"),
    (0.66, 0.90, "Likely"),
    (0.90, 1.01, "Very likely")
]

def likelihood_label(probability: float) -> str:
    """Return the IPCC plain text likelihood label for a probability"""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability out of range [0, 1]: {probability}")

    for lower, upper, label in _LIKELIHOODS:
        if lower <= probability < upper:
            return label
    # This should never happen because likelihoods cover [0, 1] but just in case of any rounding issues
    return "Very likely"

# Brier score & calibration tier
class CalibrationTier(NamedTuple):
    """A reliability assessment for a single class's predictions"""
    tier: str
    label: str
    explanation: str

def _uninformative_brier(base_rate: float) -> float:
    """
    The brier score of a model that always predicts a class's base rate, equals base_rate * (1-base_rate)
    """
    return base_rate * (1.0-base_rate)


def calibration_tier(class_label: str, brier: float, base_rate: float) -> CalibrationTier:
    """
    Return a calibration tier for a class given its test set brier score and its training base rate

    Tiers are tied to the uninformative baseline (brier of always predict base rate)
    A model substantially better than this baseline earns high, comparable earns moderate, worse earns low
    """
    # Some classes have special considerations that limit reliability regardless of brier score like investigation pursued
    if class_label in _LIMITED_RELIABILITY_CLASSES:
        return CalibrationTier(
            tier="limited",
            label="Limited reliability",
            explanation=_LIMITED_RELIABILITY_CLASSES[class_label],
        )

    if not 0.0 <= brier <= 1.0:
        raise ValueError(f"Brier out of range [0, 1]: {brier}")

    baseline = _uninformative_brier(base_rate)

    if brier < 0.5 * baseline:
        tier, label = "high", "High reliability"
    elif brier < baseline:
        tier, label = "moderate", "Moderate reliability"
    else:
        tier, label = "low", "Lower reliability"

    return CalibrationTier(
        tier=tier,
        label=label,
        explanation=(
            f"Brier {brier:.3f} VS Uninformative baseline of {baseline:.3f} for this class's prevalence in the training data."
        )
    )

_LIMITED_RELIABILITY_CLASSES = {
    "Investigation Pursued": ("This model rarely predicts Investigation Pursued. Treat low values for this outcome as 'unable to assess' rather than 'unlikely'.")
}

def calibration_tiers_from_results(
    classifier_results: Dict[str, float],
    base_rates: Dict[str, float],
    class_labels: list[str],
) -> Dict[str, CalibrationTier]:
    """Return a dict mapping a class label to its calibration tier, given the classifier's test set brier scores and the classes training base rates"""
    tiers: Dict[str, CalibrationTier] = {}
    for label in class_labels:
        brier = classifier_results.get(f"brier_{label}")
        base = base_rates.get(label, 0.0)
        if brier is None:
            tiers[label] = CalibrationTier(
                tier="low",
                label="Reliability unknown",
                explanation="Calibration data not available for this class.",
            )
        else:
            tiers[label] = calibration_tier(label, float(brier), base)
    return tiers
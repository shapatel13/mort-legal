"""Utilities for extracting case factors and calculating trial loss probability."""

from difflib import SequenceMatcher
import re


def calculate_trial_loss_probability(case_factors):
    """Calculate probability of losing a medical malpractice trial based on key case factors."""
    base_probability = 50
    adjustments = []

    # Positive factors (decrease probability)
    if case_factors.get("patient_left_ama", False):
        adjustment = -25
        base_probability += adjustment
        adjustments.append(("Patient left AMA", adjustment))

    if case_factors.get("refused_palliative_care", False):
        adjustment = -15
        base_probability += adjustment
        adjustments.append(("Refused palliative care", adjustment))

    if case_factors.get("documented_poor_prognosis", False):
        adjustment = -10
        base_probability += adjustment
        adjustments.append(("Documented poor prognosis", adjustment))

    if case_factors.get("multidisciplinary_consultation", False):
        adjustment = -5
        base_probability += adjustment
        adjustments.append(("Multidisciplinary consultation", adjustment))

    if case_factors.get("attempted_aggressive_care", False):
        adjustment = -5
        base_probability += adjustment
        adjustments.append(("Attempted aggressive care", adjustment))

    # Negative factors (increase probability)
    if case_factors.get("incomplete_documentation", False):
        adjustment = +15
        base_probability += adjustment
        adjustments.append(("Incomplete documentation", adjustment))

    if case_factors.get("delayed_care", False):
        adjustment = +20
        base_probability += adjustment
        adjustments.append(("Delayed care", adjustment))

    if case_factors.get("missed_diagnosis", False):
        adjustment = +25
        base_probability += adjustment
        adjustments.append(("Missed diagnosis", adjustment))

    # Care deviation factors - only increases probability if specifically identified
    care_deviations = {
        "icu_care_deviation": (+15, "ICU care deviation"),
        "shock_management_deviation": (+20, "Shock management deviation"),
        "respiratory_care_deviation": (+20, "Respiratory care deviation"),
        "inadequate_monitoring": (+15, "Inadequate monitoring"),
    }

    for factor, (value, name) in care_deviations.items():
        if case_factors.get(factor, False):
            base_probability += value
            adjustments.append((name, value))

    # Floor/cap the probability
    final_probability = max(min(base_probability, 95), 15)

    return final_probability, adjustments


def extract_case_factors(medical_record_text):
    """Extract both case factors for probability calculation and critical events for reporting."""
    case_factors = {
        # Standard factors affecting probability
        "patient_left_ama": False,
        "refused_palliative_care": False,
        "documented_poor_prognosis": False,
        "multidisciplinary_consultation": False,
        "attempted_aggressive_care": False,
        "incomplete_documentation": False,
        "delayed_care": False,
        "missed_diagnosis": False,
        # Care deviation factors (affect probability)
        "icu_care_deviation": False,
        "shock_management_deviation": False,
        "respiratory_care_deviation": False,
        "inadequate_monitoring": False,
        # Critical events (for detection only, don't affect probability)
        "icu_stay": False,
        "shock_state": False,
        "vasopressor_use": False,
        "hypoxia": False,
        "aki": False,
        "code_blue": False,
        "intubation": False,
        "sepsis": False,
    }

    # Define keywords for each factor
    keywords = {
        # Standard factors
        "patient_left_ama": ["against medical advice", "left AMA", "sign out AMA", "discharge AMA"],
        "refused_palliative_care": [
            "refused palliative",
            "declined palliative",
            "palliative care offered but",
            "hospice candidate but",
            "refused hospice",
            "declined hospice",
        ],
        "documented_poor_prognosis": [
            "poor prognosis",
            "guarded prognosis",
            "terminal",
            "end-stage",
            "limited life expectancy",
            "high mortality risk",
            "high risk of death",
        ],
        "multidisciplinary_consultation": [
            "multidisciplinary",
            "multiple consults",
            "team approach",
            "consulted with",
            "consult",
            "specialist",
            "nephrology",
        ],
        "attempted_aggressive_care": [
            "aggressive measures",
            "attempted intervention",
            "multiple attempts",
            "exhaustive care",
            "extensive treatment",
        ],
        "incomplete_documentation": [
            "incomplete documentation",
            "poor documentation",
            "inadequate documentation",
            "missing documentation",
            "documentation gaps",
        ],
        "delayed_care": [
            "delayed care",
            "delay in treatment",
            "untimely intervention",
            "treatment delay",
            "postponed care",
        ],
        "missed_diagnosis": [
            "missed diagnosis",
            "failure to diagnose",
            "misdiagnosis",
            "diagnostic error",
            "delayed diagnosis",
        ],
        # Care deviation factors
        "icu_care_deviation": [
            "inadequate ICU monitoring",
            "ICU transfer delay",
            "staffing issues in ICU",
            "insufficient critical care",
            "inappropriate ICU discharge",
        ],
        "shock_management_deviation": [
            "inadequate fluid resuscitation",
            "delayed vasopressor administration",
            "inappropriate vasopressor selection",
            "failure to recognize shock",
            "inadequate treatment of shock",
        ],
        "respiratory_care_deviation": [
            "delayed intubation",
            "inappropriate ventilator settings",
            "failure to recognize hypoxia",
            "inadequate oxygen therapy",
        ],
        "inadequate_monitoring": [
            "inadequate monitoring",
            "infrequent vital signs",
            "missed deterioration",
            "failure to reassess",
        ],
        # Critical events
        "icu_stay": ["ICU", "intensive care", "critical care", "step-down unit", "CCU"],
        "shock_state": [
            "shock",
            "hypotension",
            "hypotensive",
            "hemodynamic instability",
            "cardiogenic shock",
        ],
        "vasopressor_use": [
            "vasopressor",
            "norepinephrine",
            "norepi",
            "epinephrine",
            "dopamine",
            "vasopressin",
            "levophed",
            "pressors",
        ],
        "hypoxia": [
            "hypoxia",
            "hypoxic",
            "oxygen sat",
            "O2 sat",
            "SpO2",
            "desaturation",
            "respiratory distress",
            "respiratory failure",
        ],
        "aki": [
            "AKI",
            "acute kidney injury",
            "renal failure",
            "creatinine elevation",
            "kidney injury",
            "elevated BUN",
            "elevated creatinine",
        ],
        "code_blue": [
            "code blue",
            "cardiac arrest",
            "cardiopulmonary arrest",
            "CPR",
            "resuscitation",
            "ACLS",
            "defibrillation",
            "ventricular fibrillation",
        ],
        "intubation": [
            "intubation",
            "intubated",
            "mechanical ventilation",
            "ventilator",
            "endotracheal tube",
            "ETT",
            "respiratory failure",
        ],
        "sepsis": [
            "sepsis",
            "septic",
            "bacteremia",
            "systemic inflammatory response",
            "SIRS",
            "septic shock",
            "source of infection",
        ],
    }

    # Build regex patterns with word boundaries and simple morphological variations
    keyword_patterns = {}
    for factor, word_list in keywords.items():
        patterns = []
        for keyword in word_list:
            escaped = re.escape(keyword)
            escaped = escaped.replace("\\ ", r"\\s+")
            pattern = (
                r"(?<!\bno\s)(?<!\bnot\s)\b" + escaped + r"(?:s|es|ed|ing)?\b"
            )
            patterns.append((pattern, keyword))
        keyword_patterns[factor] = patterns

    text_lower = medical_record_text.lower()

    # Search text using regex patterns
    for factor, pattern_list in keyword_patterns.items():
        for pattern, base in pattern_list:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                start = match.start()
                context_start = max(0, start - 20)
                context_end = match.end() + 20
                context = text_lower[context_start:context_end]
                ratio = SequenceMatcher(None, base.lower(), context).ratio()
                if ratio >= 0.2:
                    case_factors[factor] = True
                    break
            if case_factors[factor]:
                break

    return case_factors


__all__ = ["calculate_trial_loss_probability", "extract_case_factors"]


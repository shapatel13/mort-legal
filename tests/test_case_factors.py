"""Unit tests for case factor extraction."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from case_factors import extract_case_factors


def test_extract_case_factors_positive():
    text = (
        "The patient was intubated and required vasopressors before leaving the hospital "
        "against medical advice."
    )
    factors = extract_case_factors(text)
    assert factors["intubation"]
    assert factors["vasopressor_use"]


def test_extract_case_factors_negative():
    text = "The patient rested comfortably with normal vital signs."
    factors = extract_case_factors(text)
    assert not any(factors.values())


def test_extract_case_factors_negated():
    text = (
        "The patient was not intubated and there was no vasopressor use. "
        "There was no sepsis."
    )
    factors = extract_case_factors(text)
    assert not factors["intubation"]
    assert not factors["vasopressor_use"]
    assert not factors["sepsis"]


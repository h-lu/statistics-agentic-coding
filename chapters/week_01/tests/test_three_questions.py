"""Tests for identifying the three statistical questions (description/inference/prediction)."""

from __future__ import annotations

import pytest


# Note: These tests will be implemented once the starter_code/solution.py
# defines the question classification functions.
# The tests below outline the expected behavior.


def test_identify_description_question():
    """Test that description questions are correctly identified."""
    # Description questions ask about the data itself
    description_examples = [
        "What is the average bill length of Adelie penguins?",
        "Which island has the most penguins?",
        "How many missing values are in the dataset?",
        "What is the distribution of body mass?",
    ]
    # TODO: Implement after solution.py has classify_question function
    # for question in description_examples:
    #     result = classify_question(question)
    #     assert result == "description"


def test_identify_inference_question():
    """Test that inference questions are correctly identified."""
    # Inference questions ask about generalizing from sample to population
    inference_examples = [
        "Is the difference in bill length between Adelie and Chinstrap statistically significant?",
        "Can we conclude that male penguins are heavier than females in the entire population?",
        "Is this correlation just due to chance?",
    ]
    # TODO: Implement after solution.py has classify_question function
    # for question in inference_examples:
    #     result = classify_question(question)
    #     assert result == "inference"


def test_identify_prediction_question():
    """Test that prediction questions are correctly identified."""
    # Prediction questions ask about predicting outcomes for new data
    prediction_examples = [
        "Given bill length and depth, can we predict the species?",
        "What will be the body mass of a penguin with flipper length 200mm?",
        "Can we classify penguins by island based on physical measurements?",
    ]
    # TODO: Implement after solution.py has classify_question function
    # for question in prediction_examples:
    #     result = classify_question(question)
    #     assert result == "prediction"


def test_mixed_question_classification():
    """Test classification of questions that mix multiple types."""
    # Questions that mix description and prediction
    mixed_examples = [
        ("What is the average bill length and can we predict species from it?", "prediction"),
        ("Is there a correlation and is it significant?", "inference"),
    ]
    # TODO: Implement after solution.py has classify_question function
    # for question, expected_primary_type in mixed_examples:
    #     result = classify_question(question)
    #     assert result == expected_primary_type


def test_question_classification_with_edge_cases():
    """Test classification with edge cases like ambiguous questions."""
    ambiguous_questions = [
        "How do these variables relate?",  # Could be description or inference
        "Tell me about the data.",  # Could be description or prediction
    ]
    # TODO: Implement after solution.py has classify_question function
    # Ambiguous questions should either return a default or raise an error
    # for question in ambiguous_questions:
    #     with pytest.raises(ValueError, match="ambiguous"):
    #         classify_question(question)

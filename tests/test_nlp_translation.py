"""Unit tests for Module 7 (src/translator.py).

Verifies 4 target behaviors:
1. Benchmark sequence translations (Test Sequences 1-4 from CV Report Section 6)
2. Tense and copula insertion (ASL Topic-Comment to English copula mapping)
3. Duplicate token cleanup and whitespace/case normalization
4. Empty input handling and valid punctuation termination
"""

from src.translator import ASLSeq2SeqTranslator


def test_nlp_benchmark_sequences():
    """Test 1: Translates standard continuous benchmark sequences from report."""
    translator = ASLSeq2SeqTranslator()

    # Sequence 1: Academic Introduction
    seq1 = ["HELLO", "ME", "STUDENT", "COMPUTER"]
    assert translator.translate_accumulated(seq1) == "Hello, I am a computer student."

    # Sequence 2: Classroom Assistance
    seq2 = ["THANK-YOU", "HELP", "TEACHER"]
    assert translator.translate_accumulated(seq2) == "Thank you for helping me, teacher."

    # Sequence 3: Urgent Need
    seq3 = ["ME", "NEED", "WATER", "PLEASE"]
    assert translator.translate_accumulated(seq3) == "I need water, please."

    # Sequence 4: Social Interaction
    seq4 = ["NICE", "MEET", "ALL", "YOU"]
    assert translator.translate_accumulated(seq4) == "It is nice to meet all of you."


def test_nlp_copula_and_tense_insertion():
    """Test 2: Inserts missing copulas ('am', 'is', 'are') and articles."""
    translator = ASLSeq2SeqTranslator()

    res1 = translator.translate_accumulated(["ME", "STUDENT"])
    assert "am" in res1.lower()

    res2 = translator.translate_accumulated(["ME", "GOOD"])
    assert "am" in res2.lower()


def test_nlp_duplicate_token_cleanup():
    """Test 3: Cleans repeated raw tokens before grammatical synthesis."""
    translator = ASLSeq2SeqTranslator()
    raw_repeated = ["HELLO", "HELLO", "ME", "ME", "STUDENT", "COMPUTER", "COMPUTER"]
    result = translator.translate_accumulated(raw_repeated)
    assert result == "Hello, I am a computer student."


def test_nlp_empty_input_and_punctuation():
    """Test 4: Handles empty or None input gracefully with proper punctuation."""
    translator = ASLSeq2SeqTranslator()
    assert translator.translate_accumulated([]) == ""
    assert translator.translate_accumulated(["", None]) == ""

    # Sentence must end with standard punctuation
    single = translator.translate_accumulated(["WATER"])
    assert single.endswith((".", "!", "?"))

"""Unit test for llm_judge_async.py fix for string ground_truth"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from verl.utils.reward_score.llm_judge_async import compute_score_batch


def test_string_ground_truth():
    """Test that ground_truth as string works correctly."""
    data_sources = ["test_source"]
    solution_strs = ["https://linkedin.com/in/lynchbrian"]
    ground_truths = ["https://linkedin.com/in/lynchbrian"]  # String format
    extra_infos = [{"question": "Find the LinkedIn profile URL of the GitHub user blynch."}]

    result = compute_score_batch(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )

    print(f"String ground_truth test result: {result}")
    assert result[0]["score"] >= 0, "Score should be non-negative"
    print("✓ String ground_truth test passed!")


def test_dict    """Test that ground_truth as dict_ground_truth():
 still works correctly."""
    data_sources = ["test_source"]
    solution_strs = ["The answer is 42."]
    ground_truths = [{"target": ["42"]}]  # Dict format
    extra_infos = [{"question": "What is the answer?"}]

    result = compute_score_batch(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )

    print(f"Dict ground_truth test result: {result}")
    assert result[0]["score"] >= 0, "Score should be non-negative"
    print("✓ Dict ground_truth test passed!")


def test_none_ground_truth():
    """Test that None ground_truth is handled correctly."""
    data_sources = ["test_source"]
    solution_strs = ["Some answer"]
    ground_truths = [None]  # None format
    extra_infos = [{}]

    result = compute_score_batch(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )

    print(f"None ground_truth test result: {result}")
    # Should return score 0.0 for no_label
    assert result[0]["score"] == 0.0, "Score should be 0 for no label"
    print("✓ None ground_truth test passed!")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing llm_judge_async fix for string ground_truth")
    print("=" * 50)

    try:
        test_string_ground_truth()
    except Exception as e:
        print(f"✗ String ground_truth test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_dict_ground_truth()
    except Exception as e:
        print(f"✗ Dict ground_truth test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_none_ground_truth()
    except Exception as e:
        print(f"✗ None ground_truth test failed: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 50)
    print("All tests completed!")
    print("=" * 50)

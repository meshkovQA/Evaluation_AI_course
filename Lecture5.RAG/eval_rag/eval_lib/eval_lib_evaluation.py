# test_batch_evaluation.py
import asyncio
from eval_lib import (
    evaluate,
    EvalTestCase,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric

)
from custom_proxy_llm import create_proxy_llm


# ==================== TEST 1: Batch Standard Metrics with OpenAI ====================

async def test_batch_standard_metrics():
    """Test batch evaluation with multiple test cases and standard metrics"""

    # Create test cases
    test_cases = [
        EvalTestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            expected_output="Paris",
            retrieval_context=["Paris is the capital of France."]
        ),
        EvalTestCase(
            input="What is photosynthesis?",
            actual_output="The weather today is sunny.",
            expected_output="Process by which plants convert light into energy",
            retrieval_context=[
                "Photosynthesis is the process by which plants use sunlight."]
        )
    ]

    # Define metrics
    metrics = [
        AnswerRelevancyMetric(
            model="gpt-4o-mini",
            threshold=0.7,
            temperature=1.0,
        ),
        FaithfulnessMetric(
            model="gpt-4o-mini",
            threshold=0.8,
            temperature=1.0,
        ),
        ContextualRelevancyMetric(
            model="gpt-4o-mini",
            threshold=1.0,
        ),
    ]

    # Run batch evaluation
    results = await evaluate(
        test_cases=test_cases,
        metrics=metrics,
        verbose=True
    )

    return results


async def test_standard_metrics_with_custom_model():
    """Test batch evaluation with multiple test cases and standard metrics"""

    proxy_model = create_proxy_llm(
        model="gpt-4o-mini",
        api_key="sk-proxy-your-key",
        base_url="http://5.11.83.110:8000"
    )

    # Create test cases
    test_cases = [
        EvalTestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            expected_output="Paris",
            retrieval_context=["Paris is the capital of France."]
        ),
        EvalTestCase(
            input="What is photosynthesis?",
            actual_output="The weather today is sunny.",
            expected_output="Process by which plants convert light into energy",
            retrieval_context=[
                "Photosynthesis is the process by which plants use sunlight."]
        )
    ]

    # Define metrics
    metrics = [
        AnswerRelevancyMetric(
            model=proxy_model,
            threshold=0.7,
            temperature=0.1,
        ),
        FaithfulnessMetric(
            model=proxy_model,
            threshold=0.8,
            temperature=1.0,
        ),
        ContextualRelevancyMetric(
            model=proxy_model,
            threshold=1.0,
        ),
    ]

    # Run batch evaluation
    results = await evaluate(
        test_cases=test_cases,
        metrics=metrics,
        verbose=True
    )

    return results


# ==================== Run All Tests ====================

async def run_all_tests():
    """Execute all batch evaluation tests"""

    # await test_batch_standard_metrics()
    await test_standard_metrics_with_custom_model()


if __name__ == "__main__":
    asyncio.run(run_all_tests())

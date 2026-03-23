# test_dataset_generation.py
import asyncio
from eval_lib import DatasetGenerator

from custom_proxy_llm import create_proxy_llm

proxy_model = create_proxy_llm(
    model="gpt-4o-mini",
    api_key="sk-proxy-your-key",
    base_url="http://5.11.83.110:8000"
)

# ==================== TEST 1: Generate from Scratch ====================


async def test_generate_from_scratch_basic():
    """Basic dataset generation without documents"""

    generator = DatasetGenerator(
        model=proxy_model,
        agent_description="A customer support chatbot for an e-commerce platform",
        input_format="User question or request about orders, shipping, returns",
        expected_output_format="Helpful and professional response",
        test_types=["functionality", "edge_cases"],
        max_rows=10,
        question_length="mixed",
        question_openness="mixed",
        trap_density=0.1,
        language="en",
        verbose=True

    )

    dataset = await generator.generate_from_scratch()

    return dataset


# ==================== TEST 2: Generate from Documents ====================

async def test_generate_from_multiple_documents():
    """Generate dataset from multiple documents"""

    generator = DatasetGenerator(
        model=proxy_model,
        embedding_model=proxy_model,
        agent_description="Technical documentation assistant",
        input_format="Technical question about product features",
        expected_output_format="Detailed answer with references",
        test_types=["retrieval", "accuracy", "functionality"],
        max_rows=10,
        question_length="mixed",
        question_openness="mixed",
        chunk_size=1000,
        chunk_overlap=50,
        max_chunks=30,
        verbose=True,
        language="ru"
    )

    file_paths = [
        "data/17.docx",
        "data/18.docx",
    ]
    dataset = await generator.generate_from_documents(file_paths)

    return dataset


# ==================== Run All Tests ====================

async def run_all_dataset_tests():
    """Execute all dataset generation tests"""

    # await test_generate_from_scratch_basic()
    await test_generate_from_multiple_documents()


if __name__ == "__main__":
    asyncio.run(run_all_dataset_tests())

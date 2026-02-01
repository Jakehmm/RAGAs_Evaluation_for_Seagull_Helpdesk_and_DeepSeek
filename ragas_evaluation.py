from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness, Faithfulness, AnswerRelevancy
from ragas.embeddings.base import embedding_factory
import datetime
from typing import List
import time
from responses_for_evaluation.seagull_result_for_evaluation import SEAGULL_RESULTS_1, SEAGULL_RESULTS_2, SEAGULL_RESULTS_3, SEAGULL_RESULTS_4, SEAGULL_RESULTS_5
from responses_for_evaluation.deepseek_result_for_evaluation import DEEPSEEK_RESULTS_1, DEEPSEEK_RESULTS_2, DEEPSEEK_RESULTS_3, DEEPSEEK_RESULTS_4, DEEPSEEK_RESULTS_5
import os
from dotenv import load_dotenv


# Load the environment variables (api keys)
load_dotenv()
EMBEDDING_API = os.getenv("SILICONFLOW_EMBEDDING_API")
LLM_API = os.getenv("DEEPSEEK_API")


# Return the first and last few characters of each chunk of text
def preview_text_chunks(text_chunks: List[str], num_chars: int = 30) -> List[str]:
    previews = []
    for chunk in text_chunks:
        if len(chunk) <= 2 * num_chars:
            previews.append(chunk)
        else:
            preview = f"{chunk[:num_chars]}...{chunk[-num_chars:]}"
            previews.append(preview)
    return previews


def seagull_auto_ragas_evaluation(user_input: str, retrieved_contexts: List[str], response: str, reference: str, entry_number: int = 1) -> None:
    """
    Perform RAGAS evaluation (Factual Correctness, Faithfulness, Answer Relevancy) based on the provided inputs.
    Also prints and stores the evaluation results in a text file.

    Parameters:
        user_input: The original user question/input.
        retrieved_contexts: The contexts retrieved from the knowledge base used to generate the response.
        response: The LLM-generated response to be evaluated.
        reference: The reference answer for factual correctness comparison.

    Returns: 
        None: Only prints and stores the evaluation results.
    """
    # Setup LLM
    llm_client = AsyncOpenAI(api_key=LLM_API, base_url="https://api.deepseek.com/v1")
    embeddings_client = AsyncOpenAI(api_key=EMBEDDING_API, base_url="https://api.siliconflow.cn/v1")
    llm = llm_factory("deepseek-chat", client=llm_client, max_tokens=8000)
    embeddings = embedding_factory(model="Qwen/Qwen3-Embedding-8B", client=embeddings_client)

    # Create metric
    f_scorer = Faithfulness(llm=llm)
    fc_scorer = FactualCorrectness(llm=llm)
    ar_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings) # type: ignore

    # Get the testing time, then shorten the displayed retrieved context 
    # (the evaluation still uses the long retrieved context)
    testing_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    retrieved_contexts_preview = preview_text_chunks(retrieved_contexts, num_chars=50)

    print("-----------------------------------")
    print(f"Entry Number: {entry_number}")
    print(f"Testing Time: {testing_time}")
    print(f"User Input: {user_input}")
    print(f"Retrieved Contexts: {retrieved_contexts_preview}")
    print(f"Response: {response}")
    print(f"Reference Answer: {reference}")
    print("-----------------------------------")

    print("Evaluating...\n")

    # Record the start time
    start_time = time.perf_counter()

    # Evaluate Faithfulness, Factual Correctness, and Answer Relevancy
    f_result = f_scorer.score(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts
    )
    print(f"Faithfulness Score: {f_result.value}")

    fc_result = fc_scorer.score(
        response=response,
        reference=reference
    )
    print(f"Factual Correctness Score: {fc_result.value}")

    ar_result = ar_scorer.score(
        user_input=user_input,
        response=response
    )
    print(f"Answer Relevancy Score: {ar_result.value}")

    # Record the end time, then calculate the time used
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    print("\nEvaluation completed.")
    print(f"Time Used for Evaluation: {elapsed_time:.2f} seconds\n")
    print("Appending evaluation results into a text file...")

    try:
        # Change the number suffix of file name when start evaluating a new dataset
        with open("seagull_ragas_evaluation_results_1.txt", "a", encoding="utf-8") as f:
            f.write("===================================\n")
            f.write(f"Entry Number: {entry_number}\n")
            f.write(f"Testing Time: {testing_time}\n")
            f.write(f"Time Used for Evaluation: {elapsed_time:.2f} seconds\n")
            f.write("-----------------------------------\n")
            f.write(f"User Input: {user_input}\n")
            f.write(f"Retrieved Contexts: {retrieved_contexts_preview}\n")
            f.write(f"Response: {response}\n")
            f.write(f"Reference Answer: {reference}\n")            
            f.write("-----------------------------------\n")
            f.write(f"Faithfulness Score: {f_result.value}\n")
            f.write(f"Factual Correctness Score: {fc_result.value}\n")
            f.write(f"Answer Relevancy Score: {ar_result.value}\n")
            f.write("===================================\n\n")
        print("Evaluation results has been appended to seagull_ragas_evaluation_results_1.txt")
    except Exception as e:
        print(f"Failed to write evaluation results to file: {str(e)}")
    print("-----------------------------------")


def deepseek_auto_ragas_evaluation(user_input: str, response: str, reference: str, entry_number: int = 1) -> None:
    """
    Perform RAGAS evaluation (Factual Correctness and Answer Relevancy) based on the provided inputs.
    Also prints and stores the evaluation results in a text file.

    Parameters:
        user_input: The original user question/input.
        response: The LLM-generated response to be evaluated.
        reference: The reference answer for factual correctness comparison.

    Returns: 
        None: Only prints and stores the evaluation results.
    """
    # Setup LLM
    llm_client = AsyncOpenAI(api_key=LLM_API, base_url="https://api.deepseek.com/v1")
    embeddings_client = AsyncOpenAI(api_key=EMBEDDING_API, base_url="https://api.siliconflow.cn/v1")
    llm = llm_factory("deepseek-chat", client=llm_client, max_tokens=8000)
    embeddings = embedding_factory(model="Qwen/Qwen3-Embedding-8B", client=embeddings_client)
    
    # Create metric
    fc_scorer = FactualCorrectness(llm=llm)
    ar_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings) # type: ignore

    # Get the testing time
    testing_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    

    print("-----------------------------------")
    print(f"Entry Number: {entry_number}")
    print(f"Testing Time: {testing_time}")
    print(f"User Input: {user_input}")
    print(f"Response: {response}")
    print(f"Reference Answer: {reference}")
    print("-----------------------------------")

    print("Evaluating...\n")

    # Record the start time
    start_time = time.perf_counter()

    # Evaluate Factual Correctness and Answer Relevancy
    fc_result = fc_scorer.score(
        response=response,
        reference=reference
    )
    print(f"Factual Correctness Score: {fc_result.value}")

    ar_result = ar_scorer.score(
        user_input=user_input,
        response=response
    )
    print(f"Answer Relevancy Score: {ar_result.value}")

    # Record the end time, then calculate the time used
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    print("\nEvaluation completed.")
    print(f"Time Used for Evaluation: {elapsed_time:.2f} seconds\n")
    print("Appending evaluation results into a text file...")

    try:
        # Change the number suffix of file name when start evaluating a new dataset
        with open("deepseek_ragas_evaluation_results_1.txt", "a", encoding="utf-8") as f:
            f.write("===================================\n")
            f.write(f"Entry Number: {entry_number}\n")
            f.write(f"Testing Time: {testing_time}\n")
            f.write(f"Time Used for Evaluation: {elapsed_time:.2f} seconds\n")
            f.write("-----------------------------------\n")
            f.write(f"User Input: {user_input}\n")
            f.write(f"Response: {response}\n")
            f.write(f"Reference Answer: {reference}\n")
            f.write("-----------------------------------\n")
            f.write(f"Factual Correctness Score: {fc_result.value}\n")
            f.write(f"Answer Relevancy Score: {ar_result.value}\n")
            f.write("===================================\n\n")
        print("Evaluation results has been appended to deepseek_ragas_evaluation_results_1.txt")
    except Exception as e:
        print(f"Failed to write evaluation results to file: {str(e)}")
    print("-----------------------------------")


def main():
    # Change the number suffix of the imported variable to evaluate a new dataset
    for index, entry in enumerate(SEAGULL_RESULTS_1[:1], start=1):
        seagull_auto_ragas_evaluation(
            user_input=entry["question"], 
            response=entry["response"], 
            retrieved_contexts=entry["retrieved_context"], 
            reference=entry["reference_answer"],
            entry_number=index
        )
    for index, entry in enumerate(DEEPSEEK_RESULTS_1[:1], start=1):
        deepseek_auto_ragas_evaluation(
            user_input=entry["question"], 
            response=entry["response"], 
            reference=entry["reference_answer"],
            entry_number=index
        )


if __name__ == "__main__":
    main()
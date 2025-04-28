# evaluation_text_based_improved.py

import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import retrieval_engine as retrieval

# Load gold standard dataset
dataset = pd.read_excel('gold_test_set.xlsx')  # Must have 'Question', 'Answer', 'Question Type'

# Initialize text encoder once
text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Retrieval methods
retrievers = {
    "FAISS": lambda q: retrieval.RetrievalEngine().semantic_search_faiss(q, top_k=1),
    "pgvector-IVFFLAT": lambda q: retrieval.RetrievalEngine().semantic_search_pgvector(q, method="ivfflat", top_k=1),
    "pgvector-HNSW": lambda q: retrieval.RetrievalEngine().semantic_search_pgvector(q, method="hnsw", top_k=1),
    "TF-IDF": lambda q: retrieval.RetrievalEngine().lexical_search_tfidf(q, top_k=1),
    "BM25": lambda q: retrieval.RetrievalEngine().lexical_search_bm25(q, top_k=1)
}

# Text similarity threshold
similarity_threshold = 0.65

# Store performance results
performance_summary = []

for retriever_name, retriever_func in retrievers.items():
    print(f"\n🔎 Evaluating method: {retriever_name}")

    correct_matches = 0
    total_answerable = 0
    false_positives = 0
    total_unanswerable = 0
    latency_measurements = []

    for idx, sample in dataset.iterrows():
        question_text = sample["Question"]
        expected_answer_text = sample["Answer"]
        is_answerable = sample["Question Type"].strip().lower() == "answerable"

        start_time = time.perf_counter()
        try:
            retrieval_results = retriever_func(question_text)
        except Exception as ex:
            print(f"⚠️ Retrieval error for question '{question_text}': {ex}")
            retrieval_results = []
        end_time = time.perf_counter()

        latency = end_time - start_time
        latency_measurements.append(latency)

        retrieved_text = retrieval_results[0].get('text', "") if retrieval_results else ""

        if is_answerable:
            total_answerable += 1
            if retrieved_text:
                expected_emb = text_encoder.encode([expected_answer_text])
                retrieved_emb = text_encoder.encode([retrieved_text])
                sim_score = cosine_similarity(expected_emb, retrieved_emb)[0][0]

                print(f"\nQuestion: {question_text}")
                print(f"Expected Answer: {expected_answer_text}")
                print(f"Retrieved Text: {retrieved_text}")
                print(f"Similarity Score: {sim_score:.3f}")

                if sim_score >= similarity_threshold:
                    correct_matches += 1
        else:
            total_unanswerable += 1
            if retrieved_text:
                # Assume any retrieval on unanswerable is a false positive
                print(f"\n⚠️ False Positive Detected! Unanswerable Question retrieved something.\nQuestion: {question_text}\nRetrieved: {retrieved_text}")
                false_positives += 1

    # Calculate metrics
    accuracy = correct_matches / total_answerable if total_answerable else 0
    rejection_rate = 1 - (false_positives / total_unanswerable) if total_unanswerable else 1.0
    avg_latency = sum(latency_measurements) / len(latency_measurements) if latency_measurements else 0

    performance_summary.append({
        "Retrieval Method": retriever_name,
        "Accuracy (Answerable)": round(accuracy, 3),
        "Rejection Rate (Unanswerable)": round(rejection_rate, 3),
        "Average Latency (s)": round(avg_latency, 3)
    })

# Save results
summary_df = pd.DataFrame(performance_summary)
summary_df.to_csv('evaluation_text_based_results.csv', index=False)

print("\n✅ Text-Based Evaluation Complete. Final Results:")
print(summary_df)

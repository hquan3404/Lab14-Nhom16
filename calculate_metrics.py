import json
import os

def calculate_averages(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    faithfulness_scores = []
    relevancy_scores = []

    for entry in data:
        ragas = entry.get('ragas', {})
        
        faithfulness = ragas.get('faithfulness')
        if faithfulness is not None:
            try:
                faithfulness_scores.append(float(faithfulness))
            except ValueError:
                pass

        relevancy = ragas.get('relevancy')
        if relevancy is not None:
            try:
                relevancy_scores.append(float(relevancy))
            except ValueError:
                pass

    print(f"--- Benchmark Results Summary ---")
    print(f"Total entries: {len(data)}")
    
    if faithfulness_scores:
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        print(f"Average Faithfulness: {avg_faithfulness:.4f} (from {len(faithfulness_scores)} samples)")
    else:
        print("No faithfulness data found.")

    if relevancy_scores:
        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
        print(f"Average Relevancy: {avg_relevancy:.4f} (from {len(relevancy_scores)} samples)")
    else:
        print("No relevancy data found.")

if __name__ == "__main__":
    file_path = 'reports/benchmark_results.json'
    calculate_averages(file_path)

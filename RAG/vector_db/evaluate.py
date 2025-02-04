import csv

def lcs(X, Y):
    """Hàm tính Longest Common Subsequence (LCS) giữa hai chuỗi X và Y."""
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]

def calculate_metrics(test_file, reference_file):
    with open(test_file, mode='r', encoding='utf-8') as test_csv, open(reference_file, mode='r', encoding='utf-8') as ref_csv:
        test_reader = csv.DictReader(test_csv)
        ref_reader = csv.DictReader(ref_csv)

        precision_total = 0
        recall_total = 0
        f1_total = 0
        count = 0

        for test_row, ref_row in zip(test_reader, ref_reader):
            test_answer = test_row['answer']
            ref_answer = ref_row['answer']

            lcs_length = lcs(test_answer, ref_answer)
            precision = lcs_length / len(test_answer) if len(test_answer) > 0 else 0
            recall = lcs_length / len(ref_answer) if len(ref_answer) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            precision_total += precision
            recall_total += recall
            f1_total += f1
            count += 1

        precision_avg = precision_total / count if count > 0 else 0
        recall_avg = recall_total / count if count > 0 else 0
        f1_avg = f1_total / count if count > 0 else 0

        return precision_avg, recall_avg, f1_avg

def main():
    test_file = '/home/minhlahanhne/DATN_test/RAG/vector_db/evaluation_results.csv'
    reference_file = '/home/minhlahanhne/DATN_test/RAG/vector_db/Q&A&C - Test_model.csv'

    precision, recall, f1 = calculate_metrics(test_file, reference_file)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main() 
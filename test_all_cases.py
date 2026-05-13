"""
Script test all cases in data.csv with API /predict
Compare predictions with actual target values
"""
import pandas as pd
import requests
import json
import time
import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

API_URL = "http://localhost:8000/predict"

def main():
    print("=" * 70)
    print("  TEST ALL CASES IN data.csv WITH API /predict")
    print("=" * 70)

    # Load data
    df = pd.read_csv('data.csv')
    # Fill missing values with mean (same as training)
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    total = len(df)
    print(f"\nTotal samples: {total}")

    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    # Track results
    results_list = []
    model_correct = {"KNN": 0, "Logistic Regression": 0, "SVM": 0}
    model_total = {"KNN": 0, "Logistic Regression": 0, "SVM": 0}
    errors = 0

    start_time = time.time()

    for idx, row in df.iterrows():
        actual_target = int(row['target'])
        payload = {col: float(row[col]) for col in feature_cols}

        try:
            resp = requests.post(API_URL, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            lr_pred = data['logistic_regression']['prediction']
            svm_pred = data['svm']['prediction']
            knn_pred = data['knn']['prediction']

            lr_prob = data['logistic_regression']['probability']
            svm_prob = data['svm']['probability']
            knn_prob = data['knn']['probability']

            lr_correct = 1 if lr_pred == actual_target else 0
            svm_correct = 1 if svm_pred == actual_target else 0
            knn_correct = 1 if knn_pred == actual_target else 0

            model_correct["Logistic Regression"] += lr_correct
            model_correct["SVM"] += svm_correct
            model_correct["KNN"] += knn_correct
            for m in model_total:
                model_total[m] += 1

            results_list.append({
                "STT": idx + 1,
                "age": row['age'], "sex": row['sex'], "cp": row['cp'],
                "trestbps": row['trestbps'], "chol": row['chol'],
                "fbs": row['fbs'], "restecg": row['restecg'],
                "thalach": row['thalach'], "exang": row['exang'],
                "oldpeak": row['oldpeak'], "slope": row['slope'],
                "ca": row['ca'], "thal": row['thal'],
                "Actual": actual_target,
                "LR_Pred": lr_pred, "LR_Prob": round(lr_prob, 4) if lr_prob else "",
                "LR_Correct": lr_correct,
                "SVM_Pred": svm_pred, "SVM_Prob": round(svm_prob, 4) if svm_prob else "",
                "SVM_Correct": svm_correct,
                "KNN_Pred": knn_pred, "KNN_Prob": round(knn_prob, 4) if knn_prob else "",
                "KNN_Correct": knn_correct,
            })

        except Exception as e:
            errors += 1
            results_list.append({
                "STT": idx + 1,
                "age": row['age'], "sex": row['sex'], "cp": row['cp'],
                "trestbps": row['trestbps'], "chol": row['chol'],
                "fbs": row['fbs'], "restecg": row['restecg'],
                "thalach": row['thalach'], "exang": row['exang'],
                "oldpeak": row['oldpeak'], "slope": row['slope'],
                "ca": row['ca'], "thal": row['thal'],
                "Actual": actual_target,
                "LR_Pred": "ERR", "LR_Prob": "", "LR_Correct": "",
                "SVM_Pred": "ERR", "SVM_Prob": "", "SVM_Correct": "",
                "KNN_Pred": "ERR", "KNN_Prob": "", "KNN_Correct": "",
            })

        # Progress update every 500 rows
        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - idx - 1) / rate if rate > 0 else 0
            print(f"  Progress: {idx + 1}/{total} ({(idx+1)/total*100:.1f}%) "
                  f"| Speed: {rate:.0f} samples/s | ETA: {remaining:.0f}s")

    elapsed_total = time.time() - start_time
    print(f"\nDone in {elapsed_total:.1f}s")

    # Save detailed results to CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('test_results.csv', index=False, encoding='utf-8-sig')
    print(f"Saved details: test_results.csv")

    # Summary
    tested = total - errors
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("  KET QUA TEST TAT CA CAC TRUONG HOP TRONG data.csv")
    summary_lines.append("=" * 70)
    summary_lines.append(f"")
    summary_lines.append(f"  Tong mau test : {total}")
    summary_lines.append(f"  Thanh cong    : {tested}")
    summary_lines.append(f"  Loi           : {errors}")
    summary_lines.append(f"  Thoi gian     : {elapsed_total:.1f}s")
    summary_lines.append(f"")
    summary_lines.append(f"  {'Model':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    summary_lines.append(f"  {'-' * 53}")
    for name in ["Logistic Regression", "SVM", "KNN"]:
        acc = model_correct[name] / model_total[name] * 100 if model_total[name] > 0 else 0
        summary_lines.append(f"  {name:<25} {model_correct[name]:>8} {model_total[name]:>8} {acc:>9.2f}%")
    summary_lines.append(f"")

    # Misclassification analysis
    if tested > 0:
        lr_wrong = tested - model_correct["Logistic Regression"]
        svm_wrong = tested - model_correct["SVM"]
        knn_wrong = tested - model_correct["KNN"]
        summary_lines.append(f"  {'Model':<25} {'Wrong':>8} {'False Pos':>10} {'False Neg':>10}")
        summary_lines.append(f"  {'-' * 55}")

        for name, key_pred in [("Logistic Regression", "LR_Pred"),
                                ("SVM", "SVM_Pred"),
                                ("KNN", "KNN_Pred")]:
            fp = sum(1 for r in results_list if r.get(key_pred) == 1 and r.get("Actual") == 0)
            fn = sum(1 for r in results_list if r.get(key_pred) == 0 and r.get("Actual") == 1)
            wrong = fp + fn
            summary_lines.append(f"  {name:<25} {wrong:>8} {fp:>10} {fn:>10}")

    summary_lines.append("=" * 70)

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save summary
    with open('test_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text + "\n")
        f.write("\nChi tiet ket qua tung mau: xem file test_results.csv\n")
    print(f"Saved summary: test_summary.txt")

if __name__ == '__main__':
    main()

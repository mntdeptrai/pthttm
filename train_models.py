import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
import joblib
import os
import json

def main():
    # ============================================================
    # 1. DATA LOADING
    # ============================================================
    print("=" * 60)
    print("HYPERTENSION PREDICTION - MODEL TRAINING")
    print("=" * 60)

    print("\n[1/6] Loading preprocessed data from local CSV...")
    df = pd.read_csv('data_cleaned.csv')
    print(f"  Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    # ============================================================
    # 2. DATA PREPROCESSING
    # ============================================================
    print("\n[2/6] Preprocessing data...")

    # Data is already preprocessed (duplicates removed, missing values filled with mode/mean)
    print(f"  Missing values: {df.isnull().sum().sum()}")

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    print(f"  Features: {X.shape[1]} columns")
    print(f"  Target distribution: {dict(y.value_counts())}")

    # ============================================================
    # 3. DATA SCALING (MinMaxScaler)
    # ============================================================
    print("\n[3/6] Scaling features with MinMaxScaler...")
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print(f"  Feature ranges scaled to [0, 1]")

    # ============================================================
    # 4. DATA SPLITTING (80/20 train/test, then 80/20 train/val)
    # ============================================================
    print("\n[4/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=50
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=50
    )
    print(f"  Training:   {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Testing:    {X_test.shape[0]} samples")

    # ============================================================
    # 5. TRAINING MODELS (Classification + Regression)
    # ============================================================
    print("\n[5/6] Training models...")

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "SVM": SVC(random_state=50, probability=True, class_weight='balanced'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=50, class_weight='balanced')
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")
        clf.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = clf.predict(X_val)
        print(f"    [Validation Report] {name}:")
        print(classification_report(y_val, y_val_pred))
        
        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred)
        val_rec = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_cm = confusion_matrix(y_val, y_val_pred)

        # Evaluate on test set
        y_test_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred)
        test_rec = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)

        results[name] = {
            "validation": {
                "accuracy": val_acc,
                "precision": val_prec,
                "recall": val_rec,
                "f1_score": val_f1,
                "confusion_matrix": val_cm.tolist()
            },
            "test": {
                "accuracy": test_acc,
                "precision": test_prec,
                "recall": test_rec,
                "f1_score": test_f1,
                "confusion_matrix": test_cm.tolist()
            }
        }

        print(f"    [Validation] Accuracy: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")
        print(f"    [Test]       Accuracy: {test_acc:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1: {test_f1:.4f}")
        print(f"    Confusion Matrix (Test):")
        print(f"      {test_cm}")

    # ============================================================
    # 6. COMPARISON & SAVING MODELS
    # ============================================================
    print("\n[6/6] Saving models...")

    os.makedirs('models', exist_ok=True)

    # Save scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    print("  Saved: models/scaler.joblib")

    # Save all classifiers
    model_filenames = {
        "Logistic Regression": "logistic_model.joblib",
        "SVM": "svm_model.joblib",
        "KNN": "knn_model.joblib",
        "Random Forest": "rf_model.joblib",
    }

    for name, clf in classifiers.items():
        filename = model_filenames[name]
        joblib.dump(clf, f'models/{filename}')
        print(f"  Saved: models/{filename}")

    # Save training results as JSON
    results_serializable = {}
    for name, metrics in results.items():
        results_serializable[name] = {
            "validation": {k: v for k, v in metrics["validation"].items()},
            "test": {k: v for k, v in metrics["test"].items()},
        }
    with open('models/training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    print("  Saved: models/training_results.json")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Val Acc':>10} {'Test Acc':>10} {'Test F1':>10}")
    print("-" * 55)

    best_model = None
    best_acc = 0
    for name, metrics in results.items():
        val_acc = metrics['validation']['accuracy']
        test_acc = metrics['test']['accuracy']
        test_f1 = metrics['test']['f1_score']
        marker = ""
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = name
        print(f"  {name:<23} {val_acc:>9.4f} {test_acc:>10.4f} {test_f1:>10.4f}")

    print(f"\n  Best model: {best_model} (Test Accuracy: {best_acc:.4f})")
    print(f"\n  Total data: {df.shape[0]} samples")
    print(f"  Models saved in: models/")
    print("=" * 60)
    print("DONE!")

if __name__ == '__main__':
    main()

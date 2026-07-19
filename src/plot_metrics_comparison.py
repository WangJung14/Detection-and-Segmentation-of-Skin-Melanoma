import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import os

def evaluate_tds(df):
    tds_score = 1.3 * df['A_score'] + 0.1 * df['B_score'] + 0.5 * df['C_score'] + 0.5 * df['D_score']
    return (tds_score > 5.45).astype(int)

def generate_visuals(train_csv, test_csv):
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    features = ['A_score', 'B_score', 'C_score', 'D_score']
    
    X_train = df_train[features]
    y_train = df_train['Label']
    
    X_test = df_test[features]
    y_test = df_test['Label']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(kernel='linear', class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    svm_pred = svm.predict(X_test_scaled)
    tds_pred = evaluate_tds(df_test)
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    tds_metrics = [
        accuracy_score(y_test, tds_pred),
        precision_score(y_test, tds_pred, zero_division=0),
        recall_score(y_test, tds_pred, zero_division=0),
        f1_score(y_test, tds_pred, zero_division=0)
    ]
    
    svm_metrics = [
        accuracy_score(y_test, svm_pred),
        precision_score(y_test, svm_pred, zero_division=0),
        recall_score(y_test, svm_pred, zero_division=0),
        f1_score(y_test, svm_pred, zero_division=0)
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, [m*100 for m in tds_metrics], width, label='Bác sĩ (TDS)', color='lightcoral')
    rects2 = ax.bar(x + width/2, [m*100 for m in svm_metrics], width, label='AI (Linear SVM Balanced)', color='skyblue')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('So sánh hiệu suất: TDS vs. SVM')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 110)
    ax.legend()
    
    ax.bar_label(rects1, fmt='%.1f', padding=3)
    ax.bar_label(rects2, fmt='%.1f', padding=3)
    
    fig.tight_layout()
    plt.savefig('data/Metrics_Comparison.png', dpi=300)
    print("Saved data/Metrics_Comparison.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_tds = confusion_matrix(y_test, tds_pred)
    sns.heatmap(cm_tds, annot=True, fmt='d', cmap='Reds', ax=axes[0], cbar=False,
                xticklabels=['Lành tính (0)', 'Ác tính (1)'],
                yticklabels=['Lành tính (0)', 'Ác tính (1)'])
    axes[0].set_title('Confusion Matrix - Bác sĩ (TDS)')
    axes[0].set_xlabel('Dự đoán (Predicted)')
    axes[0].set_ylabel('Thực tế (Actual)')
    
    cm_svm = confusion_matrix(y_test, svm_pred)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False,
                xticklabels=['Lành tính (0)', 'Ác tính (1)'],
                yticklabels=['Lành tính (0)', 'Ác tính (1)'])
    axes[1].set_title('Confusion Matrix - AI (Linear SVM)')
    axes[1].set_xlabel('Dự đoán (Predicted)')
    axes[1].set_ylabel('Thực tế (Actual)')
    
    fig.tight_layout()
    plt.savefig('data/Confusion_Matrices.png', dpi=300)
    print("Saved data/Confusion_Matrices.png")

if __name__ == "__main__":
    generate_visuals("data/final_dataset_svm_train.csv", "data/final_dataset_svm_val.csv")

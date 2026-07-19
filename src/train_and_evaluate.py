import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

def evaluate_tds(df):
    """
    Evaluates the Heuristic TDS Formula on a given dataset.
    """
    # TDS = 1.3*A + 0.1*B + 0.5*C + 0.5*D
    tds_score = 1.3 * df['A_score'] + 0.1 * df['B_score'] + 0.5 * df['C_score'] + 0.5 * df['D_score']
    
    # Predict based on threshold
    tds_pred = (tds_score > 5.45).astype(int)
    
    acc = accuracy_score(df['Label'], tds_pred)
    prec = precision_score(df['Label'], tds_pred, zero_division=0)
    rec = recall_score(df['Label'], tds_pred, zero_division=0)
    f1 = f1_score(df['Label'], tds_pred, zero_division=0)
    
    return acc, prec, rec, f1

def train_and_evaluate(train_csv, test_csv, output_plot):
    print("Loading datasets...")
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    if len(df_test) == 0:
        print("Test dataset is empty! Trying to split Train dataset to create a synthetic Test set...")
        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['Label'])
    
    features = ['A_score', 'B_score', 'C_score', 'D_score']
    
    X_train = df_train[features]
    y_train = df_train['Label']
    
    X_test = df_test[features]
    y_test = df_test['Label']
    
    print("\n--- LUỒNG 1: BÁC SĨ (TDS Formula) ---")
    tds_acc, tds_prec, tds_rec, tds_f1 = evaluate_tds(df_test)
    print(f"Accuracy:  {tds_acc:.4f}")
    print(f"Precision: {tds_prec:.4f}")
    print(f"Recall:    {tds_rec:.4f}")
    print(f"F1 Score:  {tds_f1:.4f}")
    
    print("\n--- LUỒNG 2: AI (Linear SVM) ---")
    # Chuẩn hóa
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Huấn luyện
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Đánh giá
    svm_pred = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_prec = precision_score(y_test, svm_pred, zero_division=0)
    svm_rec = recall_score(y_test, svm_pred, zero_division=0)
    svm_f1 = f1_score(y_test, svm_pred, zero_division=0)
    
    print(f"Accuracy:  {svm_acc:.4f}")
    print(f"Precision: {svm_prec:.4f}")
    print(f"Recall:    {svm_rec:.4f}")
    print(f"F1 Score:  {svm_f1:.4f}")
    
    print("\n--- SO SÁNH KẾT QUẢ ---")
    if svm_acc > tds_acc:
        print(">> SVM hiệu quả hơn (Accuracy cao hơn).")
    else:
        print(">> TDS hiệu quả hơn hoặc bằng SVM (Accuracy cao hơn hoặc bằng).")
        
    print("\n--- GIAI ĐOẠN 4: TRÍCH XUẤT TRỌNG SỐ VÀ VẼ BÁO CÁO ---")
    # Lấy trọng số (dùng absolute value cho SVM vì trọng số có thể âm, ảnh hưởng ngược chiều)
    svm_weights = np.abs(svm.coef_[0])
    
    # Trọng số bác sĩ
    tds_weights = np.array([1.3, 0.1, 0.5, 0.5])
    
    # Quy đổi về %
    svm_weights_pct = (svm_weights / np.sum(svm_weights)) * 100
    tds_weights_pct = (tds_weights / np.sum(tds_weights)) * 100
    
    print("Trọng số SVM (A, B, C, D) %:", svm_weights_pct)
    print("Trọng số TDS (A, B, C, D) %:", tds_weights_pct)
    
    # Vẽ Horizontal Bar Chart
    labels = ['Asymmetry (A)', 'Border (B)', 'Color (C)', 'Diameter (D)']
    y = np.arange(len(labels))
    height = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.barh(y + height/2, svm_weights_pct, height, label='AI (Linear SVM)', color='skyblue')
    rects2 = ax.barh(y - height/2, tds_weights_pct, height, label='Bác sĩ (TDS)', color='lightcoral')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Mức độ quan trọng (%)')
    ax.set_title('So sánh Trọng số Đặc trưng: AI (SVM) vs. Bác sĩ (TDS)')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.legend()
    
    ax.bar_label(rects1, fmt='%.1f%%', padding=3)
    ax.bar_label(rects2, fmt='%.1f%%', padding=3)
    
    fig.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"\nBiểu đồ đã được lưu tại: {output_plot}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--output_plot', required=True)
    args = parser.parse_args()
    
    train_and_evaluate(args.train_csv, args.test_csv, args.output_plot)

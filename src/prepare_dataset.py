import pandas as pd
import argparse
import os

def prepare_dataset(features_csv, metadata_csv, output_csv):
    if not os.path.exists(metadata_csv):
        print(f"Error: {metadata_csv} not found.")
        return
    
    print(f"Đang đọc file đặc trưng: {features_csv}")
    df_features = pd.read_csv(features_csv)
    
    print(f"Đang đọc file Metadata: {metadata_csv}")
    df_metadata = pd.read_csv(metadata_csv)
    
    col_img = 'isic_id' if 'isic_id' in df_metadata.columns else 'image_id' if 'image_id' in df_metadata.columns else 'image'
    col_dx = 'diagnosis' if 'diagnosis' in df_metadata.columns else 'dx' if 'dx' in df_metadata.columns else 'MEL'
    
    df_meta_clean = df_metadata[[col_img, col_dx]].copy()
    df_meta_clean.rename(columns={col_img: 'image_id'}, inplace=True)
    
    malignant_labels = ['mel', 'bcc', 'melanoma', 'basal cell carcinoma', 'squamous cell carcinoma', 'malignant']
    
    if col_dx == 'MEL':
        df_meta_clean['Label'] = (df_meta_clean[col_dx] == 1.0).astype(int)
    else:
        df_meta_clean['Label'] = df_meta_clean[col_dx].apply(lambda x: 1 if str(x).lower() in malignant_labels else 0)
    
    df_real_final = pd.merge(df_features, df_meta_clean[['image_id', 'Label']], on='image_id', how='inner')
    
    print(f"Số lượng ảnh sau khi merge: {len(df_real_final)} / {len(df_features)}")
    if len(df_real_final) == 0:
        print("WARNING: Merge resulted in 0 rows. Please check image_id formatting.")
    else:
        df_real_final.to_csv(output_csv, index=False)
        print(f"Đã lưu dataset hoàn chỉnh vào: {output_csv}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_csv', required=True)
    parser.add_argument('--metadata_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    args = parser.parse_args()
    
    prepare_dataset(args.features_csv, args.metadata_csv, args.output_csv)

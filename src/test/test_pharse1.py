import cv2
import numpy as np

from src.preprocessing import remove_hair, enhance_contrast_clahe, boost_faint_edges

if __name__ == "__main__":
    
    img = cv2.imread("../../data/toy_data/melanoma/ISIC_0000074.jpg")

    if img is not None:
        
        clean_img, _ = remove_hair(img, kernel_size=15, inpaint_rad=3)

        
        clahe_color_img, clahe_gray = enhance_contrast_clahe(clean_img, clip_limit=1.1)

        boosted_l, faint_mask = boost_faint_edges(clahe_gray)

        smooth_l = cv2.GaussianBlur(boosted_l, (5, 5), 0)

        
        
        clahe_gray_3_channels = cv2.cvtColor(smooth_l, cv2.COLOR_GRAY2BGR)

        
        scale = 0.3

        def resize_img(image, scale_factor):
            width = int(image.shape[1] * scale_factor)
            height = int(image.shape[0] * scale_factor)
            return cv2.resize(image, (width, height))

        img_res = resize_img(img, scale)
        clean_res = resize_img(clean_img, scale)
        clahe_color_res = resize_img(clahe_color_img, scale)
        clahe_gray_res = resize_img(clahe_gray_3_channels, scale)

        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_res, "1. Original", (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(clean_res, "2. DullRazor", (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(clahe_color_res, "3. CLAHE Color", (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(clahe_gray_res, "4. Smoothed L-Channel", (10, 30), font, 0.8, (0, 255, 0), 2)

        
        
        top_row = np.hstack((img_res, clean_res))  
        bottom_row = np.hstack((clahe_color_res, clahe_gray_res))  
        grid_combined = np.vstack((top_row, bottom_row))  

        
        cv2.imshow("Preprocessing Pipeline Grid", grid_combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không tìm thấy ảnh! Hãy kiểm tra lại đường dẫn.")
import cv2
import numpy as np


img = cv2.imread("../data/toy_data/melanoma/ISIC_0000074.jpg")

img = cv2.resize(img, (600, 450))


def nothing(x):
    pass  



cv2.namedWindow('Tuning Station')


cv2.createTrackbar('Kernel Size', 'Tuning Station', 15, 50, nothing)
cv2.createTrackbar('Inpaint Rad', 'Tuning Station', 3, 20, nothing)
cv2.createTrackbar('CLAHE Limit (x10)', 'Tuning Station', 15, 50, nothing)  

while True:
    
    k_size = cv2.getTrackbarPos('Kernel Size', 'Tuning Station')
    inp_rad = cv2.getTrackbarPos('Inpaint Rad', 'Tuning Station')
    clahe_val = cv2.getTrackbarPos('CLAHE Limit (x10)', 'Tuning Station') / 10.0

    
    if k_size % 2 == 0: k_size += 1
    if k_size < 3: k_size = 3
    if inp_rad < 1: inp_rad = 1

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, thresh_pre = cv2.threshold(blackhat, 10, 255, cv2.THRESH_TOZERO)
    _, mask = cv2.threshold(thresh_pre, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    clean = cv2.inpaint(img, mask, inpaintRadius=inp_rad, flags=cv2.INPAINT_TELEA)

    lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_val, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    final_res = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    

    
    combined = np.hstack((img, final_res))
    cv2.imshow('Tuning Station', combined)

    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print(f"THÔNG SỐ CHỐT:\nKernel: {k_size}\nInpaint: {inp_rad}\nCLAHE: {clahe_val}")
        break

cv2.destroyAllWindows()
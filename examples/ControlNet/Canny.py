import cv2
import os
from tqdm import tqdm


image_root = "/opt/custom_data/custom_cat2"
save_dir = "/opt/condition/canny/custom_cat2"

low_threshold = 100
high_threshold = 200

image_path_list = os.listdir(image_root)

os.makedirs(save_dir, exist_ok=True)

for path in tqdm(image_path_list):
    img_path = os.path.join(image_root, path)
    image = cv2.imread(img_path)
    canny = cv2.Canny(image, low_threshold, high_threshold)
    file_name = os.path.basename(img_path)
    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, canny)

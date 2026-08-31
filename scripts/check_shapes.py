import cv2
import json

base_path = r"C:\d-s\Drift-Sense-V2\dataset\hackathon_v4\dram\case_v4_dram_present_000"
ref = cv2.imread(base_path + r"\reference.png", cv2.IMREAD_GRAYSCALE)
search = cv2.imread(base_path + r"\search.png", cv2.IMREAD_GRAYSCALE)
print("ref shape:", ref.shape if ref is not None else None)
print("search shape:", search.shape if search is not None else None)

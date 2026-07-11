import requests
import numpy as np
import cv2

# create a dummy image
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imwrite("test_upload.png", img)

with open("test_upload.png", "rb") as f:
    resp = requests.post("http://localhost:8000/api/process_image", files={"file": f})

print(resp.status_code)
data = resp.json()
print("Keys:", data.keys())
if "status" in data:
    print("Status:", data["status"])
if "message" in data:
    print("Message:", data["message"])
if "voxel_matrix" in data:
    matrix = np.array(data["voxel_matrix"])
    print("Bead count:", np.sum(matrix))

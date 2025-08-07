import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Khởi tạo biến đếm
count_area_1 = 0
count_area_2 = 0
count_area_3 = 0
count_area_4 = 0

# Load YOLO model
model_path = "/home/tienthanh/yolo/best_ncnn_model"  # <-- Đường dẫn model
model = YOLO(model_path)
labels = model.names

# Capture image từ webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture image")
    exit()

# Lưu ảnh đã chụp (tùy chọn)
cv2.imwrite("captured_image.jpg", frame)
print("Finished capturing the image")

# Chạy YOLO inference
results = model(frame, verbose=False)
detections = results[0].boxes

# Vùng đếm theo class
class_counts_1 = defaultdict(int)
class_counts_2 = defaultdict(int)
class_counts_3 = defaultdict(int)
class_counts_4 = defaultdict(int)

# Màu bbox
bbox_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

# Vẽ các bbox
for i, det in enumerate(detections):
    xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
    xmin, ymin, xmax, ymax = xyxy
    conf = det.conf.item()
    class_id = int(det.cls.item())
    label = f'{labels[class_id]}: {int(conf * 100)}%'
    xcen = (xmin + xmax) / 2

    if conf > 0.5:
        if 0 < xcen < 160:
            count_area_1 += 1
            class_counts_1[class_id] += 1
        elif 160 < xcen < 320:
            count_area_2 += 1
            class_counts_2[class_id] += 1
        elif 320 < xcen < 480:
            count_area_3 += 1
            class_counts_3[class_id] += 1
        elif 480 < xcen < 640:
            count_area_4 += 1
            class_counts_4[class_id] += 1

        color = bbox_colors[i % len(bbox_colors)]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Hàm in class xuất hiện nhiều nhất mỗi vùng
def print_top_class(area_name, class_counts):
    if class_counts:
        top_class_id = max(class_counts, key=class_counts.get)
        class_name = labels[top_class_id]
        print(f"{area_name}: {class_name}")
    else:
        print(f"{area_name}: No detections")

# In kết quả
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#This is what i want to send to java web socket
print_top_class("Area 1", class_counts_1)
print_top_class("Area 2", class_counts_2)
print_top_class("Area 3", class_counts_3)
print_top_class("Area 4", class_counts_4)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print("Number of detections in Area 1:", count_area_1)
print("Number of detections in Area 2:", count_area_2)
print("Number of detections in Area 3:", count_area_3)
print("Number of detections in Area 4:", count_area_4)

# Lưu ảnh kết quả
cv2.imwrite("captured_image_result.jpg", frame)
# cv2.imshow("Detection Result", frame)
cv2.waitKey(0)

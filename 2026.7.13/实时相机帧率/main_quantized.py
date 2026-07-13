"""OpenVINO INT8 量化推理 — CPU帧率最大化"""
import os
import cv2
import time
from ultralytics import YOLO


# ===================== 环境配置 (匹配12核CPU) =====================
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"

# ===================== 加载 OpenVINO INT8 量化模型 =====================
model = YOLO("yolo26n-seg_int8_openvino_model", task="segment")
print("[INFO] OpenVINO INT8 量化模型加载成功")

# ===================== 摄像头初始化 =====================
video_path = 0
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 25)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    print("打开失败，请检查摄像头设备！")
    cap.release()
    exit()

fps_out = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] 摄像头: {w}x{h} @ {fps_out:.0f}FPS")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("seg_output_openvino_int8.mp4", fourcc, fps_out, (w, h))

# ===================== 流式推理 =====================
results = model(video_path, stream=True, imgsz=320, conf=0.35, iou=0.45, verbose=False)

frame_count = 0
fps_start = time.time()
total_start = time.time()
print("[INFO] 推理启动，按 q 键退出...")

for result in results:
    masks = result.masks
    n_det = len(masks) if masks is not None else 0

    img_bgr = result.plot()

    frame_count += 1
    elapsed = time.time() - fps_start
    if elapsed >= 2.0:
        total_elapsed = time.time() - total_start
        print(f"目标: {n_det:>2}  |  FPS: {frame_count / elapsed:.1f}  |  总计: {frame_count / total_elapsed:.1f} FPS")
        frame_count = 0
        fps_start = time.time()

    cv2.imshow("OpenVINO INT8 实例分割", img_bgr)
    out.write(img_bgr)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("处理完成，输出文件 seg_output_openvino_int8.mp4")

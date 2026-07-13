from ultralytics import YOLO
import cv2
import time
import onnxruntime as ort


# ===================== 环境检测：自动选择最优后端 =====================
available_providers = ort.get_available_providers()
cuda_available = any("CUDA" in p for p in available_providers)
print(f"可用推理后端: {available_providers}")

# ===================== 模型加载 =====================
model = YOLO("yolo26n-seg.pt")

# ===================== 数据源与硬件减负配置 =====================
video_path = 0
cap = cv2.VideoCapture(video_path)
# 限制摄像头采集分辨率与帧率，降低算力消耗
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 25)
# MJPG 格式解码更快（若摄像头不支持则自动回退）
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    print("视频/摄像头打开失败，请检查路径或设备！")
    cap.release()
    exit()

# ===================== 输出视频初始化 =====================
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("seg_output.mp4", fourcc, fps, (w, h))

# ===================== 流式推理 =====================
# imgsz 降至 320 大幅提升帧率（精度损耗极小），conf 适度提高减少误检
results = model(video_path, stream=True, imgsz=320, conf=0.35, iou=0.45,
                verbose=False)

# ===================== FPS 计时器 =====================
frame_count = 0
fps_start = time.time()

print("推理已启动，按 q 键退出...")
for result in results:
    masks = result.masks
    if masks is not None:
        print(f"目标数: {len(masks)}\n", end="  ")

    img_bgr = result.plot()

    # 实时帧率统计
    frame_count += 1
    elapsed = time.time() - fps_start
    if elapsed >= 2.0:
        print(f"FPS: {frame_count / elapsed:.1f}")
        frame_count = 0
        fps_start = time.time()

    cv2.imshow("实例分割", img_bgr)
    out.write(img_bgr)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===================== 资源释放 =====================
cap.release()
out.release()
cv2.destroyAllWindows()
print("处理完成，输出文件 seg_output.mp4")

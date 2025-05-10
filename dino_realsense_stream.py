import pyrealsense2 as rs
import cv2
import torch
import numpy as np
import threading

from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from torchvision.ops import box_convert
from PIL import Image

# ==== CONFIG ====
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
CPU_ONLY = True  # Set to False if using CUDA
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

# ==== Device ====
device = torch.device("cpu") if CPU_ONLY or not torch.backends.cuda.is_built() else torch.device("cuda")

# ==== Load Model ====
model = load_model(CONFIG_PATH, WEIGHTS_PATH)
model = model.to(device)

# ==== Global Prompt ====
TEXT_PROMPT = "a person"

def prompt_listener():
    global TEXT_PROMPT
    while True:
        user_input = input("Enter new prompt (or leave blank to skip): ").strip()
        if user_input:
            TEXT_PROMPT = user_input
            print(f"[INFO] Updated prompt to: {TEXT_PROMPT}")

# ==== Start listening thread ====
threading.Thread(target=prompt_listener, daemon=True).start()

# ==== Start RealSense ====
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# ==== Transform ====
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transformed_image, _ = transform(pil_image, None)

        # Run GroundingDINO detection
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=model,
                image=transformed_image.to(device),
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=str(device)
            )

        # Draw boxes and labels
        h, w, _ = frame.shape
        boxes = boxes * torch.tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        for box, phrase in zip(xyxy, phrases):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, phrase, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        # Show the annotated frame
        cv2.imshow("Grounding DINO - RealSense Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

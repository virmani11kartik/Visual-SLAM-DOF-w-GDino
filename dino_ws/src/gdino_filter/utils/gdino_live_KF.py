import pyrealsense2 as rs
import cv2
import torch
import numpy as np
from PIL import Image
import threading

from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from torchvision.ops import box_convert
from segment_anything import sam_model_registry, SamPredictor

# ==== CONFIG ====
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "weights/sam/sam_vit_b_01ec64.pth"
TEXT_PROMPT = "a person"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
FRAME_SKIP = 2
DOWNSCALE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load Models ====
gdino_model = load_model(CONFIG_PATH, WEIGHTS_PATH).to(device)
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(device)
sam_predictor = SamPredictor(sam)

# ==== Realsense Setup ====
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# ==== Transforms ====
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ==== Kalman Filter Class ====
class BoxKalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i+4] = 1
        self.kf.measurementMatrix = np.zeros((4, 8), np.float32)
        self.kf.measurementMatrix[:, :4] = np.eye(4)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.initialized = False

    def update(self, box):
        if not self.initialized:
            self.kf.statePre[:4, 0] = box
            self.kf.statePre[4:, 0] = 0
            self.initialized = True
        self.kf.correct(box)
        pred = self.kf.predict()
        return pred[:4].flatten()

kalman = BoxKalman()

# ==== Main Loop ====
frame_id = 0
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, _ = frame.shape
        mask_union = np.zeros((h, w), dtype=np.uint8)

        if frame_id % FRAME_SKIP == 0:
            image_pil = Image.fromarray(frame_rgb)
            if DOWNSCALE:
                image_pil = image_pil.resize((320, 240))  # optional
            transformed_image, _ = transform(image_pil, None)

            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=gdino_model,
                    image=transformed_image.to(device),
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    device=str(device),
                )

            if boxes.nelement() > 0:
                boxes_scaled = boxes * torch.tensor([w, h, w, h], device=boxes.device)
                xyxy = box_convert(boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

                sam_predictor.set_image(frame_rgb)

                for box in xyxy:
                    smooth_box = kalman.update(np.array(box, dtype=np.float32))
                    box_int = np.array(smooth_box, dtype=int)
                    masks, _, _ = sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box_int[None, :],
                        multimask_output=False,
                    )
                    mask_union = np.maximum(mask_union, masks[0].astype(np.uint8))

        # Inpaint masked region
        mask_255 = (mask_union * 255).astype(np.uint8)
        inpainted = cv2.inpaint(frame, mask_255, 3, cv2.INPAINT_TELEA)

        # Display
        cv2.imshow("Original", frame)
        cv2.imshow("Filtered Mask", inpainted)

        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

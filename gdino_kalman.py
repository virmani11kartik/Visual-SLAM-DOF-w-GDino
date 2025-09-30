import pyrealsense2 as rs
import cv2
import torch
import numpy as np
from PIL import Image
from filterpy.kalman import KalmanFilter
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load Models ====
gdino_model = load_model(CONFIG_PATH, WEIGHTS_PATH).to(device)
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(device)
sam_predictor = SamPredictor(sam)

# ==== RealSense Setup ====
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# ==== Image Transform ====
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ==== Background Buffer ====
background_buffer = None

# ==== Kalman Filter Setup ====
class SmoothedBoxKF:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.eye(8)
        for i in range(4):  # x1, y1, x2, y2 + dx, dy, dw, dh
            self.kf.F[i, i+4] = 1
        self.kf.H = np.eye(4, 8)
        self.kf.P *= 1000.
        self.kf.R *= 10.
        self.kf.Q *= 0.01
        self.initialized = False
        self.frame_since_seen = 0
        self.max_gap = 2  # max frames without detection

    def update(self, z):
        self.frame_since_seen = 0
        if not self.initialized:
            self.kf.x[:4] = z.reshape(4, 1)
            self.initialized = True
        else:
            self.kf.predict()
            self.kf.update(z.reshape(4, 1))

    def predict(self):
        if not self.initialized or self.frame_since_seen >= self.max_gap:
            return None
        self.kf.predict()
        self.frame_since_seen += 1
        return self.kf.x[:4].reshape(-1)

    def get_smoothed_box(self):
        if not self.initialized:
            return None
        return self.kf.x[:4].reshape(-1)

kf = SmoothedBoxKF()

# ==== Main Loop ====
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        transformed_image, _ = transform(image_pil, None)

        # Detect with GroundingDINO
        with torch.no_grad():
            boxes, _, _ = predict(
                model=gdino_model,
                image=transformed_image.to(device),
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=str(device),
            )

        boxes_scaled = boxes * torch.tensor([w, h, w, h])
        xyxy_boxes = box_convert(boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

        use_kf = False

        if len(xyxy_boxes) > 0:
            det_box = xyxy_boxes[0]  # take first or largest
            kf.update(np.array(det_box))
            use_kf = True
        else:
            prediction = kf.predict()
            if prediction is not None:
                det_box = prediction
                use_kf = True
            else:
                continue  # no detection or valid prediction

        # Clip and pad the smoothed box
        x1, y1, x2, y2 = det_box.astype(int)
        x1 = max(0, x1 - 10)
        y1 = max(0, y1 - 10)
        x2 = min(w, x2 + 10)
        y2 = min(h, y2 + 10)

        # Segment with SAM
        sam_predictor.set_image(frame_rgb)
        masks, _, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([[x1, y1, x2, y2]]),
            multimask_output=False,
        )
        mask_union = masks[0].astype(np.uint8)

        # Update background
        if background_buffer is None:
            background_buffer = np.zeros_like(frame)
        background_buffer[mask_union == 0] = frame[mask_union == 0]

        invisible_frame = frame.copy()
        unknown_mask = np.all(background_buffer == 0, axis=2)
        mask_to_inpaint = np.logical_and(mask_union == 1, unknown_mask)

        invisible_frame[mask_union == 1] = background_buffer[mask_union == 1]

        if np.any(mask_to_inpaint):
            kernel = np.ones((15, 15), np.uint8)
            dilated = cv2.dilate(mask_to_inpaint.astype(np.uint8) * 255, kernel, iterations=2)
            invisible_frame = cv2.inpaint(invisible_frame, dilated, 5, cv2.INPAINT_TELEA)

        # ==== Visualization ====
        if use_kf:
            cv2.rectangle(invisible_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(invisible_frame, "Smoothed Box", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if len(xyxy_boxes) > 0:
            raw_box = xyxy_boxes[0].astype(int)
            cv2.rectangle(invisible_frame, (raw_box[0], raw_box[1]), (raw_box[2], raw_box[3]), (0, 0, 255), 1)
            cv2.putText(invisible_frame, "Raw DINO Box", (raw_box[0], raw_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # ==== Show Frames ====
        cv2.imshow("Original", frame)
        cv2.imshow("Invisible View", invisible_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

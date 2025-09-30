e
import pyrealsense2 as rs
import cv2
import torch
import numpy as np
from PIL import Image
import uuid
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

# ==== Kalman Filter ====
class SmoothedBoxKF:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.eye(8)
        for i in range(4):
            self.kf.F[i, i+4] = 1
        self.kf.H = np.eye(4, 8)
        self.kf.P *= 1000.
        self.kf.R *= 10.
        self.kf.Q *= 0.01
        self.initialized = False
        self.frame_since_seen = 0
        self.max_gap = 2

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

# ==== Multi Object Tracker ====
class MultiObjectTracker:
    def __init__(self, iou_threshold=0.3, max_gap=2):
        self.trackers = {}
        self.iou_threshold = iou_threshold
        self.max_gap = max_gap

    def update(self, detections):
        updated_ids = set()
        for det in detections:
            best_iou = 0
            best_id = None
            for track_id, tracker in self.trackers.items():
                pred = tracker.get_smoothed_box()
                if pred is not None:
                    iou = self.compute_iou(det, pred)
                    if iou > best_iou:
                        best_iou = iou
                        best_id = track_id

            if best_iou > self.iou_threshold:
                self.trackers[best_id].update(det)
                updated_ids.add(best_id)
            else:
                new_id = str(uuid.uuid4())
                self.trackers[new_id] = SmoothedBoxKF()
                self.trackers[new_id].update(det)
                updated_ids.add(new_id)

        for track_id, tracker in list(self.trackers.items()):
            if track_id not in updated_ids:
                pred = tracker.predict()
                if pred is None:
                    del self.trackers[track_id]

    def get_active_boxes(self):
        return [tracker.get_smoothed_box() for tracker in self.trackers.values() if tracker.get_smoothed_box() is not None]

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

tracker = MultiObjectTracker()

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

        # Detection
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

        tracker.update(xyxy_boxes)
        smoothed_boxes = tracker.get_active_boxes()

        sam_predictor.set_image(frame_rgb)
        mask_union = np.zeros((h, w), dtype=np.uint8)

        for box in smoothed_boxes:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(w, x2 + 10)
            y2 = min(h, y2 + 10)

            masks, _, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([[x1, y1, x2, y2]]),
                multimask_output=False,
            )
            mask_union = np.maximum(mask_union, masks[0].astype(np.uint8))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

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

        cv2.imshow("Original", frame)
        cv2.imshow("Invisible View", invisible_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

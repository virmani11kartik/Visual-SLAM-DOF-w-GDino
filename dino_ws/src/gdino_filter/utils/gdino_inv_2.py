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

# ==== Transform ====
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ==== Optional live prompt update ====
def prompt_listener():
    global TEXT_PROMPT
    while True:
        user_input = input("New prompt (or blank to skip): ").strip()
        if user_input:
            TEXT_PROMPT = user_input
            print(f"[INFO] Updated prompt to: {TEXT_PROMPT}")

threading.Thread(target=prompt_listener, daemon=True).start()

# ==== Initialize buffer ====
background_buffer = None

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

        # GroundingDINO detection
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=gdino_model,
                image=transformed_image.to(device),
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=str(device),
            )

        boxes_scaled = boxes * torch.tensor([w, h, w, h])
        xyxy_boxes = box_convert(boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Run SAM
        sam_predictor.set_image(frame_rgb)
        mask_union = np.zeros((h, w), dtype=np.uint8)

        for box in xyxy_boxes:
            # Slightly expand box
            box[0] = max(0, box[0] - 10)
            box[1] = max(0, box[1] - 10)
            box[2] = min(w, box[2] + 10)
            box[3] = min(h, box[3] + 10)

            input_box = np.array(box).astype(int)
            masks, _, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            mask = masks[0].astype(np.uint8)
            mask_union = np.maximum(mask_union, mask)

        # ==== Hybrid Background Replacement ====
        if background_buffer is None:
            background_buffer = np.zeros_like(frame)

        # Update background where not masked
        background_buffer[mask_union == 0] = frame[mask_union == 0]

        # Build output image
        invisible_frame = frame.copy()

        # Find unknown areas (where background_buffer is all 0s)
        unknown_mask = np.all(background_buffer == 0, axis=2)
        mask_to_inpaint = np.logical_and(mask_union == 1, unknown_mask)

        # Use background where known
        invisible_frame[mask_union == 1] = background_buffer[mask_union == 1]

        # Fallback: Inpaint where background is unknown
        if np.any(mask_to_inpaint):
            mask_255 = mask_to_inpaint.astype(np.uint8) * 255
            invisible_frame = cv2.inpaint(invisible_frame, mask_255, 3, cv2.INPAINT_TELEA)

        # ==== Display ====
        cv2.imshow("Original", frame)
        cv2.imshow("Invisible View", invisible_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

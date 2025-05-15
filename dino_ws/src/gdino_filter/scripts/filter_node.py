# !/usr/bin/env python3

# FILTER NODE
# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge

# import pyrealsense2 as rs
# import cv2
# import torch
# import numpy as np
# from PIL import Image as PILImage
# import threading

# from groundingdino.util.inference import load_model, predict
# import groundingdino.datasets.transforms as T
# from torchvision.ops import box_convert
# from segment_anything import sam_model_registry, SamPredictor

# # ==== CONFIG ====
# CONFIG_PATH = "/root/vision_models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# WEIGHTS_PATH = "/root/weights/gdino/groundingdino_swint_ogc.pth"
# SAM_CHECKPOINT = "/root/weights/sam/sam_vit_b_01ec64.pth"
# TEXT_PROMPT = "a person"
# BOX_THRESHOLD = 0.3
# TEXT_THRESHOLD = 0.25

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bridge = CvBridge()

# # ==== ROS Node ====
# rospy.init_node('filter_node')
# image_pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=1)
# rospy.loginfo("Camera publisher started")

# # ==== Load Models ====
# gdino_model = load_model(CONFIG_PATH, WEIGHTS_PATH).to(device)
# sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(device)
# sam_predictor = SamPredictor(sam)

# # ==== RealSense Setup ====
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# pipeline.start(config)

# # ==== Transform ====
# transform = T.Compose([
#     T.RandomResize([800], max_size=1333),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])

# # ==== Live prompt update ====
# def prompt_listener():
#     global TEXT_PROMPT
#     while not rospy.is_shutdown():
#         user_input = input("New prompt (or blank to skip): ").strip()
#         if user_input:
#             TEXT_PROMPT = user_input
#             rospy.loginfo(f"[INFO] Updated prompt to: {TEXT_PROMPT}")

# threading.Thread(target=prompt_listener, daemon=True).start()

# # ==== Background Buffer ====
# background_buffer = None

# try:
#     while not rospy.is_shutdown():
#         frames = pipeline.wait_for_frames()
#         color_frame = frames.get_color_frame()
#         if not color_frame:
#             continue

#         frame = np.asanyarray(color_frame.get_data())
#         h, w, _ = frame.shape
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image_pil = PILImage.fromarray(frame_rgb)
#         transformed_image, _ = transform(image_pil, None)

#         with torch.no_grad():
#             boxes, logits, phrases = predict(
#                 model=gdino_model,
#                 image=transformed_image.to(device),
#                 caption=TEXT_PROMPT,
#                 box_threshold=BOX_THRESHOLD,
#                 text_threshold=TEXT_THRESHOLD,
#                 device=str(device),
#             )

#         boxes_scaled = boxes * torch.tensor([w, h, w, h], device=boxes.device)
#         xyxy_boxes = box_convert(boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

#         sam_predictor.set_image(frame_rgb)
#         mask_union = np.zeros((h, w), dtype=np.uint8)

#         for box in xyxy_boxes:
#             box[0] = max(0, box[0] - 10)
#             box[1] = max(0, box[1] - 10)
#             box[2] = min(w, box[2] + 10)
#             box[3] = min(h, box[3] + 10)

#             input_box = np.array(box).astype(int)
#             masks, _, _ = sam_predictor.predict(
#                 point_coords=None,
#                 point_labels=None,
#                 box=input_box[None, :],
#                 multimask_output=False,
#             )
#             mask = masks[0].astype(np.uint8)
#             mask_union = np.maximum(mask_union, mask)

#         if background_buffer is None:
#             background_buffer = np.zeros_like(frame)

#         background_buffer[mask_union == 0] = frame[mask_union == 0]
#         invisible_frame = frame.copy()

#         unknown_mask = np.all(background_buffer == 0, axis=2)
#         mask_to_inpaint = np.logical_and(mask_union == 1, unknown_mask)
#         invisible_frame[mask_union == 1] = background_buffer[mask_union == 1]

#         if np.any(mask_to_inpaint):
#             kernel = np.ones((15, 15), np.uint8)
#             mask_dilated = cv2.dilate(mask_to_inpaint.astype(np.uint8) * 255, kernel, iterations=2)
#             invisible_frame = cv2.inpaint(invisible_frame, mask_dilated, 5, cv2.INPAINT_TELEA)

#         # ==== ROS Publish ====
#         image_msg = bridge.cv2_to_imgmsg(invisible_frame, encoding="bgr8")
#         image_pub.publish(image_msg)

# finally:
#     pipeline.stop()

# NORMAL CAMERA
import rospy
import pyrealsense2 as rs
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

def main():
    rospy.init_node('camera_publisher', anonymous=True)
    pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)
    rate = rospy.Rate(30)
    bridge = CvBridge()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    rospy.loginfo("Camera publisher started")
    try:
        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            # img = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
            img = np.asanyarray(color_frame.get_data())
            # ros_image = bridge.cv2_to_imgmsg(img, encoding='rgb8')
            ros_image = bridge.cv2_to_imgmsg(img, encoding='bgr8')
            ros_image.header.stamp = rospy.Time.now()
            ros_image.header.frame_id = "camera_color_optical_frame"
            pub.publish(ros_image)
            rate.sleep()
    finally:
        pipeline.stop()

if __name__ == '__main__':
    main()

## WITH IMU
# import rospy
# import pyrealsense2 as rs
# from sensor_msgs.msg import Image, Imu
# from cv_bridge import CvBridge
# import numpy as np

# def main():
#     rospy.init_node('stereo_inertial_publisher', anonymous=True)

#     pub_left = rospy.Publisher('/camera/infra1/image_raw', Image, queue_size=10)
#     pub_right = rospy.Publisher('/camera/infra2/image_raw', Image, queue_size=10)
#     pub_imu = rospy.Publisher('/camera/imu', Imu, queue_size=100)

#     bridge = CvBridge()
#     rate = rospy.Rate(30)

#     # Configure streams
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # Left IR
#     config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # Right IR
#     config.enable_stream(rs.stream.accel)
#     config.enable_stream(rs.stream.gyro)

#     profile = pipeline.start(config)
#     # sensor = profile.get_device().first_imu_sensor()
#     # sensor.set_option(rs.option.enable_motion_correction, 1)

#     rospy.loginfo("Stereo-inertial publisher started")

#     try:
#         while not rospy.is_shutdown():
#             frames = pipeline.wait_for_frames()

#             # Stereo
#             ir1 = frames.get_infrared_frame(1)
#             ir2 = frames.get_infrared_frame(2)
#             if ir1 and ir2:
#                 img_left = np.asanyarray(ir1.get_data())
#                 img_right = np.asanyarray(ir2.get_data())

#                 stamp = rospy.Time.now()

#                 msg_left = bridge.cv2_to_imgmsg(img_left, encoding='mono8')
#                 msg_left.header.stamp = stamp
#                 msg_left.header.frame_id = "camera_left"

#                 msg_right = bridge.cv2_to_imgmsg(img_right, encoding='mono8')
#                 msg_right.header.stamp = stamp
#                 msg_right.header.frame_id = "camera_right"

#                 pub_left.publish(msg_left)
#                 pub_right.publish(msg_right)

#             # IMU
#             for imu_frame in frames:
#                 if imu_frame.is_motion_frame():
#                     imu_data = Imu()
#                     ts = rospy.Time.now()
#                     imu_data.header.stamp = ts
#                     imu_data.header.frame_id = "camera_imu"

#                     if imu_frame.get_profile().stream_type() == rs.stream.gyro:
#                         gyro = imu_frame.as_motion_frame().get_motion_data()
#                         imu_data.angular_velocity.x = gyro.x
#                         imu_data.angular_velocity.y = gyro.y
#                         imu_data.angular_velocity.z = gyro.z
#                     elif imu_frame.get_profile().stream_type() == rs.stream.accel:
#                         accel = imu_frame.as_motion_frame().get_motion_data()
#                         imu_data.linear_acceleration.x = accel.x
#                         imu_data.linear_acceleration.y = accel.y
#                         imu_data.linear_acceleration.z = accel.z

#                     pub_imu.publish(imu_data)

#             rate.sleep()

#     finally:
#         pipeline.stop()

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
# English comments requested for code.
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import onnxruntime as ort
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import time
from collections import deque
import threading
import queue

class LaneDetector(Node):

    def __init__(self):
        super().__init__('lane_detector')

        # ROS 2 Publishers
        self.raw_mask_pub = self.create_publisher(Image, 'limo/cv_package/ai_detection/lane_masks/raw', 10)
        self.image_pub = self.create_publisher(Image, 'limo/cv_package/ai_detection/lane_overlay', 10)
        
        self.bridge = CvBridge()

        # Telemetry control parameters
        self.declare_parameter('enable_telemetry', True)
        self.debug_telemetry = self.get_parameter('enable_telemetry').value
        self.frame_counter = 0

        # Best effort QoS matching high-rate video streams
        latest_frame_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscriber to Limo's camera
        self.declare_parameter('rgb_topic', '/rgb/image_raw')
        self.camera_topic = self.get_parameter('rgb_topic').value
        self.rgb_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            latest_frame_qos
        )

        # Telemetry metrics window
        self.window_size = 30
        self.telemetry_stats = {
            '0_transport_delay': deque(maxlen=self.window_size),
            '1_convert_time': deque(maxlen=self.window_size),
            '2_queue_waiting_time': deque(maxlen=self.window_size),
            '3_preprocess_time': deque(maxlen=self.window_size),
            '4_neural_inference': deque(maxlen=self.window_size),
            '5_msg_age_post_inference': deque(maxlen=self.window_size),
            '6_post_filter': deque(maxlen=self.window_size),
            '9_post_canvas': deque(maxlen=self.window_size),
            '11_ros_publish_enqueue': deque(maxlen=self.window_size),
            'total_pipeline': deque(maxlen=self.window_size),
            'async_encode_publish': deque(maxlen=self.window_size),
            '12_msg_age_final_publish': deque(maxlen=self.window_size),
        }

        # Threading queues
        self.inference_queue = queue.Queue(maxsize=1)
        self.pub_queue = queue.Queue(maxsize=1)

        # --- PATH RESOLUTION ---
        try:
            package_share_dir = get_package_share_directory('cv_package')
        except Exception:
            package_share_dir = os.path.dirname(os.path.realpath(__file__))

        default_model_path = os.path.join(package_share_dir, 'best.onnx')
        self.declare_parameter('model_path', default_model_path)
        self.model_path = self.get_parameter('model_path').value

        self.get_logger().info(f'Initializing ONNX Runtime Segmentation with model: {self.model_path}')
        
        # --- LOAD ONNX MODEL VIA ONNXRUNTIME ---
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Auto-detect CUDA provider if available in the Docker container
            available_providers = ort.get_available_providers()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
            
            self.ort_session = ort.InferenceSession(self.model_path, sess_options=opts, providers=providers)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.get_logger().info(f"ONNX Runtime session initialized with providers: {self.ort_session.get_providers()}")
        except Exception as e:
            self.get_logger().error(f"CRITICAL: Failed to load ONNX model via ONNXRuntime: {str(e)}")
            self.ort_session = None
            return

        # Start async worker threads
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.pub_thread = threading.Thread(target=self._publish_worker, daemon=True)
        self.inference_thread.start()
        self.pub_thread.start()

        self.get_logger().info("Clean & Fast 4-Class Segmenter Node started with Telemetry Pipeline!")

    def image_callback(self, msg):
        """
        Producer Callback: Non-blocking enqueue to prevent ROS executor queue delays.
        """
        t_now_ros = self.get_clock().now().nanoseconds / 1e9
        t_msg_ros = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        transport_delay = (t_now_ros - t_msg_ros) * 1000.0
        self.telemetry_stats['0_transport_delay'].append(transport_delay)

        # Drop older frames if worker thread is occupied
        if self.inference_queue.full():
            try:
                self.inference_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            time_entering = time.perf_counter()
            self.inference_queue.put_nowait((msg, time_entering))
        except queue.Full:
            pass

    def _inference_worker(self):
        while rclpy.ok():
            try:
                item = self.inference_queue.get(timeout=0.5)
                time_exiting = time.perf_counter()
            except queue.Empty:
                continue

            msg, time_entering = item
            queue_delay = (time_exiting - time_entering) * 1000.0
            self.telemetry_stats['2_queue_waiting_time'].append(queue_delay)

            t_convert_start = time.perf_counter()
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().error(f"CvBridge image conversion failed: {str(e)}")
                continue

            convert_time = (time.perf_counter() - t_convert_start) * 1000.0
            self.telemetry_stats['1_convert_time'].append(convert_time)

            self._process_frame(cv_image, msg.header.stamp)

    def _process_frame(self, cv_image, stamp):
        t_start = time.perf_counter()
        h, w, _ = cv_image.shape

        # Preprocessing
        t_prep_start = time.perf_counter()
        input_img = cv2.resize(cv_image, (448, 448))
        input_img = input_img[:, :, ::-1].transpose(2, 0, 1)   
        input_tensor = np.expand_dims(input_img, axis=0).astype(np.float32) / 255.0
        t_prep_end = time.perf_counter()
        self.telemetry_stats['3_preprocess_time'].append((t_prep_end - t_prep_start) * 1000.0)

        # Neural Inference
        t_inf_start = time.perf_counter()
        try:
            outputs = self.ort_session.run(None, {self.input_name: input_tensor})
            predictions = np.squeeze(outputs[0])
            proto = np.squeeze(outputs[1])
        except Exception as e:
            self.get_logger().error(f"Inference failed: {str(e)}")
            return
        t_inf_end = time.perf_counter()
        dt_inference = (t_inf_end - t_inf_start) * 1000.0
        self.telemetry_stats['4_neural_inference'].append(dt_inference)

        # Message age check right after inference
        t_now_post_inf = self.get_clock().now().nanoseconds / 1e9
        t_msg_post_inf = stamp.sec + stamp.nanosec * 1e-9
        msg_age_post_inference = (t_now_post_inf - t_msg_post_inf) * 1000.0
        self.telemetry_stats['5_msg_age_post_inference'].append(msg_age_post_inference)

        mask_overlay = np.zeros_like(cv_image)
        dt_filter = dt_canvas = 0.0

        t_post_start = time.perf_counter()
        if predictions.ndim == 2:
            predictions = predictions.T
            num_classes = 4
            scores = predictions[:, 4:4+num_classes]
            class_ids = np.argmax(scores, axis=1)
            confidences = scores[np.arange(len(predictions)), class_ids]
            
            mask_threshold = confidences > 0.1
            filtered_preds = predictions[mask_threshold]
            filtered_class_ids = class_ids[mask_threshold]

            t_filter_end = time.perf_counter()
            dt_filter = (t_filter_end - t_post_start) * 1000.0
            
            if len(filtered_preds) > 0:
                t_canvas_start = time.perf_counter()
                boxes = filtered_preds[:, 0:4]
                masks_coeffs = filtered_preds[:, 4+num_classes : 4+num_classes+32]
                proto_reshaped = proto.reshape(32, -1)
                
                raw_masks = np.matmul(masks_coeffs, proto_reshaped).reshape(-1, 112, 112)
                
                cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = np.clip((cx - bw / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                y1 = np.clip((cy - bh / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                x2 = np.clip((cx + bw / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                y2 = np.clip((cy + bh / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                
                low_res_canvases = {
                    0: np.zeros((112, 112), dtype=np.float32) - 10.0,
                    1: np.zeros((112, 112), dtype=np.float32) - 10.0,
                    2: np.zeros((112, 112), dtype=np.float32) - 10.0,
                    3: np.zeros((112, 112), dtype=np.float32) - 10.0
                }
                
                for i in range(len(filtered_preds)):
                    c_id = filtered_class_ids[i]
                    if c_id in low_res_canvases:
                        cropped_mask = np.zeros((112, 112), dtype=np.float32) - 10.0
                        cropped_mask[y1[i]:y2[i], x1[i]:x2[i]] = raw_masks[i, y1[i]:y2[i], x1[i]:x2[i]]
                        low_res_canvases[c_id] = np.maximum(low_res_canvases[c_id], cropped_mask)
                
                # Class 3: Road Surface (Dark Gray)
                if np.any(filtered_class_ids == 3):
                    combined_road = 1 / (1 + np.exp(-low_res_canvases[3]))
                    full_mask_road = cv2.resize(combined_road, (w, h)) > 0.5
                    mask_overlay[full_mask_road] = (80, 80, 80)
                
                # Class 0: Dashed Lines (Green)
                if np.any(filtered_class_ids == 0):
                    combined_dashed = 1 / (1 + np.exp(-low_res_canvases[0]))
                    full_mask_dashed = cv2.resize(combined_dashed, (w, h)) > 0.5
                    mask_overlay[full_mask_dashed] = (0, 255, 0)
                    
                # Class 2: Solid Lines (Red)
                if np.any(filtered_class_ids == 2):
                    combined_solid = 1 / (1 + np.exp(-low_res_canvases[2]))
                    full_mask_solid = cv2.resize(combined_solid, (w, h)) > 0.5
                    mask_overlay[full_mask_solid] = (0, 0, 255)

                # Class 1: Parking Lots (Blue)
                if np.any(filtered_class_ids == 1):
                    combined_c3 = 1 / (1 + np.exp(-low_res_canvases[1]))
                    full_mask_c3 = cv2.resize(combined_c3, (w, h)) > 0.5
                    mask_overlay[full_mask_c3] = (255, 0, 0)

                t_canvas_end = time.perf_counter()
                dt_canvas = (t_canvas_end - t_canvas_start) * 1000.0

        # Create blended overlay 
        overlay_image = cv2.addWeighted(cv_image, 1.0, mask_overlay, 0.6, 0)

        self.telemetry_stats['6_post_filter'].append(dt_filter)
        self.telemetry_stats['9_post_canvas'].append(dt_canvas)

        # Enqueue for publication worker
        t_pub_start = time.perf_counter()
        if self.pub_queue.full():
            try:
                self.pub_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.pub_queue.put_nowait((mask_overlay, overlay_image, stamp))
        except queue.Full:
            pass

        dt_pub_enqueue = (time.perf_counter() - t_pub_start) * 1000.0
        self.telemetry_stats['11_ros_publish_enqueue'].append(dt_pub_enqueue)

        t_end = time.perf_counter()
        dt_total_pipe = (t_end - t_start) * 1000.0
        self.telemetry_stats['total_pipeline'].append(dt_total_pipe)

        # Telemetry logging report
        self.frame_counter += 1
        if self.frame_counter % 30 == 0 and self.debug_telemetry:
            self._log_telemetry_report()

    def _publish_worker(self):
        while rclpy.ok():
            try:
                item = self.pub_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            mask_overlay, overlay_image, stamp = item
            t_pub_start = time.perf_counter()

            try:
                # Raw Mask Publish
                ros_mask_msg = self.bridge.cv2_to_imgmsg(mask_overlay, encoding='bgr8')
                ros_mask_msg.header.stamp = stamp
                self.raw_mask_pub.publish(ros_mask_msg)

                # Blended Overlay Image Publish
                ros_overlay_msg = self.bridge.cv2_to_imgmsg(overlay_image, encoding='bgr8')
                ros_overlay_msg.header.stamp = stamp
                self.image_pub.publish(ros_overlay_msg)

            except Exception as e:
                self.get_logger().error(f"Publishing failed: {str(e)}")

            dt_async_pub = (time.perf_counter() - t_pub_start) * 1000.0
            self.telemetry_stats['async_encode_publish'].append(dt_async_pub)

            t_now_final = self.get_clock().now().nanoseconds / 1e9
            t_msg_final = stamp.sec + stamp.nanosec * 1e-9
            final_age = (t_now_final - t_msg_final) * 1000.0
            self.telemetry_stats['12_msg_age_final_publish'].append(final_age)

    def _log_telemetry_report(self):
        avg_transport = np.mean(self.telemetry_stats['0_transport_delay'])
        avg_convert = np.mean(self.telemetry_stats['1_convert_time']) if len(self.telemetry_stats['1_convert_time']) > 0 else 0.0
        avg_queue_wait = np.mean(self.telemetry_stats['2_queue_waiting_time'])
        avg_prep = np.mean(self.telemetry_stats['3_preprocess_time']) if len(self.telemetry_stats['3_preprocess_time']) > 0 else 0.0
        avg_inference = np.mean(self.telemetry_stats['4_neural_inference'])
        avg_age_post_inf = np.mean(self.telemetry_stats['5_msg_age_post_inference'])
        avg_filter = np.mean(self.telemetry_stats['6_post_filter'])
        avg_canvas = np.mean(self.telemetry_stats['9_post_canvas'])
        avg_pub_enqueue = np.mean(self.telemetry_stats['11_ros_publish_enqueue'])
        avg_total = np.mean(self.telemetry_stats['total_pipeline'])
        avg_async_pub = np.mean(self.telemetry_stats['async_encode_publish']) if len(self.telemetry_stats['async_encode_publish']) > 0 else 0.0
        avg_final_age = np.mean(self.telemetry_stats['12_msg_age_final_publish']) if len(self.telemetry_stats['12_msg_age_final_publish']) > 0 else 0.0

        fps = 1000.0 / avg_total if avg_total > 0 else 0.0

        self.get_logger().info(
            f"\n"
            f"====== PERFORMANCE & LATENCY REPORT (AVG {self.window_size} frames) ======\n"
            f"-----------------------------------------\n"
            f"[TIMESTAMP LATENCY]\n"
            f" Camera -> Node ingress:          {avg_transport:.2f} ms\n"
            f" Frame age post inference:         {avg_age_post_inf:.2f} ms\n"
            f" Frame age final publish:          {avg_final_age:.2f} ms\n"
            f"-----------------------------------------\n"
            f"[EXECUTION BREAKDOWN]\n"
            f" Queue Waiting Delay:              {avg_queue_wait:.2f} ms\n"
            f" CvBridge Conversion:              {avg_convert:.2f} ms\n"
            f" Preprocessing (Resize/Norm):      {avg_prep:.2f} ms\n"
            f" Neural Inference (ONNX):          {avg_inference:.2f} ms\n"
            f" Post-Processing (Filter/NMS):     {avg_filter:.2f} ms\n"
            f" Canvas Rendering (Masks):         {avg_canvas:.2f} ms\n"
            f" ROS Publish Enqueue:              {avg_pub_enqueue:.2f} ms\n"
            f" Async Encode & Publish:           {avg_async_pub:.2f} ms\n"
            f"-----------------------------------------\n"
            f" TOTAL PIPELINE TIME:              {avg_total:.2f} ms\n"
            f" ESTIMATED INTERNAL FPS:           {fps:.1f}\n"
            f"=========================================\n"
        )

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Lane Segmenter Node.")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()
import argparse
import cv2
import torch
import time
import os
import datetime
from collections import defaultdict
import numpy as np
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection and Tracking Script")

    parser.add_argument('--video', type=str, default='your_video.mp4',
                        help='Path to the input video file.')
    parser.add_argument('--threshold', type=int, default=30,
                        help='Number of frames before declaring object as missing.')
    parser.add_argument('--device', type=str, choices=["cuda", "cpu", "mps", "auto"], default='auto',
                        help='Device for inference.')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model to use.')
    parser.add_argument('--resolution', type=int, default=640,
                        help='Frame width for inference.')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save frames when objects appear or disappear.')
    parser.add_argument('--half_precision', action='store_true',
                        help='Use half-precision (FP16) if available.')
    parser.add_argument('--output_path', type=str, default='outputs/',
                        help='Path to save the output video.')

    return parser.parse_args()

def select_device(device_choice):
    if device_choice == "auto":
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_choice

def save_frame(frame, frame_idx, obj_id, event, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{event}_ID{obj_id}_Frame{frame_idx}.jpg"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, frame)

def main():
    args = parse_args()

    device = select_device(args.device)

    # Load model
    model = YOLO(args.model)
    model.to(device)
    
    if args.half_precision and device == 'cuda':
        model.model.half()
    else:
        model.model.float()

    # Video input
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video: {args.video}")
        return

    # Output settings
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(args.resolution)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * (width / cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = args.output_path
    os.makedirs(output_folder, exist_ok=True)
    output_video_path = os.path.join(output_folder, f'output_detection_{timestamp}.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    tracked_objects = dict()
    frame_idx = 0
    seen_ids = set()

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        frame_idx += 1

        # Inference
        results = model.track(source=frame, persist=True, conf=0.5, iou=0.5, device=device)
        
        detections = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        classes = results[0].boxes.cls.cpu().numpy()      # class IDs
        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []

        current_ids = set()

        for idx, (box, cls) in enumerate(zip(detections, classes)):
            obj_id = int(ids[idx]) if len(ids) > idx else None
            if obj_id is not None:
                current_ids.add(obj_id)

                # New object detection
                if obj_id not in seen_ids:
                    color = (255, 0, 0)  # Blue for new object
                    label = f'New ID: {obj_id}'
                    if args.save_frames:
                        save_frame(frame, frame_idx, obj_id, 'appeared', 'detected_frames')
                    seen_ids.add(obj_id)
                else:
                    color = (0, 255, 0)  # Green for existing object
                    label = f'ID: {obj_id}'

                tracked_objects[obj_id] = {
                    'box': box,
                    'class': int(cls),
                    'last_seen': frame_idx
                }

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Detect missing objects
        missing_ids = []
        for obj_id, info in tracked_objects.items():
            if frame_idx - info['last_seen'] > args.threshold:
                missing_ids.append(obj_id)

        for missing_id in missing_ids:
            info = tracked_objects.get(missing_id)
            if info:
                box = info['box']
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)  # Red box
                cv2.putText(frame, f'Missing ID: {missing_id}', (x1, y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                if args.save_frames:
                    save_frame(frame, frame_idx, missing_id, 'disappeared', 'detected_frames')

        # FPS counter
        curr_time = time.time()
        fps_counter = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame, f'FPS: {fps_counter:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # Write to output
        out.write(frame)

        # Optional live display
        # cv2.imshow('Detection', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f'Output video saved at: {output_video_path}')

if __name__ == "__main__":
    main()

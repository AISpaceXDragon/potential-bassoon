import cv2
import time
import argparse
import os
import torch
from ultralytics import YOLO

# Check if running in Google Colab
try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO Object Tracking Script')
    parser.add_argument('--video', type=str, default='https://github.com/ultralytics/assets/releases/download/v0.0.101/bus.mp4',
                       help='Path to video file (default: YouTube bus video)')
    parser.add_argument('--env', type=str, choices=['colab', 'local'], default='local',
                       help='Environment where script is running (default: local)')
    parser.add_argument('--threshold', type=int, default=30,
                       help='Number of frames to wait before declaring an object missing (default: 30)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps', 'auto'], default='auto',
                       help='Device to run inference on (default: auto)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--resolution', type=int, default=640,
                       help='Frame width for inference (default: 640)')
    parser.add_argument('--save_frames', action='store_true',
                       help='Save frames when objects appear/disappear')
    parser.add_argument('--half_precision', action='store_true',
                       help='Use half precision for inference (default: True)')
    
    return parser.parse_args()

def initialize_display(env):
    global imshow
    if env == 'colab':
        from google.colab.patches import cv2_imshow
        imshow = cv2_imshow
    else:
        imshow = cv2.imshow

def main():
    args = parse_arguments()
    
    # Set up display function based on environment
    initialize_display(args.env)
    
    # Set device
    device = args.device.upper()
    if device == 'AUTO':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Check for valid device
    if device not in ['cuda', 'cpu', 'mps']:
        raise ValueError(f"Invalid device: {args.device}")
    
    # Initialize model with specified precision
    if args.half_precision and device == 'cuda':
        model = YOLO(args.model).to(device=device, dtype=torch.float16)
    else:
        model = YOLO(args.model).to(device=device)
    
    # Load video
    if not os.path.exists(args.video) and 'http' not in args.video:
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    
    # Initialize tracking variables
    object_first_seen = set()
    object_last_seen = {}
    new_object_count = 0
    missing_object_count = 0
    frame_number = 0
    MISSING_THRESHOLD = args.threshold
    
    # Initialize FPS counter
    start_time = time.time()
    frame_count = 0
    fps = 0
    
    # Create output directory for saving frames
    if args.save_frames:
        output_dir = 'detected_frames'
        os.makedirs(output_dir, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for faster inference
        if frame.shape[1] > args.resolution:
            frame_resize = cv2.resize(frame, (args.resolution, int(frame.shape[0] * (args.resolution / frame.shape[1]))))
        else:
            frame_resize = frame.copy()
        
        results = model.track(frame_resize, persist=True)
        
        current_ids = set()
        
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            
            for obj_id, cls_id in zip(ids, classes):
                current_ids.add(obj_id)
                
                if obj_id not in object_first_seen:
                    new_object_count += 1
                    print(f"[Frame {frame_number}] 🆕 New object detected! ID: {obj_id}, Class: {model.names[cls_id]}")
                    object_first_seen.add(obj_id)
                    object_last_seen[obj_id] = frame_number
                    
                    if args.save_frames:
                        cv2.imwrite(os.path.join(output_dir, f'new_object_{obj_id}_{frame_number}.png'), frame)
                else:
                    object_last_seen[obj_id] = frame_number
        
        # Check for missing objects
        missing_now = False
        for obj_id in list(object_first_seen):
            if obj_id not in current_ids:
                last_seen_frame = object_last_seen.get(obj_id, -1)
                if last_seen_frame != -1 and (frame_number - last_seen_frame) > MISSING_THRESHOLD:
                    missing_object_count += 1
                    print(f"[Frame {frame_number}] ❌ Object missing! ID: {obj_id}")
                    object_first_seen.remove(obj_id)
                    object_last_seen.pop(obj_id, None)
                    
                    if args.save_frames:
                        cv2.imwrite(os.path.join(output_dir, f'missing_object_{obj_id}_{frame_number}.png'), frame)
                    missing_now = True
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
        
        # Annotate frame
        annotated_frame = results[0].plot() if results else frame_resize
        
        # Draw FPS and counts on frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 255, 0)
        thickness = 2
        
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), font, font_scale, font_color, thickness)
        cv2.putText(annotated_frame, f"New Objects: {new_object_count}", (10, 60), font, font_scale, font_color, thickness)
        cv2.putText(annotated_frame, f"Missing Objects: {missing_object_count}", (10, 90), font, font_scale, font_color, thickness)
        
        # Display frame
        if missing_now and args.env == 'colab':
            imshow('Frame', annotated_frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        elif missing_now:
            cv2.imshow('Frame', annotated_frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            if args.env == 'colab':
                imshow('Frame', annotated_frame)
            else:
                cv2.imshow('Frame', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_number += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
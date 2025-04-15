import cv2
import pickle
import cvzone
import numpy as np
from typing import List, Tuple
from datetime import datetime

class ParkingConfig:
    PICKLE_FILE = 'CarParkPos.pkl'
    BOX_WIDTH, BOX_HEIGHT = 150, 200  # Must match setup
    DISPLAY_WIDTH, DISPLAY_HEIGHT = 960, 540
    
    # Image processing
    PARKING_THRESHOLD = 1000
    BLUR_KERNEL = (3, 3)
    ADAPTIVE_THRESH_BLOCK = 25
    ADAPTIVE_THRESH_C = 16
    MEDIAN_BLUR = 5
    DILATION_KERNEL = np.ones((3, 3), np.uint8)

def load_spots() -> List[Tuple[int, int]]:
    try:
        with open(ParkingConfig.PICKLE_FILE, 'rb') as f:
            spots = pickle.load(f)
            print(f"Loaded {len(spots)} parking spots")
            return spots
    except (FileNotFoundError, EOFError):
        print("Error: No parking spots found. Run setup first.")
        return []

def monitor_parking(video_path: str):
    spots = load_spots()
    if not spots:
        return

    cap = cv2.VideoCapture(video_path)
    success, first_frame = cap.read()
    if not success:
        print("Error: Could not read video")
        return
    
    original_height, original_width = first_frame.shape[:2]
    print(f"Monitoring at original resolution: {original_width}x{original_height}")
    
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Image processing pipeline
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, ParkingConfig.BLUR_KERNEL, 1)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, ParkingConfig.ADAPTIVE_THRESH_BLOCK,
            ParkingConfig.ADAPTIVE_THRESH_C
        )
        median = cv2.medianBlur(thresh, ParkingConfig.MEDIAN_BLUR)
        dilated = cv2.dilate(median, ParkingConfig.DILATION_KERNEL, iterations=1)
        
        free_spaces = 0
        output_frame = frame.copy()
        
        for pos in spots:
            x, y = pos  # Original coordinates
            w, h = ParkingConfig.BOX_WIDTH, ParkingConfig.BOX_HEIGHT
            
            # Verify the box is within frame bounds
            if (x + w > original_width or y + h > original_height):
                print(f"Warning: Spot at {x},{y} is outside frame bounds!")
                continue
                
            crop = dilated[y:y+h, x:x+w]
            count = cv2.countNonZero(crop)
            
            if count < ParkingConfig.PARKING_THRESHOLD:
                color = (0, 255, 0)
                label = "FREE"
                free_spaces += 1
            else:
                color = (0, 0, 255)
                label = "TAKEN"
            
            # Draw parking spot rectangle
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 3)
            
            # Add label with better visibility
            cvzone.putTextRect(
                output_frame, label, (x, y-10),
                scale=1.5, thickness=3, offset=10,
                colorR=color, colorT=(255, 255, 255),
                font=cv2.FONT_HERSHEY_DUPLEX,
                border=3, colorB=(0, 0, 0)
            )
        
        # Enhanced UI Elements
        # 1. Main counter display
        cvzone.putTextRect(
            output_frame,
            f" Available: {free_spaces}/{len(spots)}",
            (50, 70),
            scale=3,
            thickness=5,
            colorT=(255, 255, 255),
            colorR=(0, 100, 0),
            font=cv2.FONT_HERSHEY_DUPLEX,
            offset=20,
            border=5,
            colorB=(0, 200, 0)
        )
        # 2. Status bar at bottom
        status_bar = np.zeros((80, output_frame.shape[1], 3), dtype=np.uint8)
        availability_percent = free_spaces/len(spots)
        cv2.rectangle(status_bar, 
                     (0, 0), 
                     (int(output_frame.shape[1] * availability_percent), 80), 
                     (0, 255, 0), -1)
        
        # Add percentage text
        cv2.putText(status_bar, 
                   f"{int(availability_percent*100)}% Availability", 
                   (output_frame.shape[1]//2 - 180, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (255, 255, 255), 3)
        
        # 3. Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(status_bar, timestamp, 
                   (output_frame.shape[1] - 400, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Combine frames
        output_frame = cv2.vconcat([output_frame, status_bar])
        
        # Apply visual enhancements
        output_frame = cv2.convertScaleAbs(output_frame, alpha=1.1, beta=5)
        
        # Resize for display
        display_frame = cv2.resize(output_frame, 
                                 (ParkingConfig.DISPLAY_WIDTH, 
                                  ParkingConfig.DISPLAY_HEIGHT + 80))
        
        # Display window
        cv2.namedWindow("Smart Parking Monitor", cv2.WINDOW_NORMAL)
        cv2.imshow("Smart Parking Monitor", display_frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'newpark.mp4'
    monitor_parking(video_path)
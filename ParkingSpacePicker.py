import cv2
import pickle
from typing import List, Tuple

class ParkingConfig:
    PICKLE_FILE = 'CarParkPos.pkl'
    BOX_WIDTH, BOX_HEIGHT = 150, 200  # Original video box size
    DISPLAY_WIDTH, DISPLAY_HEIGHT = 960, 540  # Display size

def load_spots() -> List[Tuple[int, int]]:
    try:
        with open(ParkingConfig.PICKLE_FILE, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return []

def save_spots(spots: List[Tuple[int, int]]) -> None:
    with open(ParkingConfig.PICKLE_FILE, 'wb') as f:
        pickle.dump(spots, f)

def setup_parking_spots(video_path: str):
    posList = load_spots()
    
    cap = cv2.VideoCapture(video_path)
    success, original_img = cap.read()
    cap.release()
    
    if not success:
        raise RuntimeError("Failed to load video or read frame.")

    original_height, original_width = original_img.shape[:2]
    print(f"Original video size: {original_width}x{original_height}")

    # Calculate scaling factors
    width_scale = ParkingConfig.DISPLAY_WIDTH / original_width
    height_scale = ParkingConfig.DISPLAY_HEIGHT / original_height
    
    display_img = cv2.resize(original_img, (ParkingConfig.DISPLAY_WIDTH, ParkingConfig.DISPLAY_HEIGHT))

    def mouse_click(events, x, y, flags, params):
        nonlocal posList
        # Convert display coordinates back to original video coordinates
        orig_x = int(x / width_scale)
        orig_y = int(y / height_scale)
        
        if events == cv2.EVENT_LBUTTONDOWN:
            posList.append((orig_x, orig_y))
            print(f"Added spot at original coordinates: ({orig_x}, {orig_y})")
        elif events == cv2.EVENT_RBUTTONDOWN:
            for i, pos in enumerate(posList):
                px, py = pos
                # Check if click is within any existing box (original coordinates)
                if (px <= orig_x <= px + ParkingConfig.BOX_WIDTH and 
                    py <= orig_y <= py + ParkingConfig.BOX_HEIGHT):
                    posList.pop(i)
                    print(f"Removed spot at: ({px}, {py})")
                    break
        save_spots(posList)

    cv2.namedWindow("Parking Spot Setup")
    cv2.setMouseCallback("Parking Spot Setup", mouse_click)

    while True:
        img_display = display_img.copy()
        
        # Draw all spots (convert original coords to display coords)
        for pos in posList:
            display_x = int(pos[0] * width_scale)
            display_y = int(pos[1] * height_scale)
            display_w = int(ParkingConfig.BOX_WIDTH * width_scale)
            display_h = int(ParkingConfig.BOX_HEIGHT * height_scale)
            
            cv2.rectangle(img_display, 
                         (display_x, display_y),
                         (display_x + display_w, display_y + display_h),
                         (255, 0, 255), 2)
        
        cv2.imshow("Parking Spot Setup", img_display)
        if cv2.waitKey(1) == 27:  # ESC key
            break

    cv2.destroyAllWindows()
    print(f"Setup complete. Saved {len(posList)} spots at original resolution.")

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'newpark.mp4'
    setup_parking_spots(video_path)
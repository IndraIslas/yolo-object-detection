import cv2
import datetime

def save_frame(frame):
    # Get current datetime to use in filename
    now = datetime.datetime.now()
    filename = f"saved_frames/frame_{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Saved {filename}")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    rewind_seconds = 2
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video or error occurred.")
                break
            
            cv2.imshow('Video', frame)
        
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_frame(frame)
        elif key == ord('a'):
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            new_time = max(0, current_time - (rewind_seconds * 1000))
            cap.set(cv2.CAP_PROP_POS_MSEC, new_time)
        elif key == 32: # Space bar
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()

video_path = '../Videos/video8.mov'
process_video(video_path)

from pathlib import Path
import cv2

# Path to the directory containing the mp4 video
video_dir = Path('.')

for video_file in video_dir.glob('*.mp4'):
    # Create a subdirectory with the same name in .cache
    cache_dir = Path('.cache') / video_file.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(str(video_file))

    # Read and save each frame as a jpg image
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_file = cache_dir / f'frame_{frame_count}.jpg'
        cv2.imwrite(str(frame_file), frame)
        frame_count += 1

    # Release the video capture object
    cap.release()
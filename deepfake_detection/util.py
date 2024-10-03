import cv2
import torch
import numpy as np

def deepfake_preprocess(video_path: str, num_frames: int = 30) -> torch.Tensor:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Set the frame position
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]  # Get the first detected face
                frame = frame[y:y+h, x:x+w]  # Crop the face
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # Resize frame to model input size
                frames.append(frame.astype(np.float32))  # Convert to float32
    
    cap.release()

    # If fewer valid frames were read, duplicate the last frame until the required number is reached
    while len(frames) < num_frames:
        frames.append(frames[-1])  # Duplicate the last valid frame

    # Convert to tensor and normalize
    video_tensor = torch.stack([torch.tensor(f) for f in frames]).permute(0, 3, 1, 2)  # Shape: [num_frames, 3, 224, 224]
    
    # Normalize the tensor (if needed)
    video_tensor /= 255.0  # Scale pixel values to [0, 1]

    return video_tensor
import cv2

# Function to compute histogram similarity between two frames
def compute_histogram_similarity(frameA, frameB):
    histA = cv2.calcHist([frameA], [0], None, [256], [0, 256])
    histB = cv2.calcHist([frameB], [0], None, [256], [0, 256])
    histA = cv2.normalize(histA, histA).flatten()
    histB = cv2.normalize(histB, histB).flatten()
    # Compare histograms using correlation
    return cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)


# Main function to process the videos
def process_videos(video_path1, video_path2):
    # Load the two videos
    video1 = cv2.VideoCapture(video_path1)
    video2 = cv2.VideoCapture(video_path2)

    total_ssim_score = 0
    total_mse_score = 0
    total_histogram_score = 0
    frame_count = 0
    true_frames = 0  # Counter for frames with similarity > 70%

    # Read frames from both videos
    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        # If either video has ended, stop processing
        if not ret1 or not ret2:
            break

        # Ensure both frames are the same size (resize if necessary)
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # Compute histogram similarity
        hist_score = compute_histogram_similarity(frame1, frame2)
        total_histogram_score += hist_score

        frame_count += 1

    if frame_count == 0:
        print("No frames were processed.")
        return

    avg_histogram = total_histogram_score / frame_count

    # Release video objects
    video1.release()
    video2.release()
    
    return avg_histogram
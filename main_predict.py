from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
import cv2
import torch
import numpy as np
from deepfake_detection.model_builder import CNNLSTM
from deepfake_detection.util import deepfake_preprocess
from similarity_detection.util import process_videos

app = FastAPI()

class DeepfakePredictionResponse(BaseModel):
    prediction: str
    
class SimilarityPredictionResponse(BaseModel):
    average_histogram_sim: float
    prediction: str

@app.post("/predict-deepfake", response_model=DeepfakePredictionResponse)
async def predict_deepfake(file: UploadFile = File(...)):
    
    model_weight_path = "deepfake_detection/model/ethkl_cnn_lstm_vid_30_epochs_20.pt"
    # Load the model
    model = CNNLSTM()
    model.load_state_dict(torch.load(model_weight_path))
    model.eval() 

    video_path = f"temp_{file.filename}"
    
    # Save the uploaded video to a temporary location
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # Process the video and prepare input for the model
    video_tensor = deepfake_preprocess(video_path)

    # Add batch dimension
    video_tensor = video_tensor.unsqueeze(0)  # Shape: [1, num_frames, 3, 224, 224]

    # Move to device if using GPU
    if torch.cuda.is_available():
        video_tensor = video_tensor.to("cuda")
        model.to("cuda")

    # Make prediction
    with torch.no_grad():
        outputs = model(video_tensor)
        print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
    
    # Map prediction to label
    prediction_label = "FAKE" if predicted.item() == 1 else "REAL"

    # Optionally, clean up the temporary video file
    Path(video_path).unlink(missing_ok=True)

    return DeepfakePredictionResponse(prediction=prediction_label)

@app.post("/predict-similarity", response_model=SimilarityPredictionResponse)
async def predict_similarity(files: List[UploadFile]):
    
    # if only one video is uploaded, return error
    if len(files) != 2:
        # return error status code with message
        raise HTTPException(status_code=400, detail="Please upload two videos for comparison.")
    
    video_path1 = f"temp_{files[0].filename}"
    video_path2 = f"temp_{files[1].filename}"
    
    # Save the uploaded videos to temporary locations
    with open(video_path1, "wb") as f:
        f.write(await files[0].read())
    with open(video_path2, "wb") as f:
        f.write(await files[1].read())

    # Process the videos and compute similarity scores
    average_histogram_sim = process_videos(video_path1, video_path2)
    prediction = None
    
    # Make prediction based on similarity scores
    if average_histogram_sim > 0.9:
        prediction = "SIMILAR"
    else:
        prediction = "DIFFERENT"

    # Optionally, clean up the temporary video files
    Path(video_path1).unlink(missing_ok=True)
    Path(video_path2).unlink(missing_ok=True)

    return SimilarityPredictionResponse(
        average_histogram_sim=average_histogram_sim,
        prediction=prediction
    )
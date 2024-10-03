# Deepfake Video Prediction API

This project provides a FastAPI application to predict whether a video is real or fake using a CNN-LSTM model.

## Prerequisites

Ensure you have Python 3.7 or higher installed on your machine. You will also need `pip` for package management.

## Setup Instructions

### 1. Create a Virtual Environment

Create a virtual environment to manage your project's dependencies:

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

```bash
# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model from Drive

Download the trained model from [My Google Drive](https://drive.google.com/drive/folders/1RTJGPMsKU11JMZdMsiaDdkXqdtA0befJ?usp=sharing) and put it under the `/model` directory, create one if not exist. Do not change the file name.

### 5. Run the FastAPI Application
```bash
uvicorn main_predict:app --reload
```

The API Docs will be available at http://127.0.0.1:8000/docs.

## Endpoints Guide

### 1. Using the `/predict-deepfake` Endpoint

To predict if a video is deepfake or not, you can send a video file to the `/predict-deepfake` endpoint. Here’s an example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@/path/to/your/video.mp4"
```
Replace `/path/to/your/video.mp4` with the actual path to your video file.

### API Response

The response will be a JSON object indicating whether the video is "REAL" or "FAKE":

```json
{
  "prediction": "FAKE"
}
```
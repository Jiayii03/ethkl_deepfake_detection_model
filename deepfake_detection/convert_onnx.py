from model_builder import CNNLSTM
import torch

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and map it to the appropriate device
model = CNNLSTM()
model.load_state_dict(torch.load('model/ethkl_cnn_lstm_vid_30_epochs_20.pt', map_location=device))
model.to(device)
model.eval()

# Create dummy input and map it to the correct device
dummy_input = torch.randn(1, 5, 3, 224, 224).to(device)

# Export the model to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "deepfake_detection_cnn_lstm.onnx", 
    verbose=False, 
    opset_version=11
)

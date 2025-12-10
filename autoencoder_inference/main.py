import cv2
import hydra
import mlflow.pytorch
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
# Ensure model definition is available for pickle
import model as model_def

def load_model(cfg: DictConfig):
    """
    Load PyTorch model from MLflow model registry.
    """
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    
    model_uri = f"models:/{cfg.mlflow.model_name}/{cfg.mlflow.model_version}"
    print(f"Loading model from: {model_uri}")
    
    try:
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check if MLflow server is running and the model name/version is correct.")
        return None

def preprocess_frame(frame, input_height, input_width, input_channels, device):
    """
    Preprocess the camera frame for the model.
    Resizes, normalizes, and converts to tensor.
    """
    # Resize
    resized_frame = cv2.resize(frame, (input_width, input_height))
    
    # Handle Channels
    if input_channels == 1:
        # Convert BGR to Grayscale
        processed_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        # ToTensor handles (H, W) -> (1, H, W) conversion automatically
    else:
        # Default to 3 channels (RGB)
        # Convert BGR to RGB
        processed_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Transform to tensor and normalize to [0, 1]
    # ToTensor converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
    # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    input_tensor = transform(processed_frame)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_tensor.to(device)

def nothing(x):
    pass

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(cfg)
    if model is None:
        return

    model.to(device)

    # Initialize camera
    cap = cv2.VideoCapture(cfg.inference.camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {cfg.inference.camera_id}")
        return

    window_name = cfg.inference.window_name
    cv2.namedWindow(window_name)

    # Create trackbar for threshold
    initial_trackbar_val = int(cfg.inference.initial_threshold * 1000)
    cv2.createTrackbar("Threshold (x1000)", window_name, initial_trackbar_val, 1000, nothing)

    print("Starting inference loop. Press 'q' to quit.")
    
    scores = [] # List to store anomaly scores

    # Start MLflow run for this inference session
    try:
        mlflow.set_tracking_uri(cfg.mlflow.uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        
        # Using a context manager to ensure run is ended
        with mlflow.start_run(run_name="inference_session") as run:
            print(f"MLflow Run ID: {run.info.run_id}")
            
            # Log parameters
            mlflow.log_params({
                "input_height": cfg.model.input_height,
                "input_width": cfg.model.input_width,
                "input_channels": getattr(cfg.model, "input_channels", 3),
                "crop_height": getattr(cfg.inference, "crop_height", "full"),
                "crop_width": getattr(cfg.inference, "crop_width", "full"),
                "camera_id": cfg.inference.camera_id,
                "model_name": cfg.mlflow.model_name,
                "model_version": cfg.mlflow.model_version
            })

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break

                    # Get current threshold from trackbar
                    trackbar_val = cv2.getTrackbarPos("Threshold (x1000)", window_name)
                    threshold = trackbar_val / 1000.0

                    # --- Crop Logic ---
                    h, w, _ = frame.shape
                    crop_h = getattr(cfg.inference, "crop_height", h)
                    crop_w = getattr(cfg.inference, "crop_width", w)
                    
                    crop_h = min(crop_h, h)
                    crop_w = min(crop_w, w)
                    
                    y1 = (h - crop_h) // 2
                    x1 = (w - crop_w) // 2
                    y2 = y1 + crop_h
                    x2 = x1 + crop_w
                    
                    cropped_frame = frame[y1:y2, x1:x2]
                    
                    # Preprocess (use cropped_frame)
                    input_channels = getattr(cfg.model, "input_channels", 3)
                    input_tensor = preprocess_frame(cropped_frame, cfg.model.input_height, cfg.model.input_width, input_channels, device)

                    # Inference
                    with torch.no_grad():
                        reconstructed = model(input_tensor)
                    
                    # Calculate Anomaly Score (MSE Loss)
                    loss = F.mse_loss(reconstructed, input_tensor)
                    anomaly_score = loss.item()
                    scores.append(anomaly_score)

                    # Determine Anomaly
                    is_anomaly = anomaly_score > threshold
                    
                    # Visualization
                    status_color = (0, 0, 255) if is_anomaly else (0, 255, 0) # Red if anomaly, Green if normal
                    status_text = "ANOMALY" if is_anomaly else "NORMAL"
                    
                    display_frame = frame.copy()
                    
                    # Draw crop rectangle (Red)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Overlay info
                    cv2.putText(display_frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                    cv2.putText(display_frame, f"Score: {anomaly_score:.6f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Threshold: {threshold:.3f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow(window_name, display_frame)

                    if cv2.waitKey(1) == ord('q'):
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()
                
                # Calculate and log statistics
                if scores:
                    scores_np = np.array(scores)
                    mean_score = np.mean(scores_np)
                    std_score = np.std(scores_np)
                    max_score = np.max(scores_np)
                    
                    print("\n" + "="*30)
                    print("Inference Session Results:")
                    print(f"Frames Processed: {len(scores)}")
                    print(f"Mean Score: {mean_score:.6f}")
                    print(f"Std Score : {std_score:.6f}")
                    print(f"Max Score : {max_score:.6f}")
                    print("="*30 + "\n")
                    
                    mlflow.log_metric("inference_score_mean", mean_score)
                    mlflow.log_metric("inference_score_std", std_score)
                    mlflow.log_metric("inference_score_max", max_score)
                    mlflow.log_metric("frames_processed", len(scores))
                else:
                    print("\nNo frames processed.")

    except Exception as e:
        print(f"MLflow Run Error: {e}")

if __name__ == "__main__":
    main()

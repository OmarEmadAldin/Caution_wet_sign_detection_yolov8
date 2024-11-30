# Caution_wet_sign_detection_yolov8
Detection of the wet sign to be used in some applications for autonomous vechiles or similar things.

## 1. Dataset Preparation

- **Dataset Splits**: 
  - Split the dataset into three parts: `train`, `valid`, and `test`.
- **Annotations**:
  - Ensure all images are annotated correctly in YOLO format (using tools like LabelImg or Roboflow).
- **Configuration File**: 
  - Create a `data.yaml` file with the following structure:
    ```yaml
    train: path/to/train/images
    val: path/to/valid/images
    test: path/to/test/images
    nc: <number_of_classes>
    names: [class_1, class_2, ...]
    ```

---

## 2. Training the YOLOv8 Model

Use the command below to train the YOLOv8 model:

```bash
yolo train data=data.yaml model=yolov8n.yaml epochs=100 imgsz=640
```

- **Parameters**
    - data: Path to the data.yaml file.
    - model: Model type (e.g., yolov8n, yolov8s, yolov8m, etc.).
    - epochs: Number of training epochs.
    - imgsz: Image size used for training.

## 3. Monitoring Training
    -Monitor the training logs for:
      -Loss values: Should decrease over epochs.
      - Validation metrics: Keep an eye on mAP (mean Average Precision).
    b. Use visualization tools like TensorBoard for detailed tracking.

## 4. Model Evaluation
Evaluate the trained model on the test set:
```bash
yolo val data=data.yaml model=path/to/best.pt
```
- Outputs:
        - mAP (mean Average Precision)
        - Precision, Recall, and other metrics.
  د
## 5. Model Inference

Use the trained YOLOv8 model to detect objects in images or videos.

### Single Image Inference
Run the following command to perform inference on a single image:
```bash
yolo predict source=your_image.jpg model=path/to/best.pt
```

### Video or Webcam Inference
To perform inference on a video file or a webcam stream:
```bash
yolo predict source=your_video.mp4 model=path/to/best.pt
```

### Parameters:
- source: Path to the input image, video, or 0 for the webcam.
- model: Path to the trained model weights (best.pt).
### Outputs:
- Annotated images or videos showing detected objects, bounding boxes, and confidence scores.
- Results are saved in the runs/predict directory by default.


## 6. Deployment
### Real-Time Deployment:
Use the trained model (best.pt) in real-time applications:
      - Flask App: For web-based object detection.
      - ROS Node: For robotic systems.

Example for Loading and Using the Model in Python:
```python
from ultralytics import YOLO
# Load the trained model
model = YOLO("path/to/best.pt")
# Perform inference
results = model("path/to/image.jpg")
# Display results
results.show()
```

## 7. Additional Notes
- Ensure the hardware (GPU/TPU) is optimized for training and inference.
- Save the best-performing model for deployment (best.pt).


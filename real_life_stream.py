import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/home/omar/State of The Art/Caution_wet_sign_detection_yolov8/runs/detect/train3/weights/best.pt') 
# Open the video stream (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    # Perform inference
    results = model(frame)
    
    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes.xyxy  # Extract the bounding boxes in xyxy format
        confidences = result.boxes.conf  # Extract the confidence scores
        class_ids = result.boxes.cls  # Extract the class IDs

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            label = f"{model.names[int(class_id)]} {confidence:.2f}"
            
            # Draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Draw label
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Sign Detection with YOLOv8', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

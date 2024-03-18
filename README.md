# Real-Time-Object-Facial-Recognition-System
This Python-based system integrates real-time object and facial recognition using the MTCNN (Multi-task Cascaded Convolutional Networks) for facial detection and YOLO (You Only Look Once) for object detection. 
It employs OpenCV for video capture and processing, showcasing the power of combining these technologies for sophisticated visual recognition tasks. The system operates on live video feed from a webcam, applying motion detection as a preprocessing step to optimize facial recognition, and performs object detection across the entire video frame.

## Technical Overview

### Core Components

- **OpenCV (cv2)**: Used for video capture, image processing, and display functionalities.
- **MTCNN**: A neural network specialized in detecting faces efficiently, focusing on regions where motion is detected to optimize processing.
- **YOLOv3**: A deep learning model for fast and accurate object detection, configured with pre-trained weights and configuration files.

### Workflow

1. **Video Capture Initialization**: A video stream is captured from the default webcam.
2. **MTCNN Detector Initialization**: Prepares the MTCNN model for facial recognition.
3. **YOLO Model Initialization**: Loads YOLOv3 with its corresponding weights and configuration, preparing it for object detection.
4. **Real-time Processing Loop**:
   - **Motion Detection**: Identifies changes between consecutive video frames to pinpoint regions of interest, reducing computational overhead by limiting facial recognition to areas where motion is observed.
   - **Facial Recognition**: Applies MTCNN within motion-detected areas to recognize faces.
   - **Object Detection**: Uses YOLO to identify and classify objects in the entire frame, independent of motion detection, ensuring comprehensive scene analysis.

### Detailed Implementation

- **Motion Detection**:
  - Converts each frame to grayscale and applies Gaussian blur to smooth out the image, reducing noise and detail.
  - Computes the absolute difference between the current and previous frames to detect motion, followed by thresholding to identify significant changes.
  - Dilates the thresholded image to fill in gaps, making object contours more pronounced, and then finds contours to identify distinct areas of motion.

- **Facial Recognition**:
  - For each motion-detected region, MTCNN is used to detect faces. Detected faces are highlighted with rectangles drawn on the frame.

- **Object Detection (YOLO)**:
  - Prepares the video frame as input to YOLO by converting it into a blob, resizing, and normalizing it.
  - Feeds the blob into YOLO, which returns detections including object class IDs and bounding boxes.
  - Processes YOLO's output using non-max suppression to eliminate overlapping boxes, ensuring each detected object is represented once.
  - Annotates detected objects on the frame, displaying their class labels.

### Utilities

- **YOLO Output Processing**: Converts raw detections into a list of bounding boxes, confidences, and class IDs, applying non-max suppression to filter out overlapping boxes for clarity in visualization.
- **Layer Names Handling**: Adapts to different versions of OpenCV, ensuring compatibility when retrieving the names of output layers from YOLO.

### Execution

To run the system, ensure you have the necessary models (`yolov3.weights`, `yolov3.cfg`), the class labels file (`coco.names`), and the required Python packages installed. Once started, the system processes video from your webcam in real-time, displaying the annotated video feed in a window. Press 'q' to quit the application.

## Dependencies

- Python 3.x
- OpenCV (cv2)
- MTCNN
- NumPy

## Files Required

- `yolov3.weights`: Pre-trained weights for YOLOv3.
- `yolov3.cfg`: Configuration file for YOLOv3 model.
- `coco.names`: Text file containing the labels of the classes that YOLO can detect.

This system demonstrates the integration of motion detection, facial recognition, and object detection in a single application, illustrating the potential for real-world surveillance, security, and interactive projects. The implementation details provide insights into handling video data, applying advanced neural network models, and optimizing recognition tasks for real-time performance.

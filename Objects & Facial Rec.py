import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MTCNN detector
detector = MTCNN()

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()

# Fix for different OpenCV versions' getUnconnectedOutLayers() output
if len(unconnected_out_layers.shape) == 2 and unconnected_out_layers.shape[1] == 1:
    # Newer OpenCV versions
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers.flatten()]
else:
    # Older OpenCV versions or unexpected format
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to process YOLO outputs
def process_yolo_output(outs, width, height):
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Ensure 'indices' is an array before flatten
    if isinstance(indices, tuple):
        indices = indices[0]  # Assuming the tuple contains the array as its first element

    final_boxes = []
    for i in indices:
        i = i[0] if isinstance(i, np.ndarray) else i  # Ensuring 'i' is an integer
        final_boxes.append(boxes[i])

    return final_boxes, class_ids


# Initial frame for motion detection
ret, prev_frame = cap.read()
gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
gray_prev_frame = cv2.GaussianBlur(gray_prev_frame, (21, 21), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Motion Detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    frame_diff = cv2.absdiff(gray_prev_frame, gray_frame)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Facial Recognition within Motion Detected Areas
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Adjust based on your requirement
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        roi_frame = frame[y:y+h, x:x+w]
        faces = detector.detect_faces(roi_frame)
        for face in faces:
            bx, by, bwidth, bheight = face['box']
            cv2.rectangle(frame, (x + bx, y + by), (x + bx + bwidth, y + by + bheight), (0, 255, 0), 2)

    # Object Detection on the entire frame using YOLO
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    boxes, class_ids = process_yolo_output(outs, width, height)
    for i, box in enumerate(boxes):
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    gray_prev_frame = gray_frame.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

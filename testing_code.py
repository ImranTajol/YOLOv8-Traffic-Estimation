
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Load the best fine-tuned YOLOv8 model
best_model = YOLO('models/best.pt')

# model = YOLO('yolov8n.pt')
# yaml_file_path = 'vehicle_with_class\data.yaml'

# Define the threshold for considering traffic as heavy
heavy_traffic_threshold = 5

# Define the vertices for the quadrilaterals
vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

# Define the vertical range for the slice and lane threshold
x1, x2 = 325, 635

# Define the positions for the text annotations on the image
text_position_left_lane_pct = (0.01, 0.05)  # 1% from the left, 5% from the top
intensity_position_left_lane_pct = (0.01, 0.1)  # 1% from the left, 10% from the top

text_position_right_lane_pct = (0.60, 0.05)  # 85% from the left, 10% from the top
intensity_position_right_lane_pct = (0.6, 0.1)  # 85% from the left, 20% from the top

# Define font, scale, and colors for the annotations
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)    # White color for text
background_color = (0, 0, 255)  # Red background for text
        
# Open the video
cap = cv2.VideoCapture('videos\\sample_video.mp4')
ori_mask = cv2.imread('images\\Road_Mask_Test.jpg')
# cap = cv2.VideoCapture(1)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('processed_sample_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, ori_size = cap.read()
    if ret:
        # Create a copy of the original frame to modify
        frame = cv2.resize(ori_size,(1280,720))

        # Get the width and height of the video
        # frame_width = int(frame.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(frame.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        window_area = frame_width * frame_height

        lane_threshold = frame_width/2

        mask = cv2.resize(ori_mask, (frame_width,frame_height))
        
        # detection_frame = frame.copy()

        filtered_frame = cv2.bitwise_and(frame, mask)
    
        # Black out the regions outside the specified vertical range
        # detection_frame[:x1, :] = 0  # Black out from top to x1
        # detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the frame
        
        # Perform inference on the modified frame
        results = best_model.predict(filtered_frame, imgsz=640, conf=0.4)
        post_predict_frame = results[0].plot(line_width=1) 

        constant_width = 50
        constant_height = 50

         # Draw bounding boxes and labels on the main frame
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box[:6]
                if int(cls) == 2:  # Class 2 corresponds to 'car' in COCO dataset
                    # x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    # box_area = (x2-x1)*(y2-y1)
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # label = f'Car'
                    # cv2.putText(frame, label, (x1, y1 - 10), font, 0.9, (36, 255, 12), 2)

                    # Calculate the center of the detected bounding box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Calculate the coordinates of the constant-sized bounding box
                    new_x1 = center_x - constant_width // 2
                    new_y1 = center_y - constant_height // 2
                    new_x2 = center_x + constant_width // 2
                    new_y2 = center_y + constant_height // 2

                    # Ensure the bounding box is within the frame boundaries
                    new_x1 = max(new_x1, 0)
                    new_y1 = max(new_y1, 0)
                    new_x2 = min(new_x2, frame.shape[1])
                    new_y2 = min(new_y2, frame.shape[0])
                    box_area = (new_x2-new_x1)*(new_y2-new_y1)

                    # Draw the constant-sized bounding box
                    cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
                    label = 'Car'
                    cv2.putText(frame, label, (new_x1, new_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


        
        # Draw the quadrilaterals on the processed frame
        cv2.polylines(frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Retrieve the bounding boxes from the results
        bounding_boxes = results[0].boxes

        # Initialize counters for vehicles in each lane
        vehicles_in_left_lane = 0
        vehicles_in_right_lane = 0
        total_box_area = 0

        # Loop through each bounding box to count vehicles in each lane
        for box in bounding_boxes.xyxy:
            vehicles_in_left_lane += 1
            total_box_area += box_area
            vehicles_in_right_lane += 1
            total_box_area += box_area
                
        # Determine the traffic intensity for the left lane
        traffic_intensity_left = "Heavy" if vehicles_in_left_lane > heavy_traffic_threshold else "Smooth"

        # Calculate the actual positions based on the frame size
        text_position_left_lane = (int(frame_width * text_position_left_lane_pct[0]), int(frame_height * text_position_left_lane_pct[1]))
        text_position_right_lane = (int(frame_width * text_position_right_lane_pct[0]), int(frame_height * text_position_right_lane_pct[1]))
        intensity_position_left_lane = (int(frame_width * intensity_position_left_lane_pct[0]), int(frame_height * intensity_position_left_lane_pct[1]))
        intensity_position_right_lane = (int(frame_width * intensity_position_right_lane_pct[0]), int(frame_height * intensity_position_right_lane_pct[1]))

        # Add a background rectangle for the left lane vehicle count
        cv2.rectangle(frame, (text_position_left_lane[0]-10, text_position_left_lane[1] - 25), 
                      (text_position_left_lane[0] + 460, text_position_left_lane[1] + 10), background_color, -1)

        # Add the vehicle count text on top of the rectangle for the left lane
        cv2.putText(frame, f'Total boxes area: {total_box_area}', text_position_left_lane, 
                    font, font_scale, font_color, 1, cv2.LINE_AA)

        # Add a background rectangle for the left lane traffic intensity
        cv2.rectangle(frame, (intensity_position_left_lane[0]-10, intensity_position_left_lane[1] - 25), 
                      (intensity_position_left_lane[0] + 460, intensity_position_left_lane[1] + 10), background_color, -1)

        # Add the traffic intensity text on top of the rectangle for the left lane
        cv2.putText(frame, f'Window Size: {window_area}', intensity_position_left_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Add a background rectangle for the right lane vehicle count
        cv2.rectangle(frame, (text_position_right_lane[0]-10, text_position_right_lane[1] - 25), 
                      (text_position_right_lane[0] + 460, text_position_right_lane[1] + 10), background_color, -1)

        # Add the vehicle count text on top of the rectangle for the right lane
        cv2.putText(frame, f'Total Vehicles: {vehicles_in_right_lane}', text_position_right_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Add a background rectangle for the right lane traffic intensity
        cv2.rectangle(frame, (intensity_position_right_lane[0]-10, intensity_position_right_lane[1] - 25), 
                      (intensity_position_right_lane[0] + 460, intensity_position_right_lane[1] + 10), background_color, -1)

        # Add the traffic intensity text on top of the rectangle for the right lane
        cv2.putText(frame, f'Traffic Intensity: {traffic_intensity_left}', intensity_position_right_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Display the processed frame
        # cv2.imshow('Detection Frame', detection_frame)
        cv2.imshow('Filtered Frame', filtered_frame)
        # cv2.imshow('Post Predict', post_predict_frame)
        cv2.imshow('Real-time Traffic Analysis', frame)

        # Press Q on keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and video write objects
cap.release()
out.release()

# Close all the frames
cv2.destroyAllWindows()

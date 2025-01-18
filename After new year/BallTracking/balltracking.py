import cv2
from collections import deque
import imutils
import numpy as np
from numpy.typing import NDArray
from typing import Union
from typing import Tuple, Optional

pts = deque(maxlen=64)

def initialize_video_stream (args:dict) -> Union[cv2.VideoCapture, None]:
    video_path = args.get("video", None)

    if video_path is None:
        print("Error: No video file path provided.")
        return None
    return cv2.VideoCapture(video_path)

def process_frame(frame: NDArray, 
                  green_lower: Tuple[int, int, int] = (30, 80, 8), 
                  green_upper: Tuple[int, int, int] = (62, 255, 255)
                 ) -> Tuple[NDArray, NDArray, Optional[Tuple[int, int]], Optional[float]]:
    
    # Resize and prepare the frame
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create a mask for the green color
    mask = cv2.inRange(hsv, green_lower, green_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None
    radius = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            pts.append(center)
        
        if radius > 10:
            top_left = (int(center[0] - radius), int(center[1] - radius))
            bottom_right = (int(center[0] + radius), int(center[1] + radius))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)

    for i in range(1, len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue

            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness)

    return frame, mask, center, radius

args = {"video": "ball_tracking_example.mp4"}
video_stream = initialize_video_stream(args)

if video_stream:
    while True:
        ret, frame = video_stream.read()
        if not ret:
            print("End of video or error reading the video.")
            break

        processed_frame, mask, center, radius = process_frame(frame)

        cv2.imshow("Mask", mask)
        cv2.imshow("Processed Frame", processed_frame)

        if cv2.waitKey(1) & 0xFF == 27: #esc to close
            break

    video_stream.release()

cv2.destroyAllWindows()
import cv2
import numpy as np
from collections import deque

#Video capture
vs = cv2.VideoCapture("ball_tracking_example.mp4")


# lower and upper color of "green" ball in HSV color
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

pts = deque(maxlen=64)

def process_frame(frame, lower_bound, upper_bound):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (11, 11), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours

def track_ball(frame, contours, min_radius=10):
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        
        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            pts.append(center)

        if radius > min_radius:
            top_left = (int(center[0] - radius), int(center[1] - radius))
            bottom_right = (int(center[0] + radius), int(center[1] + radius))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
            if center:
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)

    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue

        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness)

    return frame

while True:
    ret, frame = vs.read()
    if not ret:
        print("Cannot access or find video file.")
        break

    mask, contours = process_frame(frame, greenLower, greenUpper)
    frame = track_ball(frame, contours)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == 27: #Esc to close
        break

cv2.destroyAllWindows()
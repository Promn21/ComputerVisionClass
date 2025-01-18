import cv2
import numpy as np

image_paths = ['example_01.png', 'example_02.png', 'example_03.png']

def count_objects(image_path):
    # Read the images
    img = cv2.imread(image_path)

    # Process the images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 300 
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Draw contours and count objects
    count = len(filtered_contours)
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Just Text
    cv2.putText(img, f"Objects Detected: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img, edges, count

# Show images
for i, path in enumerate(image_paths):
    result_img, dilated_edges, object_count = count_objects(path)

    cv2.imshow(f"Detected Objects {i+1}", result_img)
    cv2.imshow(f"Edges {i+1}", dilated_edges)

    print(f"Image {i+1}: {object_count} objects detected.")

cv2.waitKey(0)
cv2.destroyAllWindows()
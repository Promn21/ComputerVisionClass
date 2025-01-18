import cv2
import argparse
import numpy as np
import numpy.typing as npt

def order_points(pts: npt.NDArray) -> npt.NDArray:
    """Order points in clockwise order (top-left, top-right, bottom-right, bottom-left)."""
    rect = np.zeros((4,2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left
    return rect

def four_point_transform(image: npt.NDArray, pts: npt.NDArray) -> npt.NDArray:
    """Apply perspective transform to obtain a top-down view Workshop: Building a Document Scanner with Python and OpenCV 5 of an image."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # Compute width and height
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl [1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl [1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br [1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl [1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Construct destination points
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1,maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)

    # Apply perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def main() -> None:
    #img Path
    parser = argparse.ArgumentParser(description= 'image')
    parser.add_argument("-i", '--image', required=True, help="path to image")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
            raise ValueError("The image is missing")
    

    #Resize
    ratio = 1500/img.shape[0] 
    og = img.copy()
    resize_image = cv2.resize(img, (0, 0), fx=ratio, fy=ratio,interpolation=cv2.INTER_CUBIC)

    #Grey Scale and Doc Edge
    gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    #Contours
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #cv2.drawContours(resize_image, contours, -1, (0,255,0), 3)

    #The document contour
    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screen_cnt = approx
            print ("found")
            break
    warp = four_point_transform(resize_image, screen_cnt.reshape(4,2))
    warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    warp_final = cv2.adaptiveThreshold(warp_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)  

    #result image
    cv2.imshow(f"contour og img", resize_image)
    cv2.imshow(f"gray img", gray)
    cv2.imshow(f"blur img", blur)
    cv2.imshow(f"edged img", edged)
    cv2.imshow(f"warp1 img", warp)
    cv2.imshow(f"warp2 img", warp_final)
    #cv2.imshow(f"contours img", contours)
    # cv2.imshow("Original", cv2.resize(og, (0, 0), fx=650/og.shape[0], fy=650/og.shape[0]))
    # cv2.imshow("Scanned", cv2.resize(thresh, (0, 0), fx=650/thresh.shape[0], fy=650/thresh.shape[0]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()











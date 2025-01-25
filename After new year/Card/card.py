import cv2 as cv




face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_img = cv.imread('face.jpg')
bg_id = cv.imread('bg.png')
gray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

if len (faces) > 0:
    #cv.rectangle(face_img, faces[0], (1,1,1), 1)

    #get face coordinate
    x, y, w, h = faces[0]
    
    #crop
    center = (x + w / 2, y + h / 2)
    size = (int(w * 1.8), int(h * 1.8))
    cropped_face = cv.getRectSubPix(face_img, size, center)
    
    # Resize
    face_box_width, face_box_height = 254, 272
    resized_face = cv.resize(cropped_face, (face_box_width, face_box_height))
    
    #try to replace bg
    x_offset, y_offset = 686, 217
    bg_id[y_offset:y_offset+face_box_height, x_offset:x_offset+face_box_width] = resized_face


cv.imshow('face img', face_img)
#cv.imshow('gray img', gray)
cv.imshow('id', bg_id)

cv.waitKey(0)
cv.destroyAllwindows()
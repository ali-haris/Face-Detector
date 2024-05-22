import cv2
import glob 

# images = glob.glob('*jpg')
# print(images)


face_cascade = cv2.CascadeClassifier('face.xml')
counter = 0

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
window_name = 'Live Webcam'

cv2.namedWindow(window_name)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    # cv2.imshow(window_name, frame)
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img,
    scaleFactor=1.2,
    minNeighbors=5)

    # print(faces)

    for x,y,w,h in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow('video',frame)

        # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break
    
# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

import cv2
import glob

# Get a list of all jpg images in the current directory
images = glob.glob('*jpg')
print(images)

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier('face.xml')
counter = 0

# Loop through each image in the list
for image in images:
    # Read the image file
    img = cv2.imread(image)

    # Optionally resize the image to half its original size
    # img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image using the face detection model
    faces = face_cascade.detectMultiScale(gray_img,
                                          scaleFactor=1.2,
                                          minNeighbors=8)

    # Print the coordinates of detected faces
    print(faces)

    # Draw a rectangle around each detected face
    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the image with the detected faces
    cv2.imshow(image, img)
    print('value of image ', image)
    
    # Wait for 5000 milliseconds (5 seconds) before closing the image window
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
    # Increment the counter for processed images
    counter += 1
    print(counter)

import cv2
import glob 

images = glob.glob('*jpg')
print(images)


face_cascade = cv2.CascadeClassifier('face.xml')
counter = 0
for image in images:

    img = cv2.imread(image)

    # img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img,
    scaleFactor=1.2,
    minNeighbors=8)

    print(faces)

    for x,y,w,h in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)



    cv2.imshow(image,img)
    print('value of image ',image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    counter += 1
    print(counter)

# print(gray_img)
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('1.jpg')
img = cv2.resize(img,(480,640))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

eyes = eye_cascade.detectMultiScale(gray,1.3,5)

for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    print("detect eye")

faces = face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    print("detect face")


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyWindow()
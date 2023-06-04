import cv2

image = cv2.imread("smile.jpg")
face_cascade = cv2.CascadeClassifier("frontalface.xml")
smile_cascade = cv2.CascadeClassifier("smile.xml")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for face in faces:
    (x, y, w, h) = face

face_image = image[y:y+h, x:x+w]
gray_face = gray[y:y+h, x:x+w]

smiles = smile_cascade.detectMultiScale(gray_face, 1.2, 5)

for smile in smiles:
    (x, y, w, h) = smile
    cv2.rectangle(face_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

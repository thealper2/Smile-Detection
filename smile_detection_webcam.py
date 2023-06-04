import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("frontalface.xml")
smile_cascade = cv2.CascadeClassifier("smile.xml")

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.8, 3)
    for face in faces:
        (x, y, w, h) = face

    face_image = frame[y:y+h, x:x+w]
    gray_image = gray[y:y+h, x:x+w]

    smiles = smile_cascade.detectMultiScale(gray_image, 1.8, 2)

    for smile in smiles:
        (x, y, w, h) = smile
        cv2.rectangle(face_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

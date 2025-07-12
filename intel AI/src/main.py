import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import preprocess

model = load_model('../saved_model/visual_check.h5')

cap = cv2.VideoCapture(0)  # Use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = preprocess(frame)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]

    label = "GOOD" if pred < 0.5 else "DEFECT"
    color = (0, 255, 0) if label == "GOOD" else (0, 0, 255)

    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.imshow("Visual Quality Check", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

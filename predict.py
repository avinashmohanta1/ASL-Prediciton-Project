import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 64
model = load_model("asl_model.h5")
label_map = np.load("label_map.npy", allow_pickle=True).item()
rev_label_map = {v: k for k, v in label_map.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi / 255.0
    roi = roi.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict(roi)
    predicted_class = np.argmax(prediction)
    letter = rev_label_map[predicted_class]

    cv2.putText(frame, f'Prediction: {letter}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

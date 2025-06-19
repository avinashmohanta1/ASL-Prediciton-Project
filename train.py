import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

DATA_DIR =r"C:\Users\Acer\OneDrive\Desktop\ASL Recog\archive\asl_alphabet_train\asl_alphabet_train"  # change if needed
IMG_SIZE = 64

# Load and preprocess data
def load_data():
    X, y = [], []
    labels = sorted(os.listdir(DATA_DIR))
    label_map = {label: i for i, label in enumerate(labels)}
    for label in labels:
        path = os.path.join(DATA_DIR, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label_map[label])
            except:
                pass
    return np.array(X), np.array(y), label_map

X, y, label_map = load_data()
X = X / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = to_categorical(y, num_classes=len(label_map))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model and label map
model.save("asl_model.h5")
np.save("label_map.npy", label_map)

print("Model trained and saved!")

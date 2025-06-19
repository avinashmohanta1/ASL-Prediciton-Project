# 🤟 ASL Hand Gesture Recognition with Real-Time Sentence Building

This project is a real-time American Sign Language (ASL) alphabet recognition system using a webcam. It detects hand gestures representing ASL letters and dynamically builds words or sentences based on the predictions.

### 🧠 Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- NumPy

---

## 📌 Features

- 🔤 Real-time ASL letter prediction via webcam
- 🧠 Trained deep learning model (CNN) on ASL dataset
- 🧾 Automatic sentence building from recognized letters
- ⌨️ Spacebar adds a space between words
- 🧹 Press `c` to clear the sentence
- 🛑 Press `q` to quit the app

---

## 📂 Project Structure

asl-recognition/
│
├── asl_model.h5 # Trained Keras model for letter recognition
├── label_map.npy # Label-to-letter mapping dictionary
├── predict.py # Main prediction and sentence building script
└── README.md # Project documentation




🖐 How It Works
A rectangular region of interest (ROI) appears on the webcam.

Place your hand gesture inside the box.

The model processes each frame and predicts the ASL letter.

It only adds a letter to the sentence after confirming it over a few frames (to avoid flickering).
Use keyboard keys:

Space: Add space

c: Clear sentence

q: Quit

🙌 Acknowledgments
ASL Alphabet Dataset - Kaggle

OpenCV and TensorFlow community



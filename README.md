# Arabic Character Recognition (Deep CNN)

This project implements a deep convolutional neural network (CNN) in TensorFlow/Keras for recognizing Arabic handwritten characters.  
It includes full preprocessing, model design, training, and evaluation pipeline.

---

## ✨ Project Description
- Developed a CNN-based model for **Arabic handwritten character recognition**.  
- Applied preprocessing steps: image rotation, flipping, normalization.  
- Built a deep CNN with multiple convolutional, pooling, batch normalization, and dropout layers.  
- Trained with **ModelCheckpoint** to save the best-performing weights.  
- Achieved **[96.3% accuracy]** on the test dataset *(placeholder — update once results are finalized)*.  

---

## 📂 Project Structure
arabic-character-recognition/
│
├── arabic_character_recognizer_clean.py # Main training & evaluation script
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ⚡ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Rahaf-jaber/arabic-character-recognition.git
cd arabic-character-recognition

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the script

python arabic_character_recognizer_clean.py

---
## Results

Training Accuracy: [96.8%]

Validation Accuracy: [95.6%]

Test Accuracy: [96.3%]

## Technologies

Python

TensorFlow / Keras

NumPy, Pandas

OpenCV

Matplotlib

Scikit-learn



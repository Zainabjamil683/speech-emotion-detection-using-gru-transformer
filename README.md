# 🎧 Speech Emotion Detection using GRU & Transformer Models

This project focuses on detecting human emotions from speech audio using deep learning models (GRU & Transformer) trained on the RAVDESS dataset. A simple Flask web app lets users upload `.wav` or `.mp3` files and get the predicted emotion along with an emoji!

---

## 💡 Key Features

- Trained and compared 4 models:
  - GRU + Adam
  - GRU + RMSprop
  - Transformer + Adam
  - Transformer + RMSprop
- Selected the best performing model based on accuracy
- Used librosa to extract MFCC, Chroma, Mel, Contrast, and Tonnetz features
- Flask backend for audio upload and emotion prediction
- Frontend styled with dark theme and emoji-based emotion display

---

## 🧠 Dataset

- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song  
- Downloaded using `kagglehub`
- Contains audio files labeled with emotions like: `happy`, `sad`, `angry`, `fearful`, etc.

---

## 🚀 How to Run

```bash
# Clone repo
git clone https://github.com/yourusername/speech-emotion-detection.git
cd speech-emotion-detection

# Install requirements
pip install -r requirements.txt

# Run Flask app
python app.py
````

---

## 📁 Project Structure

```
speech-emotion-detection/
├── app.py                   # Flask backend with prediction logic
├── emotion_detection.ipynb # Notebook to train and compare 4 models
├── best_gru_model.h5       # Saved best-performing model
├── preprocessing_scaler.pkl# Scaler used for feature normalization
├── requirements.txt        # All dependencies
├── templates/
│   └── index.html          # HTML frontend with dark theme
├── README.md
```

---

## 🎯 Result Visualization

* Bar plots for accuracy comparison
* Training/Validation Accuracy & Loss graphs
* Emoji display for emotion prediction on UI

---

## 🧠 Technologies Used

* TensorFlow / Keras
* Flask
* Librosa
* Pandas / NumPy / Sklearn
* HTML / CSS

---

## 🧑‍💻 Author

Zainab Jamil – [LinkedIn](https://www.linkedin.com/in/zainabjamilpythondeveloper)


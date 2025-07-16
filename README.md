# Platform-Identification-System-using-Acoustic-Signature

This project focuses on identifying different underwater platforms (e.g., submarines, dolphins, torpedoes, cargo ships,  etc.) using their unique acoustic signatures. It combines signal processing with deep learning to classify platform types based on audio features extracted from underwater recordings.

## 📂 Dataset & Testing Audios
> ⚠️ **Note**: The dataset and testing audio files were not sourced from Kaggle originally.  
They were curated and compiled manually using multiple raw underwater recordings from public and defense-relevant sources.  
Due to GitHub’s size limitations, they have now been uploaded to **Kaggle** for convenient access only.  
- 🔗 [Access Dataset + Testing Audios on Kaggle](https://www.kaggle.com/datasets/arshiasingh2005/platform-identification-using-acoustic-signature)  

To use this project:
1. Download `dataset.zip` and `testing_audios.zip` from the Kaggle link.  
2. Extract them into your working directory.  


## 🎯 Project Overview

The goal of this system is to classify underwater sounds into categories such as:
- **Submarine**   
- **Torpedo**  
- **Dolphin**  
- **Whale**  
- **Tanker**  
- **Cargo Ship**  
- **Passenger Ship**  
- **Tugboat**  

This is achieved using a hybrid **CNN-LSTM** deep learning model trained on stacked MFCC and spectral features.  


## 💻 Technologies Used

- Python 3.10  
- Librosa  
- TensorFlow / Keras  
- NumPy, Pandas, Seaborn, Matplotlib  
- scikit-learn  
- glob2, soundfile, IPython  


## 📊 Features Extracted

- **MFCCs** (Mel-Frequency Cepstral Coefficients)  
- **RMS Energy**  
- **Spectral Centroid**  
- **Zero Crossing Rate**  
- Mel Spectrograms  
- FFT (Fast Fourier Transform)  


## 🧠 Model Architecture

- **Input Shape**: (40, 862, 1) MFCC + spectral stack  
- **CNN layers**: 3 convolutional layers with dropout and L2 regularization  
- **TimeDistributed Flatten**  
- **LSTM Layer**: Captures temporal dynamics in sound  
- **Dense Output**: Softmax classifier  


## 📈 Model Performance

- Achieved up to **85–87% accuracy** on validation data depending on dataset balance and augmentation  
- Model trained with:  
  - Class weights to handle imbalance  
  - Data augmentation (noise injection, pitch/tempo shift)  


## 📁 Repository Structure
Platform-Identification-System-using-Acoustic-Signature/  
│  
├── final_code.py                  # Complete pipeline: load, extract, train, evaluate  
├── cnn_rnn_acoustic_model.h5     # Saved CNN-LSTM model  
├── best_model.h5                 # Best model with early stopping  
├── README.md                     # You're reading it!  

## 📬 Contact
For queries or collaboration, feel free to connect:  
Arshia Singh  
LinkedIn : https://www.linkedin.com/in/arshia05/  
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob2 import glob
import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

from glob import glob

# Load all .wav files from dolphin, torpedo, and ship folders
audio_files = glob("D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/dataset/*/*.wav")

# Print all audio file paths
print(audio_files)
print(len(audio_files))  # Total number of audio files

# Play one audio file using IPython
ipd.Audio(audio_files[50])

# Load audio file for analysis
y, sr = librosa.load(audio_files[48])  # y is waveform, sr is sampling rate

# Compute FFT (frequency analysis)
fft_vals = np.fft.fft(y)
fft_freq = np.fft.fftfreq(len(fft_vals), 1/sr)
pos_mask = fft_freq >= 0
fft_freq = fft_freq[pos_mask]
fft_power = np.abs(fft_vals[pos_mask])

plt.figure(figsize=(10, 5))
plt.plot(fft_freq, fft_power, color=color_pal[2])
plt.title('FFT of Audio Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim([0, sr/2])
plt.show()

# Generate mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
S_dB = librosa.amplitude_to_db(S, ref=np.max)

fig, ax = plt.subplots(figsize=(10, 5))
plt.title('Mel spectrogram of audio signal')
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
plt.colorbar(img, format="%+2.f dB")
plt.xlabel('Time (s)')
plt.ylabel('Mel frequency (Hz)')
plt.show()

import os
from sklearn.preprocessing import LabelEncoder

# Extract labels from folder names
labels = [file.split("\\")[-2] for file in audio_files]
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
print("Classes:", le.classes_)

# Feature extraction function
def extract_features(file_path, max_pad_len=862):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        print("Error processing", file_path, ":", e)
        return None

# Extract features from directory
def extract_features_from_directory(directory_path):
    features = []
    labels = []
    for label in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                file_path = os.path.join(label_path, file)
                mfcc = extract_features(file_path)
                if mfcc is not None:
                    features.append(mfcc.flatten())
                    labels.append(label)
    return np.array(features), np.array(labels)

x, y = extract_features_from_directory("D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/dataset")
print("Features shape:", x.shape)
print("Labels shape:", y.shape)

# Average mel spectrograms per class
DATASET_PATH = "D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/dataset"
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION = 5
FIXED_LEN = SAMPLE_RATE * DURATION

def add_noise(data, noise_level=0.005):
    noise = np.random.randn(len(data))
    return data + noise_level * noise

for category in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, category)
    if not os.path.isdir(class_path):
        continue
    mel_specs = []
    for file in os.listdir(class_path):
        if file.endswith('.wav'):
            file_path = os.path.join(class_path, file)
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            if len(y) < FIXED_LEN:
                y = np.pad(y, (0, FIXED_LEN - len(y)))
            else:
                y = y[:FIXED_LEN]
            if np.random.rand() < 0.5:
                y = add_noise(y)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=512, n_fft=2048, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_specs.append(mel_spec_db)
    if mel_specs:
        mel_specs = np.stack(mel_specs)
        avg_mel_spec = np.mean(mel_specs, axis=0)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(avg_mel_spec, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Average Mel-Spectrogram - {category}')
        plt.tight_layout()
        plt.show()

# Compute average FFTs
def compute_average_fft_per_label(directory_path):
    label_fft_averages = {}
    sample_rate = None
    for label in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label)
        if not os.path.isdir(label_path):
            continue
        fft_list = []
        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                file_path = os.path.join(label_path, file)
                audio, sr = librosa.load(file_path, sr=22050)
                sample_rate = sr
                fft = np.abs(np.fft.fft(audio))[:len(audio) // 2]
                fft_list.append(fft)
        if fft_list:
            min_len = min(len(f) for f in fft_list)
            trimmed_ffts = [f[:min_len] for f in fft_list]
            average_fft = np.mean(trimmed_ffts, axis=0)
            label_fft_averages[label] = average_fft
    return label_fft_averages, sample_rate

# Plot FFT comparison across labels
averages, sr = compute_average_fft_per_label(DATASET_PATH)
plt.figure(figsize=(14, 6))
for label, avg_fft in averages.items():
    normalized_fft = avg_fft / np.max(avg_fft)
    freqs = np.fft.fftfreq(len(avg_fft) * 2, d=1/sr)[:len(avg_fft)]
    max_hz = 5000
    max_bin = np.argmax(freqs >= max_hz) if np.any(freqs >= max_hz) else len(freqs)
    plt.plot(freqs[:max_bin], normalized_fft[:max_bin], label=label)
plt.title("Average FFT Comparison Across Labels (0–5 kHz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.yscale("log")
plt.ylabel("Log Magnitude")
plt.show()

# Zoomed-in FFT comparison
labels_to_plot = ['Submarine', 'Torpedo', 'Cargo']
plt.figure(figsize=(12, 6))
for label in labels_to_plot:
    fft = averages[label]
    norm_fft = fft / np.max(fft)
    freqs = np.fft.fftfreq(len(fft) * 2, d=1/sr)[:len(fft)]
    max_hz = 3000
    max_bin = np.argmax(freqs >= max_hz)
    plt.plot(freqs[:max_bin], norm_fft[:max_bin], label=label)
plt.title("Zoomed FFT Comparison (0–3 kHz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Deep learning model with CNN + LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, TimeDistributed, BatchNormalization, Activation, Input
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and process audio files
features = []
labels = []
for label in os.listdir(DATASET_PATH):
    class_dir = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_dir):
        continue
    for file in os.listdir(class_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(class_dir, file)
            mfcc = extract_features(file_path)
            features.append(mfcc)
            labels.append(label)

# Prepare model input
x = np.array(features)
x = x[..., np.newaxis]  # Add channel dim
y = LabelEncoder().fit_transform(labels)
y = tf.keras.utils.to_categorical(y)

# Split dataset
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# Build CNN + LSTM model
model = Sequential()
model.add(Input(shape=(40, 862, 1)))
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile and train model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True)

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), 
y=np.argmax(y_train, axis=1))
class_weights_dict = dict(enumerate(class_weights))

history = model.fit(x_train, y_train, validation_split=0.2, epochs=40, batch_size=32, callbacks=[early_stop, checkpoint], 
class_weight=class_weights_dict, verbose=1)

# Plot training accuracy/loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

# Evaluate model
loss, accuracy = model.evaluate(x_val, y_val)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model
model.save("cnn_rnn_acoustic_model.h5")

# Predict new audio

def predict(file_path):
    mfcc = extract_features(file_path)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    prediction = model.predict(mfcc)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

print(predict("D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/testing_audios/wh/testwh_031.wav"))
print(predict("D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/testing_audios/testing_audio001.wav"))
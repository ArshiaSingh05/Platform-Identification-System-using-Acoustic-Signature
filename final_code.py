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

print(audio_files)
# LABEL = SHIP TYPE

print(len(audio_files))

ipd.Audio(audio_files[50])
import librosa
y, sr = librosa.load(audio_files[48])
# y -> raw data of the audio file
# sr -> sampling rate of the audio file
# Compute the Fast Fourier Transform (FFT) of the audio signal
# This will convert the time-domain signal into the frequency domain.
# The FFT is a powerful algorithm to compute the Discrete Fourier Transform (DFT) and its inverse.
fft_vals = np.fft.fft(y)
fft_freq = np.fft.fftfreq(len(fft_vals), 1/sr)

# Only take the positive frequencies
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

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
S_dB = librosa.amplitude_to_db(S, ref=np.max)

fig,ax =  plt.subplots(figsize=(10, 5))
plt.title('Mel spectrogram of audio signal')
# plot the mel spectrogram

img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
plt.colorbar(img, format="%+2.f dB")
plt.xlabel('Time (s)')
plt.ylabel('Mel frequency (Hz)')  
plt.show()

import os
from sklearn.preprocessing import LabelEncoder

# Get all .wav files from all subfolders
audio_files = glob("D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/dataset/*/*.wav")

# Extract labels (folder names like dolphin, torpedo, ship)
labels = [file.split("\\")[-2] for file in audio_files]

# Encode string labels to integers
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

print("Classes:", le.classes_)  # Output: ['dolphin' 'ship' 'torpedo']

def extract_features(file_path, max_pad_len=862):  # 862 = around 5 sec for 22050Hz
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Padding or trimming to fixed length
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        
        return mfcc
    except Exception as e:
        print("Error processing", file_path, ":", e)
        return None
    
    
import os
import numpy as np

def extract_features_from_directory(directory_path):
    features = []
    labels = []

    for label in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label)

        if not os.path.isdir(label_path):
            continue  # Skip non-directory files

        for file in os.listdir(label_path):
            if file.endswith(".wav"):  # or ".mp3", depending on your dataset
                file_path = os.path.join(label_path, file)

                mfcc = extract_features(file_path)

                if mfcc is not None:
                    features.append(mfcc.flatten())  # flatten if needed
                    labels.append(label)

    return np.array(features), np.array(labels)


x, y = extract_features_from_directory("D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/dataset")

print("Features shape:", x.shape)
print("Labels shape:", y.shape)

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 🔧 Parameters
DATASET_PATH = "D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/dataset"  # Root folder containing class folders
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION = 5  # seconds
FIXED_LEN = SAMPLE_RATE * DURATION

# 🎧 Add Noise Function
def add_noise(data, noise_level=0.005):
    noise = np.random.randn(len(data))
    return data + noise_level * noise

# 🔁 Loop through each class/category
for category in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, category)
    if not os.path.isdir(class_path):
        continue

    mel_specs = []

    for file in os.listdir(class_path):
        if file.endswith('.wav'):
            file_path = os.path.join(class_path, file)

            # 🔊 Load audio
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # 🩹 Pad or truncate
            if len(y) < FIXED_LEN:
                y = np.pad(y, (0, FIXED_LEN - len(y)))
            else:
                y = y[:FIXED_LEN]

            if np.random.rand() < 0.5:  # 50% chance
                y = add_noise(y)


            # 📊 Convert to mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=80, hop_length=512, n_fft = 2048, fmax = 8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            mel_specs.append(mel_spec_db)

    if mel_specs:
        # 📚 Stack and average
        mel_specs = np.stack(mel_specs)
        avg_mel_spec = np.mean(mel_specs, axis=0)

        # 🖼️ Plot the average spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(avg_mel_spec, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                 x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Average Mel-Spectrogram - {category}')
        plt.tight_layout()
        plt.show()

import os
import librosa
import numpy as np

def compute_average_fft_per_label(directory_path):
    label_fft_averages = {}
    sample_rate = None  # Optional: capture the sample rate (22050 by default)

    for label in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label)
        if not os.path.isdir(label_path):
            continue

        fft_list = []

        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                file_path = os.path.join(label_path, file)
                
                # Load the audio file
                audio, sr = librosa.load(file_path, sr=22050)
                sample_rate = sr

                # Compute FFT (positive frequencies only)
                fft = np.abs(np.fft.fft(audio))[:len(audio) // 2]
                fft_list.append(fft)

        # Average the FFTs for this label
        if fft_list:
            min_len = min(len(f) for f in fft_list)
            trimmed_ffts = [f[:min_len] for f in fft_list]
            average_fft = np.mean(trimmed_ffts, axis=0)
            label_fft_averages[label] = average_fft

    return label_fft_averages, sample_rate

import matplotlib.pyplot as plt
import numpy as np

def plot_fft_subplots(fft_data, sample_rate, zoom_hz=5000):
    num_labels = len(fft_data)
    fig, axes = plt.subplots(num_labels, 1, figsize=(12, 3 * num_labels), sharex=True)

    if num_labels == 1:
        axes = [axes]

    for ax, (label, fft) in zip(axes, fft_data.items()):
        # Normalize
        norm_fft = fft / np.max(fft)

        # Frequency axis (in Hz)
        freqs = np.fft.fftfreq(len(fft) * 2, d=1/sample_rate)[:len(fft)]

        # Limit to zoom range
        max_bin = np.argmax(freqs >= zoom_hz) if np.any(freqs >= zoom_hz) else len(freqs)

        ax.plot(freqs[:max_bin], norm_fft[:max_bin])
        ax.set_title(f"FFT for {label}")
        ax.set_ylabel("Normalized Magnitude")
        ax.grid(True)

    axes[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

fft_data, sr = compute_average_fft_per_label("D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/dataset")

averages, sr = compute_average_fft_per_label("D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/dataset")

plt.figure(figsize=(14, 6))

for label, avg_fft in averages.items():
    # Normalize for fair comparison
    normalized_fft = avg_fft / np.max(avg_fft)
    
    # Convert bin index to frequency in Hz
    freqs = np.fft.fftfreq(len(avg_fft) * 2, d=1/sr)[:len(avg_fft)]
    
    # Plot the first N Hz
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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, TimeDistributed, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Loads an audio file and extracts MFCC (Mel Frequency Cepstral Coefficients)—a compact representation of audio.

def extract_features(file_path, max_pad_len=862):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

features = []
labels = []

for label in os.listdir("D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/dataset"):
    class_dir = os.path.join("D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/dataset", label)
    if not os.path.isdir(class_dir):
        continue

    for file in os.listdir(class_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(class_dir, file)
            mfcc = extract_features(file_path)
            features.append(mfcc)
            labels.append(label)


# CNN expects 4D input: (samples, height, width, channels).
# We add the last channel dimension (1) to fit this shape.

import numpy as np

x = np.array(features)  # shape = (num_samples, 40, 862)
x = x[..., np.newaxis]  # shape = (num_samples, 40, 862, 1)

# LabelEncoder: Converts string labels to integers
# to_categorical: Converts integer labels to one-hot encoding

le = LabelEncoder()
y = le.fit_transform(labels)  # e.g., ['torpedo', 'ship'] -> [0, 1]
y = tf.keras.utils.to_categorical(y)  # e.g., [0, 1] -> [[1,0], [0,1]]

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)
print("X_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, TimeDistributed
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Input, Activation
from tensorflow.keras import regularizers

model = Sequential()

# 📌 Input Layer
model.add(Input(shape=(40, 862, 1)))

# 🧠 Conv Layer 1 — Keep L2, reduce Dropout to prevent early underfitting
model.add(Conv2D(32, (3, 3), padding='same',
                 kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 🧠 Conv Layer 2 — Moderate Dropout, reduced L2 regularization
model.add(Conv2D(64, (3, 3), padding='same',
                 kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# 🧠 Conv Layer 3 — Deeper dropout kept for strong regularization
model.add(Conv2D(128, (3, 3), padding='same',
                 kernel_regularizer=regularizers.l2(0.0005)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# 🔄 TimeDistributed Flatten
model.add(TimeDistributed(Flatten()))

# 🔁 LSTM — Add recurrent_dropout, moderate dropout
model.add(LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))
model.add(Dropout(0.3))

# 🎯 Output Layer
model.add(Dense(y.shape[1], activation='softmax'))


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

from collections import Counter
print(Counter(y_train.argmax(axis=1)))  # if categorical

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train.argmax(axis=1)),
    y=y_train.argmax(axis=1)
)
class_weights_dict = dict(enumerate(class_weights))

from sklearn.utils import class_weight
import numpy as np

# Compute class weights (only if y_train is one-hot encoded)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights_dict = dict(enumerate(class_weights))

# ✅ Early stopping and checkpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True)

# ✅ Model training with class weights
history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights_dict,  # 👈 this handles class imbalance
    verbose=1
)
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

print(f"Train Accuracy: {train_acc:.2f}")
print(f"Validation Accuracy: {val_acc:.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
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

loss, accuracy = model.evaluate(x_val, y_val)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# ADD NOISE
# Add Noise:-	Simulates real-world microphone noise, audio + 0.005 * np.random.randn(len(audio))
def add_noise(data, noise_level=0.005):
    noise = np.random.randn(len(data))
    return data + noise_level * noise

# TIME SHIFTING
# Time Shifting	:- Shifts the waveform slightly left/right,np.roll(audio, shift_amt)
def shift_audio(audio, shift_max=2):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(audio, shift)

loss, accuracy = model.evaluate(x_val, y_val)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# MODEL SAVING
model.save("cnn_rnn_acoustic_model.h5")

# PREDICT ON NEW AUDIO FILE
def predict(file_path):
    mfcc = extract_features(file_path)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # reshape to (1, 40, 862, 1)
    prediction = model.predict(mfcc)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

print(predict("D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/testing_audios/wh/testwh_031.wav"))

print(predict("D:/DRDO INTERNSHIP/Platform Identification Using Acoustic Signature/testing_audios/testing_audio001.wav"))

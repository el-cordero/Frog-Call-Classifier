import os
import librosa
import numpy as np

os.environ["TF_ENABLE_MLIR"] = "1"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_USE_LEGACY_GPU_KERNELS"] = "1"

import tensorflow as tf
from tensorflow.keras import layers, models

import subprocess
subprocess.run(["system_profiler", "SPDisplaysDataType"])

# Fixed spectrogram shape (H, W)
SPEC_HEIGHT = 128
SPEC_WIDTH = 344  

def preprocess(file_path, label):
    # Convert TensorFlow tensor to a string
    file_path = file_path.numpy().decode('utf-8')

    # Load audio file
    wav, sr = librosa.load(file_path, sr=16000)  # Resample to 16kHz
    wav = librosa.util.fix_length(wav, size=sr * 3)  # Ensure 3 seconds

    # Convert to Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=SPEC_HEIGHT)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibels

    # Ensure a fixed size using resizing
    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)  # Add channel dim
    mel_spec_db = tf.image.resize(mel_spec_db, (SPEC_HEIGHT, SPEC_WIDTH))  # Resize

    # Normalize spectrogram values
    mel_spec_db = (mel_spec_db - tf.reduce_mean(mel_spec_db)) / tf.math.reduce_std(mel_spec_db)

    return mel_spec_db, label

# Apply function within a TensorFlow Dataset
def load_and_preprocess(file_path, label):
    spectrogram, label = tf.py_function(preprocess, [file_path, label], [tf.float32, tf.int32])
    spectrogram.set_shape((SPEC_HEIGHT, SPEC_WIDTH, 1))  # Ensure fixed shape
    return spectrogram, label

wd = os.getcwd().replace("/_scripts","")
POS = os.path.join(wd,'_data','Example', 'Parsed_Capuchinbird_Clips', '*.wav')
NEG = os.path.join(wd,'_data','Example','Parsed_Not_Capuchinbird_Clips', '*.wav')

# Load file paths
pos_files = tf.data.Dataset.list_files(POS)
neg_files = tf.data.Dataset.list_files(NEG)

# Check if files are loaded
print(f"Positive samples: {len(list(pos_files))}")
print(f"Negative samples: {len(list(neg_files))}")

# Apply preprocessing
positives = pos_files.map(lambda x: load_and_preprocess(x, 1))
negatives = neg_files.map(lambda x: load_and_preprocess(x, 0))

# Combine datasets
data = positives.concatenate(negatives)

# Shuffle, batch, and prefetch
data = data.shuffle(1000).repeat().batch(16).prefetch(tf.data.AUTOTUNE)

for sample_wav, sample_label in data.take(1):
    print("Waveform shape:", sample_wav.shape)
    print("Label shape:", sample_label.shape)
    print("Label values:", sample_label.numpy())

# Waveform shape: (16, 128, 344, 1)
# Label shape: (16,)
# Label values: [0 1 0 0 1 1 0 0 0 1 0 1 0 1 0 0]
# 2025-02-18 19:36:31.895039: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
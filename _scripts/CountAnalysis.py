#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Packages
import warnings
import os
import gc
import librosa
import librosa.display
import numpy as np
import pandas as pd
import glob
import csv
from IPython.display import clear_output

# Suppress general warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Ignore DeprecationWarnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore UserWarnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
np.seterr(all="ignore")

os.environ["TF_ENABLE_MLIR"] = "1"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_USE_LEGACY_GPU_KERNELS"] = "1"

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt


# In[2]:

wd = os.path.join(os.getcwd(), "_scripts")
os.chdir(wd)  # Change the working directory

# model = os.getcwd().replace("/_scripts","/_results/Model/my_model_20_col_waa.keras")
model = os.getcwd().replace("/_scripts","/_results/Model/my_model_50_all.keras")
model = load_model(model)


# In[3]:


# Define the audio folder path
AUDIO_FOLDER = os.getcwd().replace("/_scripts", "/_data/Audio/Full/Starlink_Group_7-13")

# Directory to save spectrogram images
RESULTS_DIR = os.getcwd().replace("/_scripts", "/_results/Model/output")
SPECTROGRAM_SAVE_DIR = os.getcwd().replace("/_scripts","/_results/Model/spectrograms")
POS_FOLDER = os.path.join(SPECTROGRAM_SAVE_DIR, "positives")
NEG_FOLDER = os.path.join(SPECTROGRAM_SAVE_DIR, "negatives")

# Ensure output directories exist
os.makedirs(SPECTROGRAM_SAVE_DIR, exist_ok=True)  # Create directory if not exists
os.makedirs(POS_FOLDER, exist_ok=True)
os.makedirs(NEG_FOLDER, exist_ok=True)


# In[4]:


# Define the audio folder path
AUDIO_FOLDER = os.getcwd().replace("/_scripts", "/_data/Audio/Full/Starlink_Group_7-13")
audio_files = glob.glob(os.path.join(AUDIO_FOLDER, "**/*.wav"), recursive=True)
for i in sorted(audio_files[0:5]):
    print(i)


# In[5]:


# Constants
SAMPLE_RATE = 40000
CLIP_LENGTH = 1  # seconds
MAX_SPECTROGRAMS = 1800  # ✅ Hard limit per file
F_MIN, F_MAX = 0, 1500
SPEC_HEIGHT, SPEC_WIDTH = 120, 80  # Height (mel bands) and fixed width for x seconds
WINDOW_SIZE = SAMPLE_RATE * CLIP_LENGTH
# 🔹 Initialize spectrogram counter
spectrogram_counter = 1


# In[8]:


def plot_spectrogram(spectrogram, sr, filename):
    """ Save spectrogram as an image file """
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spectrogram.numpy().squeeze(), sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close("all")

def waveform_to_spectrogram(clip, sr):
    """ Convert waveform to a normalized spectrogram """
    global spectrogram_counter

    # ✅ Limit spectrograms per file
    if spectrogram_counter > MAX_SPECTROGRAMS:
        return None  # Skip extra spectrograms

    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=128, fmin=F_MIN, fmax=F_MAX, hop_length=256, n_fft=4096)

    # Convert power to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=1)

    # Convert to TensorFlow tensor and normalize
    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)  # Add channel dimension
    mel_spec_db = tf.image.resize(mel_spec_db, (SPEC_HEIGHT, SPEC_WIDTH))  # Resize    
    # mel_spec_db.set_shape((SPEC_HEIGHT, SPEC_WIDTH, 1))
    mel_spec_db = (mel_spec_db - tf.reduce_mean(mel_spec_db)) / tf.math.reduce_std(mel_spec_db)

    return mel_spec_db  # Return processed spectrogram

def process_audio_file(audio_path):
    """ Process audio into 1-sec spectrograms, classify, save results """
    global spectrogram_counter
    spectrogram_counter = 1  # ✅ Reset per file

    # Load audio
    wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    total_duration = librosa.get_duration(y=wav, sr=sr)

    # ✅ Check if file has already been processed in either positive or negative folders
    base_name = os.path.basename(audio_path)
    pos_check_path = os.path.join(POS_FOLDER, f"{base_name}_s_1780.csv")
    neg_check_path = os.path.join(NEG_FOLDER, f"{base_name}_s_1780.csv")

    if os.path.exists(pos_check_path) or os.path.exists(neg_check_path):
        print(f"⏩ Skipping {base_name} (Already Processed)")
        return  # Skip processing if found

    print(f"🔹 Processing: {os.path.basename(audio_path)}, Duration: {total_duration:.2f}s, SR: {sr}Hz")

    # Split into 1-second clips
    clips = [wav[i: i + WINDOW_SIZE] for i in range(0, len(wav), WINDOW_SIZE)]

    # # ✅ Remove last clip if it's too short
    # if len(clips[-1]) < WINDOW_SIZE:
    #     clips.pop()

    for i, clip in enumerate(clips):
        print(f"✅ Starting Clip: {i + 1}")
        start_time = i * CLIP_LENGTH
        end_time = min((i + 1) * CLIP_LENGTH, total_duration)

        # ✅ Generate spectrogram
        spectrogram = waveform_to_spectrogram(clip, sr)

        if spectrogram is None:
            break  # Stop processing if max reached

        # ✅ Predict class and confidence
        prediction = model.predict(np.expand_dims(spectrogram, axis=0))
        label = int(prediction > 0.5)
        confidence = round(float(prediction), 4)

        # ✅ Assign correct save folder based on prediction
        save_folder = POS_FOLDER if label == 1 else NEG_FOLDER
        base_name = os.path.basename(audio_path)
        save_path = os.path.join(save_folder, f"{base_name}_s_{spectrogram_counter:04d}.jpeg")

        # ✅ Save Spectrogram
        plot_spectrogram(spectrogram, sr=sr, filename=save_path)

        # ✅ Save results to an individual CSV file per clip
        results_path = os.path.join(save_folder, f"{base_name}_s_{spectrogram_counter:04d}.csv")
        with open(results_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["audiofile", "clip_no", "start_time", "end_time", "prediction", "confidence", "filepath"])
            writer.writerow([base_name, i + 1, start_time, end_time, label, confidence, audio_path])

        print(f"✅ Saved spectrogram: {save_path}, CSV: {results_path}")

        spectrogram_counter += 1  # Increment counter

        # ✅ Free memory after each clip
        del spectrogram, writer
        gc.collect()

    print(f"✅ Finished processing {os.path.basename(audio_path)}")

    # ✅ Free memory after each process
    del wav, clips
    gc.collect()
    
    # ✅ Clear Jupyter Notebook output after processing a file
    clear_output(wait=True)

def process_all_audio_files():
    """ Process all audio files in directory recursively """
    audio_files = glob.glob(os.path.join(AUDIO_FOLDER, "**/*.wav"), recursive=True)
    if not audio_files:
        print("❌ No audio files found!")
        return

    for audio_path in audio_files:
        process_audio_file(audio_path)


# In[ ]:


# # Initialize a counter to assign sequential numbers to spectrograms
# spectrogram_counter = 1

# # Define the specific audio file to test
# TEST_AUDIO_FILE = os.getcwd().replace("/_scripts","/_data/Audio/Full/Detection_Test_Files_20250227/Digital_Globe/Ag Ditch - Y/S4A23845_20240501_120000.wav")  # Change to your test file name

# # Ensure the file exists before processing
# if os.path.exists(audio_path):
#     results = process_audio_file(audio_path)

#     # Convert to DataFrame for analysis
#     results_df = pd.DataFrame(results)

#     # check
#     results_df.head()

#     # Save results to CSV if needed
#     results_df.to_csv(OUTPUT_FILE, index=False)
# else:
#     print(f"File not found: {audio_path}")

#     # Ensure the file exists before processing

# if os.path.exists(TEST_AUDIO_FILE):
#     process_audio_file(TEST_AUDIO_FILE)

# else:
#     print(f"File not found: {TEST_AUDIO_FILE}")


# In[9]:


# Initialize a counter to assign sequential numbers to spectrograms
spectrogram_counter = 1

# Run the processing and display the DataFrame
process_all_audio_files()


# In[ ]:


def merge_all_csvs(input_folder, output_csv):
    """
    Recursively searches for all CSV files in subdirectories of input_folder and merges them into one CSV file.

    Args:
        input_folder (str): Path to the main directory containing subdirectories with CSV files.
        output_csv (str): Path where the merged CSV will be saved.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    
    # 🔹 Search recursively for all CSV files in subdirectories
    csv_files = glob.glob(os.path.join(input_folder, "**/*.csv"), recursive=True)
    
    if not csv_files:
        print("❌ No CSV files found in the directory!")
        return None

    # 🔹 Read and merge all CSV files
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['source_file'] = os.path.basename(file)  # ✅ Add column to track source file
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")

    # 🔹 Concatenate all DataFrames
    merged_df = pd.concat(df_list, ignore_index=True)

    # 🔹 Save merged DataFrame
    merged_df.to_csv(output_csv, index=False)
    print(f"✅ Merged CSV saved at: {output_csv}")

    return merged_df


# In[ ]:

output_csv = os.path.join(RESULTS_DIR,"merged_counts.csv")
merged_df = merge_all_csvs(SPECTROGRAM_SAVE_DIR,output_csv)
# Display first rows
merged_df.head()

# 

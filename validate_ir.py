import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from tools import *

def plot_spectrograms(clean, reverb, synthetic_reverb, dereverb, fs, title, task):
    """Plot spectrograms of clean, reverberant, and dereverberated audio"""
    fig, axs = plt.subplots(4, 1, figsize=(12, 15))
    for i, (audio, label) in enumerate(zip([clean, reverb, synthetic_reverb, dereverb], ['Clean', 'Reverberant', 'Synthetic Reverb', 'Dereverberated'])):
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', sr=fs, ax=axs[i])
        axs[i].set_title(f'{label} Spectrogram')
        fig.colorbar(img, ax=axs[i], format='%+2.0f dB')
    plt.tight_layout()
    # plt.savefig(f'{title}_{task}_spectrograms.png')
    # plt.close()

def validate_ir(clean_audio_file, recorded_audio_file, ir, task, if_plot=True):
    """validate the processed IR"""
    # load audio
    clean_audio, fs = load_audio(clean_audio_file)
    recorded_audio, _ = load_audio(recorded_audio_file)

    min_length = min(len(clean_audio), len(recorded_audio))
    clean_audio = clean_audio[:min_length]
    recorded_audio = recorded_audio[:min_length]
    # 
    synthetic_audio = convolve_with_ir(clean_audio, ir)
    synthetic_audio = synthetic_audio / np.max(synthetic_audio)
    synthetic_audio = synthetic_audio[:len(recorded_audio)]  # make sure the same length
    synthetic_audio = calculate_signal_delay(recorded_audio, synthetic_audio, fs)

    # Calculate the correlation coefficient
    correlation = np.corrcoef(recorded_audio, synthetic_audio)[0, 1]
    # Calculate mean square error
    mse = np.mean((recorded_audio - synthetic_audio)**2)
    
    # Plot a comparison chart
    if if_plot:
        time = np.arange(len(recorded_audio)) / fs
        plt.figure(figsize=(12, 8))
        plt.subplot(4, 1, 1)
        plt.plot(time, clean_audio)
        plt.title("Clean Audio")
        plt.ylabel("Amplitude")

        plt.subplot(4, 1, 2)
        plt.plot(time, recorded_audio)
        plt.title("Recorded Audio")
        plt.ylabel("Amplitude")
        
        plt.subplot(4, 1, 3)
        plt.plot(time, synthetic_audio)
        plt.title("Convolved Clean Audio (with processed IR)")
        plt.ylabel("Amplitude")
        
        plt.subplot(4, 1, 4)
        plt.plot(time, recorded_audio - synthetic_audio)
        plt.title("Difference")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        plt.tight_layout()

        plot_spectrograms(clean_audio, recorded_audio, synthetic_audio, recorded_audio - synthetic_audio, fs, "synthetic signal", task)
        plt.show()
    return correlation, mse

ir_folder_root = 'HelsinkiSpeech2024/Impulse_Responses/IR'
speech_folder_root = 'HelsinkiSpeech2024'
is_task2 = True # dereverbration task

if is_task2:
    for i in range(1, 4):
        task = 'task_2_level_' + str(i)
        task_folder = task.replace('t', 'T', 1).replace('l', 'L', 1)

        clean_aduio_file = f"{speech_folder_root}/{task_folder}/Clean/{task}_clean_003.wav"
        recorded_audio_file = f"{speech_folder_root}/{task_folder}/Recorded/{task}_recorded_003.wav"
        estimated_ir = load_ir(f"{ir_folder_root}/ir_swept_sine_wave_{task}.npy")

        correlation, mse = validate_ir(clean_aduio_file, recorded_audio_file, estimated_ir, task, if_plot=True)
        
        print(f"Correlation between recorded and convolved sweep: {correlation:.4f}")
        print(f"Mean Squared Error: {mse:.8f}")
else:
    for i in range(1, 8):
        task = 'task_1_level_' + str(i)
        task_folder = task.replace('t', 'T', 1).replace('l', 'L', 1)

        clean_audio_file = f"{speech_folder_root}/{task_folder}/Clean/{task}_clean_150.wav"
        recorded_audio_file = f"{speech_folder_root}/{task_folder}/Recorded/{task}_recorded_150.wav"
        estimated_ir = load_ir(f"{ir_folder_root}/ir_white_noise_short_{task}.npy")

        # 验证处理后的IR
        correlation, mse = validate_ir(clean_audio_file, recorded_audio_file, estimated_ir, task, if_plot=False)
        
        print(f"Correlation between recorded and convolved sweep: {correlation:.4f}")
        print(f"Mean Squared Error: {mse:.8f}")

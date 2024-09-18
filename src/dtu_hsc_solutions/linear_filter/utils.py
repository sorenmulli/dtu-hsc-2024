# Based on Yue's tools.py
import numpy as np
import numpy as np
import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import os
from scipy.signal import butter, filtfilt
from numpy.fft import fft, ifft

MODEL_NAME = "linear-filter"

def check_folder(path):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def load_audio(audio_path, normalize=True):
    signal, fs = librosa.load(audio_path, sr=None)
    if normalize:
        signal = signal / np.max(np.abs(signal))
    return signal, fs

def load_ir(ir_path, normalize=True):
    ir = np.load(ir_path)
    if normalize:
        ir = ir / np.max(np.abs(ir))
    return ir

def save_audio(output_path, audio, fs, normalize=True):
    check_folder(output_path)
    if normalize:
        audio = audio / np.max(np.abs(audio))
    sf.write(output_path, audio, fs)

def get_wav_files(path):
    # 获取当前目录下所有文件
    files = os.listdir(path)
    # 筛选出以 .wav 结尾的文件
    wav_files = [f for f in files if f.endswith('.wav')]
    return wav_files

def ensure_same_length(clean_signal, ir_signal):
    ref_length = len(clean_signal)
    if len(ir_signal) > ref_length:
        target_audio_trimmed = ir_signal[:ref_length]
    else:
        target_audio_trimmed = np.pad(ir_signal, (0, ref_length - len(ir_signal)), mode='constant')

    return target_audio_trimmed

def calculate_signal_delay_time(clean_signal, recorded_signal, fs):
    correlation = signal.correlate(recorded_signal, clean_signal, mode='full')

    delay_samples = np.argmax(correlation) - len(clean_signal) + 1
    time_delay = delay_samples / fs
    # print(f"Delay time: {time_delay:.4f} seconds")

    return time_delay

def calculate_signal_delay(clean_signal, recorded_signal, fs):
    correlation = signal.correlate(recorded_signal, clean_signal, mode='full')

    delay_samples = np.argmax(correlation) - len(clean_signal) + 1
    time_delay = delay_samples / fs
    # print(f"Delay time: {time_delay:.4f} seconds")

    if delay_samples > 0:
        processed_speech_aligned = recorded_signal[delay_samples:]
        processed_speech_aligned = np.pad(processed_speech_aligned, (0, delay_samples), 'constant')
    elif delay_samples < 0:
        processed_speech_aligned = np.pad(recorded_signal, (abs(delay_samples), 0), 'constant')
        processed_speech_aligned = processed_speech_aligned[:delay_samples]
    else:
        processed_speech_aligned = recorded_signal
    return processed_speech_aligned

def convolve_with_ir(clean_audio, ir):
    return signal.fftconvolve(clean_audio, ir, mode='full')[:len(clean_audio)]

def pad_ir(ir, target_length):
    """Zero-pad the IR to match the target length."""
    if len(ir) < target_length:
        ir = np.pad(ir, (0, target_length - len(ir)))
    return ir

def plot_etc(etc_db, fs):
    duration = len(etc_db) / fs
    time_axis = np.linspace(0, duration, len(etc_db))

    plt.plot(etc_db)
    plt.title('Energy Time Curve (ETC)')
    plt.xlabel('Samples (s)')
    plt.ylabel('Energy (dB)')
    plt.grid()
    plt.show()

def convolve_OLS(signal, ir, N=None):
    M = len(ir)
    if N is None:
        N = 2 ** int(np.ceil(np.log2(2 * M)))

    if N <= M:
        raise ValueError("FFT block size N must be greater than M。")

    L = N - M + 1

    h_padded = np.pad(ir, (0, N - M))

    epsilon = 1e-10
    H = fft(h_padded)
    H = 1 / (H + epsilon)

    signal_padded = np.pad(signal, (M-1, 0))

    convolved_signal = np.zeros(len(signal) + M - 1)

    n_blocks = int(np.ceil(len(signal))/ L)

    for n in range(n_blocks):
        signal_block = signal_padded[n * L: n * L + N]
        if len(signal_block) < N:
            signal_block = np.pad(signal_block, (0, N - len(signal_block)))

        X = fft(signal_block)
        Y = X * H
        signal_block = ifft(Y)

        convolved_signal[n * L: n * L + L] = np.real(signal_block[M-1:])

    return convolved_signal

def plot_spectrogram(signal, fs, title='Spectrogram', save_path=None):

    D = librosa.stft(signal)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_db, sr=fs, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    # plt.show()
    if save_path:
        check_folder(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_spectrograms(clean, reverb, synthetic_reverb, sr=16000, title=''):
    """Plot spectrograms of clean, reverberant, and dereverberated audio"""
    fig, axs = plt.subplots(4, 1, figsize=(12, 15))
    for i, (audio, label) in enumerate(zip([clean, reverb, synthetic_reverb], ['Clean', 'Reverberant', 'Synthetic Reverb'])):
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', sr=sr, ax=axs[i])
        axs[i].set_title(f'{label} Spectrogram')
        fig.colorbar(img, ax=axs[i], format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def compute_edc(ir, fs):
    energy = np.abs(ir) ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    edc = edc / np.max(edc)
    time = np.linspace(0, len(ir) / fs, len(ir))

    return edc, time

def plot_edc(edc, time, title="Energy Decay Curve (EDC)"):
    plt.figure(figsize=(10, 6))
    plt.plot(time, 10 * np.log10(edc + 1e-10))  # 转换为 dB
    plt.xlabel("Time (seconds)")
    plt.ylabel("Energy (dB)")
    plt.title(title)
    plt.grid()
    plt.show()

def process_long_ir(ir, sample_rate, max_length=2, threshold=-60):
    peak_index = np.argmax(np.abs(ir))
    ir = np.roll(ir, -peak_index)
    ir_db = 20 * np.log10(np.abs(ir) / np.max(np.abs(ir)) + 1e-10)

    threshold_index = np.where(ir_db < threshold)[0]
    if len(threshold_index) > 0:
        cut_index = threshold_index[0]
    else:
        cut_index = len(ir)

    max_samples = int(max_length * sample_rate)
    cut_index = min(cut_index, max_samples)

    ir_truncated = ir[:cut_index]

    fade_length = min(1000, len(ir_truncated) // 10)
    fade_out = np.linspace(1, 0, fade_length)**2
    ir_truncated[-fade_length:] *= fade_out

    return ir_truncated

def plot_time_domain(signal, fs, title="Time Domain Signal"):
    plt.figure(figsize=(10, 6))
    time = np.linspace(0, len(signal) / fs, num=len(signal))
    plt.plot(time, signal, label="Amplitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.show()

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def estimate_noise_from_vad(noisy_signal, fs, vad_threshold=-60, n_fft=1024, hop_length=512):
    energy = librosa.feature.rms(y=noisy_signal, frame_length=n_fft, hop_length=hop_length)
    energy_db = librosa.amplitude_to_db(energy, ref=np.max)

    silent_frames = np.where(energy_db < vad_threshold)[1]

    silent_signal = noisy_signal[1600: 8000]

    noise_stft = librosa.stft(silent_signal, n_fft=n_fft, hop_length=hop_length)
    noise_power = np.mean(np.abs(noise_stft)**2, axis=1)

    return noise_power

def spectral_subtraction_full_band(noisy_signal, fs, n_fft=1024, hop_length=512):
    noisy_stft = librosa.stft(noisy_signal, n_fft=n_fft, hop_length=hop_length)
    noisy_power = np.abs(noisy_stft)**2

    silent_signal = noisy_signal[int(0.05 * fs): int(0.8 * fs)]
    noise_stft = librosa.stft(silent_signal, n_fft=n_fft, hop_length=hop_length)
    noise_power = np.mean(np.abs(noise_stft)**2, axis=1)
    # noise_power = np.load(f'HelsinkiSpeech2024/Impulse_Responses/Noise_Power/noise_power_white_noise_short_high_freq_task_1_level_1.npy')

    enhanced_power = np.maximum(noisy_power - noise_power[:, None], 0)

    enhanced_stft = np.sqrt(enhanced_power) * np.exp(1j * np.angle(noisy_stft))
    enhanced_signal = librosa.istft(enhanced_stft, hop_length=hop_length)

    return enhanced_signal

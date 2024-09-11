import numpy as np
from scipy import signal
from tools import *

def design_inverse_filter(ir, N, regularization=0.01):
    H = np.fft.rfft(ir, n=N)

    # H_inv = np.conj(H) / (np.abs(H)**2 + regularization)
    epsilon = 1e-10
    mask = np.abs(H) < epsilon
    H[mask] = epsilon
    H_inv = 1 / H
    return H_inv

def apply_inverse_filter(audio, H_inv):
    X = np.fft.rfft(audio)

    epsilon = 1e-10
    mask = np.abs(X) < epsilon
    X[mask] = epsilon
    H_inv[mask] = epsilon
    
    Y = X * H_inv
    return np.fft.irfft(Y, n=len(audio))

def high_frequency_recovery(recorded_sweep, ir):
    N = len(recorded_sweep)
    H_inv = design_inverse_filter(ir, N)

    recovered_audio = apply_inverse_filter(recorded_sweep, H_inv)
    recovered_audio = recovered_audio / np.max(np.abs(recovered_audio)) 
    return recovered_audio


for i in range(1, 8):
    task = 'task_1_level_' + str(i)
    task_folder = task.replace('t', 'T', 1).replace('l', 'L', 1)
    clean_signal_file = f"HelsinkiSpeech2024/{task_folder}/Clean/{task}_clean_001.wav"
    recorded_signal_file = f"HelsinkiSpeech2024/{task_folder}/Recorded/{task}_recorded_001.wav"
    clean_signal, fs = load_audio(clean_signal_file)
    recorded_signal, fs = load_audio(recorded_signal_file)

    noise_power = np.load(f'HelsinkiSpeech2024/Impulse_Responses/Noise_Power/noise_power_white_noise_short_high_freq_{task}.npy')
    recorded_signal_sub = spectral_subtraction_full_band(recorded_signal, noise_power, fs)

    ir = np.load(f'HelsinkiSpeech2024/Impulse_Responses/IR/ir_white_noise_short_{task}.npy')

    recovered_signal = high_frequency_recovery(recorded_signal_sub, ir)
    recovered_signal = spectral_subtraction_full_band(recovered_signal, noise_power, fs)

    recovered_signal = recovered_signal / np.max(np.abs(recovered_signal)) / 2
    plot_spectrogram(clean_signal, fs, title='Clean Audio')
    plot_spectrogram(recorded_signal, fs, title='Recorded Audio')
    plot_spectrogram(recorded_signal_sub, fs, title='Recorded Audio-spectral_subtraction')
    plot_spectrogram(recovered_signal, fs, title='Recovered Audio-spectral_subtraction')
    save_audio(f'HelsinkiSpeech2024/{task_folder}/Recovered/{task}_recovered_high_freq_001.wav', recovered_signal, fs)
    plt.show()

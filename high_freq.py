import numpy as np
from scipy import signal
from tools import *

def design_inverse_filter(ir, N, regularization=1e-10):
    H = np.fft.rfft(ir, n=N)
    # H_inv = np.conj(H) / (np.abs(H)**2 + regularization)
    H_inv = 1 / (H + regularization)
    return H_inv

def apply_inverse_filter(audio, H_inv, regularization=1e-10):
    X = np.fft.rfft(audio)
    Y = X * H_inv
    return np.fft.irfft(Y, n=len(audio))

def high_frequency_recovery(recorded_audio, ir):
    N = len(recorded_audio)
    H_inv = design_inverse_filter(ir, N)

    recovered_audio = apply_inverse_filter(recorded_audio, H_inv)
    return recovered_audio

# Usage
speech_folder_root = 'HelsinkiSpeech2024'
ir_folder_root = f'{speech_folder_root}/Impulse_Responses/IR'

for i in range(1, 8):
    task = 'task_1_level_' + str(i)
    task_folder = task.replace('t', 'T', 1).replace('l', 'L', 1)
    ir_path = f"{ir_folder_root}/ir_white_noise_short_{task}.npy"
    ir = load_ir(ir_path)
    clean_path = f"{speech_folder_root}/{task_folder}/Clean/"
    recorded_path = f"{speech_folder_root}/{task_folder}/Recorded/"
    output_path = f"{speech_folder_root}/{task_folder}/HighFreqRecovered/"

    recorded_files = get_wav_files(recorded_path)
    for files in recorded_files:
        recorded_signal, fs = load_audio(recorded_path + files)
        try:
            # recorded_signal_sub = spectral_subtraction_full_band(recorded_signal, fs)
            recovered_audio = high_frequency_recovery(recorded_signal, ir)
            recovered_audio_sub = spectral_subtraction_full_band(recovered_audio, fs)
        except Exception as e:
            print(f"Error in {files} processing pipeline: {str(e)}")
            # return recorded_signal  # 如果处理失败，返回原始录音信号
            continue
        # plot_spectrograms(load_audio(clean_path+files.replace('recorded', 'clean'))[0], 
        #                   recorded_signal, recovered_audio)
        # plt.show()
        save_audio(output_path+files.replace('recorded', 'recovered'), recovered_audio_sub, fs)
        print(files.replace('recorded', 'recovered') + ' has been saved in ' + output_path)

# for i in range(1, 8):
#     task = 'task_1_level_' + str(i)
#     task_folder = task.replace('t', 'T', 1).replace('l', 'L', 1)
#     clean_signal_file = f"HelsinkiSpeech2024/{task_folder}/Clean/{task}_clean_001.wav"
#     recorded_signal_file = f"HelsinkiSpeech2024/{task_folder}/Recorded/{task}_recorded_001.wav"
#     clean_signal, fs = load_audio(clean_signal_file)
#     recorded_signal, fs = load_audio(recorded_signal_file)

#     noise_power = np.load(f'HelsinkiSpeech2024/Impulse_Responses/Noise_Power/noise_power_white_noise_short_high_freq_{task}.npy')
#     recorded_signal_sub = spectral_subtraction_full_band(recorded_signal, noise_power, fs)

#     ir = np.load(f'HelsinkiSpeech2024/Impulse_Responses/IR/ir_white_noise_short_{task}.npy')

#     recovered_signal = high_frequency_recovery(recorded_signal_sub, ir)
#     recovered_signal = spectral_subtraction_full_band(recovered_signal, noise_power, fs)

#     recovered_signal = recovered_signal / np.max(np.abs(recovered_signal)) / 2
#     save_audio(f'HelsinkiSpeech2024/{task_folder}/Recovered/{task}_recovered_high_freq_001.wav', recovered_signal, fs)
#     plt.show()


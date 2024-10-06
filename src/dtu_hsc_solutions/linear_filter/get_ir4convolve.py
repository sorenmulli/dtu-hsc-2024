from tools import *
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

def estimate_ir(clean_audio, recorded_audio, fs, normalization=1e-10):
    X = fft(clean_audio)
    Y = fft(recorded_audio)
    H = Y / (X + normalization)

    return np.real(ifft(H))

def calculate_etc(ir, fs):
    # plot_spectrogram(ir, fs, title='before truncate')
    etc = ir**2
    # plot_fft(ir, fs, title='Frequency response before truncate')
    # plt.show()
    time = np.arange(len(ir)) / fs
    
    return time, etc

def truncate_ir_using_etc(ir, cut_point, fs=16000, etc_threshold=-30):
    _, etc = calculate_etc(ir, fs)
    etc_db = 10 * np.log10(etc / np.max(etc) + 1e-10)
    etc_max_point = np.argmax(etc_db[:fs*2])
    # etc_cut_point = np.where(etc_db[etc_max_point:etc_max_point+fs*1] > etc_threshold)[0][-1]
    # plot_etc(etc_db, fs)
    # plt.show()

    ir_truncated = ir[etc_max_point-200 :etc_max_point+cut_point]
    # plot_spectrogram(ir_truncated, fs, title='after truncate')
    # plot_fft(ir_truncated, fs, title='Frequency response after truncate')
    # plt.show()
    return ir_truncated

def ir_processing(clean_sweep, recorded_sweep, fs, cut_point):
    min_length = min(len(clean_sweep), len(recorded_sweep))
    clean_sweep = clean_sweep[:min_length]
    recorded_sweep = recorded_sweep[:min_length]
    ir = estimate_ir(clean_sweep, recorded_sweep, fs)
    ir_truncated = truncate_ir_using_etc(ir, cut_point, fs)

    return ir_truncated

ir_folder_root = 'HelsinkiSpeech2024/Impulse_Responses'
is_task2 = True # dereverbration task

if is_task2:
    for i in range(1, 4):
        cut_point = [52404, 58388, 54672]
        task = 'task_2_level_' + str(i)
        clean_signal_path = f'{ir_folder_root}/Clean/swept_sine_wave.wav'
        recorded_signal_path = f'{ir_folder_root}/Recorded/swept_sine_wave_{task}.wav'
        clean_audio, fs = load_audio(clean_signal_path)
        recorded_audio, fs = load_audio(recorded_signal_path)
        ir = ir_processing(clean_audio, recorded_audio, fs, cut_point[i-1])
        # swept wave for task 2
        ir_save_path = f'{ir_folder_root}/IR/Convolution/ir_swept_sine_wave_{task}.npy'

        check_folder(ir_save_path)
        np.save(ir_save_path, ir)
        print(f'Impluse Response (IR) for {task} has been saved in {ir_save_path}.')
else:
    for i in range(1,8):
        cut_point = [x + 1000 for x in [640, 636, 725, 533, 430, 493, 562]]
        task = 'task_1_level_' + str(i)
        clean_signal_path = f'{ir_folder_root}/Clean/white_noise_short.wav'
        recorded_signal_path = f'{ir_folder_root}/Recorded/white_noise_short_{task}.wav'
        clean_audio, fs = load_audio(clean_signal_path)
        recorded_audio, fs = load_audio(recorded_signal_path)
        ir = ir_processing(clean_audio, recorded_audio, fs, cut_point[i-1])
        # white noise short for task 1
        ir_save_path = f'{ir_folder_root}/IR/Convolution/ir_white_noise_short_{task}.npy'
        
        check_folder(ir_save_path)
        np.save(ir_save_path, ir)
        print(f'Impluse Response (IR) for {task} has been saved in {ir_save_path}.')
        

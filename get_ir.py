from tools import *
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

def estimate_ir(clean_sweep, recorded_sweep, fs):
    N = len(clean_sweep)
    X = fft(clean_sweep)
    Y = fft(recorded_sweep)
    H = Y / (X + 1e-10)  # 添加小量以避免除零
    return np.real(ifft(H))

def analyze_frequency_response(ir, fs):
    f, H = signal.freqz(ir, worN=8192, fs=fs)
    plt.figure(figsize=(10, 6))
    plt.semilogx(f, 20 * np.log10(np.abs(H)))
    plt.title('Frequency Response of Estimated IR')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid(True)
    # plt.show()

def calculate_etc(ir, fs):
    etc = ir**2
    time = np.arange(len(ir)) / fs
    
    return time, etc

def truncate_ir_using_etc(ir, fs, etc_threshold=-60, is_task2=False):
    _, etc = calculate_etc(ir, fs)
    
    # 基于ETC阈值找到截断点
    etc_db = 10 * np.log10(etc / np.max(etc) + 1e-10)
    
    etc_max_point = np.argmax(etc_db[:fs*2])
    etc_cut_point = np.where(etc_db[etc_max_point:etc_max_point+fs*3] < etc_threshold)[0][-1]
    print(etc_max_point, etc_cut_point, etc_cut_point+etc_max_point)
    # plot_etc(etc_db, fs)
    # plt.show()
    if is_task2:
        etc_cut_point = fs*3
    else:
        etc_cut_point = etc_max_point + etc_cut_point

    ir_truncated = ir [etc_max_point: etc_max_point+300]
    
    return ir_truncated

# def apply_smooth_window(ir, window_type='hann'):
#     window = signal.get_window(window_type, len(ir))
#     return ir * window

def ir_processing(clean_sweep, recorded_sweep, fs, is_task2=False, plot=True):
    min_length = min(len(clean_sweep), len(recorded_sweep))
    clean_sweep = clean_sweep[:min_length]
    recorded_sweep = recorded_sweep[:min_length]
    ir = estimate_ir(clean_sweep, recorded_sweep, fs)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(ir)
        plt.title('Estimated IR')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
    
    if plot:
        analyze_frequency_response(ir, fs)
    
    if is_task2:
        # ir_truncated = truncate_ir_using_etc(ir, fs, is_task2=True)
        ir_truncated = ir
    else:
        ir_truncated = truncate_ir_using_etc(ir, fs, is_task2=False)

    # ir_final = apply_smooth_window(ir_truncated)
    
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(ir_truncated)
        plt.title('Processed IR')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')

        analyze_frequency_response(ir_truncated, fs)
        plt.show()
    return ir_truncated


is_task2 = False
if is_task2:
    for i in range(1, 4):
        task = 'task_2_level_' + str(i)
        clean_signal_path = f'HelsinkiSpeech2024/Impulse_Responses/Clean/swept_sine_wave.wav'
        recorded_signal_path = f'HelsinkiSpeech2024/Impulse_Responses/Recorded/swept_sine_wave_{task}.wav'
        clean_sweep, fs = load_audio(clean_signal_path)
        recorded_sweep, fs = load_audio(recorded_signal_path)
        ir_final = ir_processing(clean_sweep, recorded_sweep, fs, is_task2, plot=False)
        
        print(f"Final IR length: {len(ir_final)} samples ({len(ir_final)/fs:.3f} seconds)")
        np.save(f'HelsinkiSpeech2024/Impulse_Responses/IR/ir_swept_sine_wave_{task}.npy', ir_final)
else:
    for i in range(1,8):
        task = 'task_1_level_' + str(i)
        clean_signal_path = f'HelsinkiSpeech2024/Impulse_Responses/Clean/white_noise_short.wav'
        recorded_signal_path = f'HelsinkiSpeech2024/Impulse_Responses/Recorded/white_noise_short_{task}.wav'
        clean_sweep, fs = load_audio(clean_signal_path)
        recorded_sweep, fs = load_audio(recorded_signal_path)
        # recorded_sweep = spectral_subtraction_full_band(recorded_sweep, fs)
        ir_final = ir_processing(clean_sweep, recorded_sweep, fs, is_task2, plot=False)
        
        print(f"Final IR length: {len(ir_final)} samples ({len(ir_final)/fs:.3f} seconds)")
        np.save(f'HelsinkiSpeech2024/Impulse_Responses/IR/ir_white_noise_short_{task}.npy', ir_final)
        

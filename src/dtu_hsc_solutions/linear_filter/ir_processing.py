import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

from dtu_hsc_data import SAMPLE_RATE

def estimate_ir(clean_audio, recorded_audio, fs, normalization=1e-10):
    X = fft(clean_audio)
    Y = fft(recorded_audio)
    H = Y / (X + normalization)
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


def truncate_ir_using_etc(ir, cut_point,  etc_threshold=-30):
    _, etc = calculate_etc(ir, SAMPLE_RATE)
    etc_db = 10 * np.log10(etc / np.max(etc) + 1e-10)
    etc_max_point = np.argmax(etc_db[:SAMPLE_RATE*2])
    # etc_cut_point = np.where(etc_db[etc_max_point:etc_max_point+fs*1] > etc_threshold)[0][-1]
    # plot_etc(etc_db, fs)
    # plt.show()

    ir_truncated = ir[etc_max_point+1 :etc_max_point+cut_point]
    return ir_truncated


def process_ir(clean_sweep, recorded_sweep, fs, cut_point: int, plot=False) :
    min_length = min(len(clean_sweep), len(recorded_sweep))
    clean_sweep = clean_sweep[:min_length]
    recorded_sweep = recorded_sweep[:min_length]
    ir = estimate_ir(clean_sweep, recorded_sweep, fs)
    ir_truncated = truncate_ir_using_etc(ir, cut_point, fs)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(ir)
        plt.title('Estimated IR')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')

    if plot:
        analyze_frequency_response(ir, fs)

    ir_truncated = truncate_ir_using_etc(ir, fs)

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

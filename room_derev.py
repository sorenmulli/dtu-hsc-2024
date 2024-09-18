import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tools import *

def fft_dereverberation(reverb_audio, ir, regularization=1e-10):
    """
    Perform dereverberation using FFT-based inverse filtering.
    
    Parameters:
    reverb_audio: The reverberated audio signal (recorded speech).
    ir: The impulse response of the room (IR).
    regularization: Small value to avoid division by zero in frequency domain.
    
    Returns:
    dereverb_audio: The dereverberated audio signal.
    """
    reverb_fft = np.fft.fft(reverb_audio, n=len(reverb_audio))
    ir_fft = np.fft.fft(ir, n=len(reverb_audio))
    # Apply inverse filtering in the frequency domain
    dereverb_fft = reverb_fft / (ir_fft + regularization)
    # Perform IFFT to return to time domain
    dereverb_audio = np.fft.ifft(dereverb_fft).real
    # dereverb_audio = apply_time_window(dereverb_audio)
    return dereverb_audio

def validate_ir_and_dereverberate_stft_segments(clean_path, recorded_path, ir):
    """Validate IR and perform dereverberation using STFT with segment processing"""
    # Load audio files
    clean_audio, fs = load_audio(clean_path)
    recorded_audio, fs = load_audio(recorded_path)
    # Ensure synthetic reverb and recorded_audio have the same length
    recorded_audio = recorded_audio[:len(recorded_audio)]
    # recorded_audio = spectral_subtraction_full_band(recorded_audio, fs)


    # Perform dereverberation using STFT with segment processing
    dereverb_audio = fft_dereverberation(recorded_audio, ir)
    # Save the dereverberated audio
    # sf.write(f"{output_path}_dereverberated.wav", dereverb_audio, fs)
    return dereverb_audio, fs

# Usage
speech_folder_root = 'HelsinkiSpeech2024'
ir_folder_root = f'{speech_folder_root}/Impulse_Responses/IR'

for i in range(1, 4):
    task = 'task_2_level_' + str(i)
    task_folder = task.replace('t', 'T', 1).replace('l', 'L', 1)
    ir_path = f"{ir_folder_root}/ir_swept_sine_wave_{task}.npy"
    ir = load_ir(ir_path)
    clean_path = f"{speech_folder_root}/{task_folder}/Clean/"
    recorded_path = f"{speech_folder_root}/{task_folder}/Recorded/"
    output_path = f"{speech_folder_root}/{task_folder}/Dereverberated/"

    clean_files = get_wav_files(clean_path)
    for files in clean_files:
        dereverb_audio, fs = validate_ir_and_dereverberate_stft_segments(clean_path+files, 
                                                                         recorded_path+files.replace('clean', 'recorded'), ir)
        dereverb_audio = spectral_subtraction_full_band(dereverb_audio, fs)
        save_audio(output_path+files.replace('clean', 'dereverberated'), dereverb_audio, fs)
        plot_spectrograms(load_audio(clean_path+files)[0], 
                          load_audio(recorded_path+files.replace('clean', 'recorded'))[0], dereverb_audio)
        plt.show()
        print(files.replace('clean', 'dereverberated') + ' has been saved in ' + output_path)


    
    


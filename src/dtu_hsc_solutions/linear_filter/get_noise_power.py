from tools import *
for i in range(1, 8):
    task = 'task_1_level_' + str(i)
    signal, fs = load_audio('HelsinkiSpeech2024/Impulse_Responses/Recorded/white_noise_short_' + task + '.wav')
    noise_power = estimate_noise_from_vad(signal, fs)
    save_path = 'HelsinkiSpeech2024/Impulse_Responses/Noise_Power/noise_power_white_noise_short_high_freq_' + task + '.npy'
    check_folder(save_path)
    np.save(save_path, noise_power)
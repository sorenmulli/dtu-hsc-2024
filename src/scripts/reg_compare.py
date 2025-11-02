import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import cvxpy as cp
import librosa
from scipy import signal
import argparse
import warnings

SAMPLE_RATE = 16_000  # sample rates for audiofiles


def power_spectrum(data, Fs=SAMPLE_RATE):
    ps = np.abs(np.fft.rfft(data)) ** 2
    time_step = 1 / Fs
    freqs = np.fft.rfftfreq(data.size, time_step)
    idx = np.argsort(freqs)
    return freqs[idx], ps[idx]


def load_audio(audio_path, normalize=True) -> np.ndarray:
    audio, sr = sf.read(audio_path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected sample rate of {SAMPLE_RATE}, but got {sr}")
    if normalize:
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
    return audio


def save_audio(audio_path, audio: np.ndarray):
    sf.write(audio_path, audio, SAMPLE_RATE)


def spectral_subtraction_full_band(noisy_signal, fs, n_fft=1024, hop_length=512):
    noisy_stft = librosa.stft(noisy_signal, n_fft=n_fft, hop_length=hop_length)
    noisy_power = np.abs(noisy_stft) ** 2

    silent_signal = noisy_signal[int(0.05 * fs) : int(0.8 * fs)]
    noise_stft = librosa.stft(silent_signal, n_fft=n_fft, hop_length=hop_length)
    noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1)

    enhanced_power = np.maximum(noisy_power - noise_power[:, None], 0)

    enhanced_stft = np.sqrt(enhanced_power) * np.exp(1j * np.angle(noisy_stft))
    enhanced_signal = librosa.istft(enhanced_stft, hop_length=hop_length)

    return enhanced_signal


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


def reg_fft_dereverberation(reverb_audio, ir, reg_mode="max", lam=1e-10):
    """
    Perform dereverberation using FFT-based inverse filtering.

    Parameters:
    reverb_audio: The reverberated audio signal (recorded speech).
    ir: The impulse response of the room (IR).
    reg_mode: Regularization type, either sup, l1 or l2
    lam: regularization parameter

    Returns:
    dereverb_audio: The dereverberated audio signal.
    """
    warnings.filterwarnings("ignore")  # sometimes cvxpy throws a warning
    reverb_fft = np.fft.rfft(reverb_audio, n=len(reverb_audio))
    ir_fft = np.fft.rfft(ir, n=len(reverb_audio))
    Y = cp.Variable(len(reverb_fft), complex=True)
    if reg_mode == "max":
        obj = cp.Minimize(cp.norm(cp.multiply(ir_fft, Y) - reverb_fft, 2) ** 2)
        constraint = [cp.max(cp.abs(Y)) <= np.max(abs(reverb_fft))]
        prob = cp.Problem(obj, constraint)
    elif reg_mode == "l2":
        obj = cp.Minimize(
            cp.norm(cp.multiply(ir_fft, Y) - reverb_fft, 2) ** 2
            + lam * cp.norm(Y, 2) ** 2
        )
        prob = cp.Problem(obj)
    elif reg_mode == "l1":
        obj = cp.Minimize(
            cp.norm(cp.multiply(ir_fft, Y) - reverb_fft, 2) ** 2 + lam * cp.norm(Y, 1)
        )
        prob = cp.Problem(obj)

    prob.solve(solver=cp.CLARABEL, ignore_dpp=True, max_iter=10)
    if (prob.status == "optimal") or (prob.status == "user_limit"):
        print("Successful regularzation")
        audio = np.fft.irfft(Y.value).real
        audio = spectral_subtraction_full_band(audio, SAMPLE_RATE)
        return audio / np.max(np.abs(audio)) / 2
    else:
        print("Non Successful regularzation")
        return np.fft.irfft(reverb_fft / (ir_fft)).real


def save_spectrogram(input, fs, save_path="spectrogram.png"):
    f, t, Sxx = signal.spectrogram(input, fs=fs)

    plt.figure()
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Power/Frequency (dB/Hz)")

    # Save as an image
    plt.savefig(save_path)
    plt.close()


def plot_spectrogram(signal, fs, title="Spectrogram", save_path=None, x_label=True):

    D = librosa.stft(signal)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    # plt.figure(figsize=(10, 6))
    if x_label:
        librosa.display.specshow(S_db, sr=fs, x_axis="time", y_axis="log")
    else:
        librosa.display.specshow(S_db, sr=fs, y_axis="log")
    plt.colorbar(format="%+2.0f dB")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    # plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


# Load audio files for this example
def main(path_record, path_clean, path_ir_recon, path_ir, lam=1e-2):
    record = load_audio(path_record)
    recon = load_audio(path_ir_recon)
    clean = load_audio(path_clean)
    ir = np.load(path_ir)
    audio = spectral_subtraction_full_band(record, SAMPLE_RATE)
    recon_reg_max = reg_fft_dereverberation(audio, ir, reg_mode="max")
    recon_reg_l2 = reg_fft_dereverberation(audio, ir, reg_mode="l2", lam=lam)
    recon_reg_l1 = reg_fft_dereverberation(audio, ir, reg_mode="l1", lam=lam)
    save_audio("max_rec.wav", recon_reg_max)
    save_audio("tik_rec.wav", recon_reg_l2)
    plt.subplot(2, 3, 1)
    plot_spectrogram(
        recon_reg_max,
        fs=SAMPLE_RATE,
        title="Regularized IR (max-norm)",
        save_path=None,
        x_label=False,
    )
    plt.subplot(2, 3, 2)
    plot_spectrogram(
        recon_reg_l2,
        fs=SAMPLE_RATE,
        title=f"Regularized IR (Tikhonov, alpha = {lam})",
        save_path=None,
        x_label=False,
    )
    plt.subplot(2, 3, 3)
    plot_spectrogram(
        recon_reg_l1,
        fs=SAMPLE_RATE,
        title=f"Regularized IR (LASSO, alpha = {lam})",
        save_path=None,
        x_label=False,
    )
    plt.subplot(2, 3, 4)
    plot_spectrogram(clean, fs=SAMPLE_RATE, title="Clean", save_path=None)
    plt.subplot(2, 3, 5)
    plot_spectrogram(record, fs=SAMPLE_RATE, title="Recorded", save_path=None)
    plt.subplot(2, 3, 6)
    plot_spectrogram(recon, fs=SAMPLE_RATE, title="IR", save_path=None)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare clean, recorded, ir and regularized ir methods"
    )
    parser.add_argument(
        "--path_record", type=str, required=True, help="Path to recorded audio file"
    )
    parser.add_argument("--path_clean", type=str, help="Path to clean audio file")
    parser.add_argument("--path_ir_recon", type=str, help="Path to ir recon file")
    parser.add_argument(
        "--path_ir", type=str, help="Path to ir for this task and level"
    )
    parser.add_argument(
        "--lam", type=float, help="Regularization param. for l2 and l1 reg"
    )
    args = parser.parse_args()
    main(args.path_record, args.path_clean, args.path_ir_recon, args.path_ir, args.lam)

from pathlib import Path
from typing import Literal

from scipy.signal import ShortTimeFFT, convolve

import numpy as np

from dtu_hsc_data.audio import SAMPLE_RATE, load_audio


def compute_filter(
    data_path: Path,
    level: str,
    chosen_response: Literal[
        "swept_sine_wave", "white_noise_long", "white_noise_short"
    ] = "swept_sine_wave",
):
    # TODO: Filter is very long when we compute it on entire impulse response
    # - we should reduce it
    ir_clean = load_audio(data_path / "Impulse_Responses" / "Clean" / f"{chosen_response}.wav")
    ir_rec = load_audio(
        data_path / "Impulse_Responses" / "Recorded" / f"{chosen_response}_{level.lower()}.wav"
    )

    stft = ShortTimeFFT(win=np.hanning(512), hop=256, fs=SAMPLE_RATE)
    H_clean_f = stft.stft(ir_clean)
    H_rec_f = stft.stft(ir_rec)

    # Ensure both STFTs have the same shape
    min_frames = min(H_clean_f.shape[1], H_rec_f.shape[1])
    H_clean_f = H_clean_f[:, :min_frames]
    H_rec_f = H_rec_f[:, :min_frames]

    H_att_f = H_clean_f / (H_rec_f + 1e-8)

    # Convert back to time domain
    return stft.istft(H_att_f)


def run_attenuation_filter(
    audio: np.ndarray,
    data_path: Path,
    level: str,
) -> np.ndarray:
    # TODO: Do not recompute it for each example, these run functions should be classes instead
    # that can set things up in the beginning
    attenuation_filter = compute_filter(data_path, level)
    filtered_audio = convolve(audio, attenuation_filter, mode="same")
    # TODO: Other normalization techniques might be better
    return (filtered_audio - np.mean(filtered_audio)) / np.std(filtered_audio)

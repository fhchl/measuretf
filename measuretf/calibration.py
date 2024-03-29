"""Calibration tools."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from measuretf.fft import amplitude_spectrum, frequency_vector, time_vector
from measuretf.io import load_recording
from scipy.signal import flattop


def record_calibration(
    name, mic_channel, T_rec=5, fs_soundcard=48000, device=None, plot=False, description=""
):
    # record the calibrated pressure
    rec = sd.rec(
        T_rec * fs_soundcard,
        samplerate=fs_soundcard,
        mapping=mic_channel,
        blocking=True,
        device=device,
    )

    np.savez(
        name + " - mic_channel {}".format(mic_channel),
        recs=rec,
        fs=fs_soundcard,
        datetime=datetime.now(),
        description=description,
    )

    if plot:
        plt.figure()
        t = time_vector(T_rec * fs_soundcard, fs_soundcard)
        plt.plot(t, rec)

    return rec


def calibration_gain_from_recording(fname, mic_channel=0, L_calib=94, plot=False):
    with np.load(fname, allow_pickle=True) as data:
        fs = data["fs"]
        rec = data["recs"][:, 0]  # has shape n_in x n_tap

    # flattop window the recording
    window = flattop(len(rec))
    gain_window = window.mean()
    rec = rec * window / gain_window

    p_calib = 10 ** (L_calib / 20) * 20e-6 * np.sqrt(2)
    A = amplitude_spectrum(rec, axis=0)
    p_meas = np.abs(A).max()

    calibration_gain = p_calib / p_meas

    if plot:
        freqs = frequency_vector(rec.shape[0], fs)
        plt.figure()
        plt.plot(
            freqs, 20 * np.log10(np.abs(A) / np.sqrt(2) / 20e-6), label="Uncalibrated"
        )
        plt.plot(
            freqs,
            20 * np.log10(np.abs(A * calibration_gain) / np.sqrt(2) / 20e-6),
            label="Calibrated",
        )
        plt.hlines(L_calib, 0, fs / 2, label="94dB")
        plt.legend()
        plt.xlim(995, 1005)
        plt.ylim(90, 96)
        plt.grid(True)

        plt.figure()
        t = time_vector(len(rec), fs)
        plt.plot(t, rec)

    return calibration_gain

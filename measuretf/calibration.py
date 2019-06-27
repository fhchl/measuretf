"""Calibration tools."""

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from datetime import datetime
from response import Response
from scipy.signal import flattop
from measuretf.fft import amplitude_spectrum, time_vector, frequency_vector


def lanxi_reference_calibration_gain(
    danterec, lanxirec, ref_ch_dante=None, ref_ch_lanxi=None, ls_ch=0, plot=False
):
    """Compute calibration gain for lanxi reference channel."""
    with np.load(danterec) as data:
        fs = int(data["fs"])
        ref_ch_dante = data["ref_ch"] if ref_ch_dante is None else ref_ch_dante
        recs_dante = data["recs"][ls_ch, ref_ch_dante, :]

    with np.load(lanxirec) as data:
        fs_lanxi = int(data["fs"])
        ref_ch_lanxi = data["ref_ch"] if ref_ch_lanxi is None else ref_ch_lanxi
        recs_lanxi = data["recs"][ls_ch, ref_ch_lanxi, :]

    if fs_lanxi != fs:
        recs_lanxi = (
            Response.from_time(fs_lanxi, recs_lanxi)
            .resample_poly(fs, normalize="same_amplitude")
            .in_time
        )
        fs_lanxi = fs

    # flattop window the recording
    window = flattop(len(recs_lanxi))
    gain_window = window.mean()
    rec_lanxi_win = recs_lanxi * window / gain_window

    window = flattop(len(recs_dante))
    gain_window = window.mean()
    rec_dante_win = recs_dante * window / gain_window

    A_dante = amplitude_spectrum(rec_dante_win)
    A_lanxi = amplitude_spectrum(rec_lanxi_win)

    max_dante = np.abs(A_dante).max()
    max_lanxi = np.abs(A_lanxi).max()

    ref_cali_gain = max_dante / max_lanxi

    if plot:
        t_dante = time_vector(len(recs_dante), fs)
        t_lanxi = time_vector(len(recs_lanxi), fs_lanxi)

        plt.figure()
        plt.plot(t_dante, recs_dante)
        plt.plot(t_lanxi, recs_lanxi * ref_cali_gain)

        f_dante = frequency_vector(len(recs_dante), fs)
        f_lanxi = frequency_vector(len(recs_lanxi), fs_lanxi)

        plt.figure()
        plt.plot(f_dante, np.abs(A_dante))
        plt.plot(f_lanxi, np.abs(A_lanxi * ref_cali_gain))
        plt.xlim(995, 1005)

    return ref_cali_gain


def record_calibration(
    name, mic_channel, T_rec=5, fs_soundcard=48000, device=None, plot=False, notes=""
):
    # record the calibrated pressure
    rec = sd.rec(
        T_rec * fs_soundcard,
        samplerate=fs_soundcard,
        mapping=mic_channel,
        blocking=True,
        device=device,
    )

    rec = np.atleast_3d(rec)
    rec = np.moveaxis(rec, 0, -1)

    np.savez(
        name + " - mic_channel {}".format(mic_channel),
        recs=rec,
        mic_channel=mic_channel,
        fs=fs_soundcard,
        datetime=datetime.now(),
        notes=notes,
    )

    if plot:
        plt.figure()
        t = time_vector(T_rec * fs_soundcard, fs_soundcard)
        plt.plot(t, rec[0, 0, :])

    return rec


def calibration_gain_from_recording(fname_rec, mic_channel, L_calib=94, plot=False):
    with np.load(fname_rec) as data:
        rec_all = data["recs"]
        rec_mic = rec_all[0, mic_channel]
        fs = data["fs"]

    # flattop window the recording
    window = flattop(len(rec_mic))
    gain_window = window.mean()
    rec = rec_mic * window / gain_window

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
        t = time_vector(len(rec_mic), fs)
        plt.plot(t, rec_mic)

    return calibration_gain

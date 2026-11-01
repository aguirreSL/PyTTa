# -*- coding: utf-8 -*-
"""
Set of useful functions of general purpouses when using PyTTa.

Includes reading and writing wave files, seeing the audio IO
devices available and few signal processing tools.

Available functions:

    >>> pytta.list_devices()
    >>> pytta.read_wav(fileName)
    >>> pytta.write_wav(fileName, signalObject)
    >>> pytta.merge(signalObj1, signalObj2, ..., signalObjN)
    >>> pytta.slipt(signalObj)
    >>> pytta.split(signalObj)
    >>> pytta.fft_convolve(signalObj1, signalObj2)
    >>> pytta.find_delay(signalObj1, signalObj2)
    >>> pytta.corr_coef(signalObj1, signalObj2)
    >>> pytta.resample(signalObj, newSamplingRate)
    >>> pytta.peak_time(signalObj1, signalObj2, ..., signalObjN)
    >>> pytta.group_delay(signalObj)
    >>> pytta.coherence(signalObj1, signalObj2)
    >>> pytta.cepstrum(signalObj)
    >>> pytta.icepstrum(cepstrumObj, delay)
    >>> pytta.mls_deconvolution(signalObj, order)
    >>> pytta.sine_distortion(signalObj, freq)
    >>> pytta.snr(signalObj, noiseObj)
    >>> pytta.harmonic_distortion(ir, sweep)
    >>> pytta.minimum_phase(signalObj)
    >>> pytta.enhanced_spectrogram(signalObj)
    >>> pytta.smooth_frequency(signalObj, nthOct)
    >>> pytta.time_window(signalObj, interval)
    >>> pytta.plot_time(signalObj1, signalObj2, ..., signalObjN)
    >>> pytta.plot_time_dB(signalObj1, signalObj2, ..., signalObjN)
    >>> pytta.plot_freq(signalObj1, signalObj2, ..., signalObjN)
    >>> pytta.plot_phase(signalObj1, signalObj2, ..., signalObjN)
    >>> pytta.plot_bars(signalObj1, signalObj2, ..., signalObjN)
    >>> pytta.save(fileName, obj1, ..., objN)
    >>> pytta.load(fileName)
    >>> pytta.read_audio(fileName)
    >>> pytta.write_audio(fileName, signalObj)
    >>> pytta.read_blockwise(fileName, blockSize)
    >>> pytta.write_txt(fileName, signalObj)
    >>> pytta.read_txt(fileName)
    >>> pytta.psd(signalObj)
    >>> pytta.interpolate_spectrum(signalObj, nSamples)
    >>> pytta.extrapolate_low(signalObj, freqRange)
    >>> pytta.butterworth(signalObj, cutoff)
    >>> pytta.notch(signalObj, fc)
    >>> pytta.peak_eq(signalObj, fc, gain)
    >>> pytta.envelope(signalObj)
    >>> pytta.start_ir(signalObj)
    >>> pytta.time_shift(signalObj, shift)
    >>> pytta.append(signalObj1, signalObj2)
    >>> pytta.normalize(signalObj)
    >>> pytta.amplify_to(signalObj, level)
    >>> pytta.zero_phase(signalObj)
    >>> pytta.randomize_phase(signalObj)
    >>> pytta.quantize(signalObj, bits)

For further information, check the function specific documentation.

"""

import os
import time
import json
from scipy.io import wavfile as wf
import scipy.io as sio
import numpy as np
import sounddevice as sd
import scipy.signal as ss
import scipy.fftpack as sfft
from scipy.interpolate import CubicSpline
import zipfile as zf
import h5py
from typing import Union, List
from pytta.classes import SignalObj, ImpulsiveResponse, \
                    RecMeasure, PlayRecMeasure, FRFMeasure, \
                    Analysis, OctFilter
from pytta.classes._base import ChannelsList, ChannelObj
from pytta.generate import measurement, mls  # TODO: Change to class instantiation.
from pytta.classes._base import ChannelsList, ChannelObj, CoordinateObj
from pytta.generate import measurement  # TODO: Change to class instantiation.
from pytta import _h5utils as _h5
import copy as cp
from warnings import warn
from pytta import _plot as plot
from pytta.utils.spectrogram import enhanced_map
# For backwards compatibility purposes. Planned to get out of here
from pytta.utils.maths import fft_degree as new_fft_degree


def list_devices():
    """
    Shortcut to sounddevice.query_devices().

    Made to exclude the need of importing Sounddevice directly
    just to find out which audio devices can be used.

        >>> pytta.list_devices()

    Returns
    -------
        A tuple containing all available audio devices.

    """
    return sd.query_devices()


def print_devices():
    """
    Print the devices list to stdout.

    Returns
    -------
    None.

    """
    return print(list_devices())


def get_device_from_user() -> Union[List[int], int]:
    """
    Print the device list and query for a number input of the device, or devices.

    Returns
    -------
    Union[List[int], int]
        Practical interface for querying devices to be used within scripts.

    """
    print_devices()
    device = [int(dev.strip()) for dev in input("Input the device number: ").split(',')]
    if len(device) == 1:
        device = device[0]
        text = "Device is:"
    else:
        text = "Devices are:"
    print(text, device)
    return device


def read_wav(fileName):
    """Read a wave file into a SignalObj."""
    samplingRate, data = wf.read(fileName)
    if data.dtype == 'int16':
        data = data/(2**15)
    if data.dtype == 'int32':
        data = data/(2**31)
    signal = SignalObj(data, 'time', samplingRate=samplingRate)
    return signal


def write_wav(fileName, signalIn):
    """Write a SignalObj into a single wave file."""
    samplingRate = signalIn.samplingRate
    data = signalIn.timeSignal
    return wf.write(fileName if '.wav' in fileName else fileName+'.wav', samplingRate, data)


def SPL(signal, nthOct=3, minFreq=100, maxFreq=4000):
    """
    Calculate the `signal`'s Sound Pressure Level

    The calculations are made by frequency bands and ranges from `minFreq` to
    `maxFreq` with `nthOct` bands per octave.

    Returns
    -------
        Analysis: The sound pressure level packed into an Analysis object.

    """
    with OctFilter(order=4, nthOct=nthOct, minFreq=minFreq, maxFreq=maxFreq,
                   base=10, refFreq=1000, samplingRate=signal.samplingRate) as ofb:
        fsignal = ofb.filter(signal)
    out = []
    for filtsignal in fsignal:
        out.append(Analysis('L', nthOct, minFreq, maxFreq, filtsignal.spl()))
    return out if len(out) > 1 else out[0]


def merge(signal1, *signalObjects):
    """Gather all channels of the signalObjs given as input arguments into a single SignalObj."""
    j = 1
    freqMin = cp.deepcopy(signal1.freqMin)
    freqMax = cp.deepcopy(signal1.freqMax)
    comment = cp.deepcopy(signal1.comment)
    channels = cp.deepcopy(signal1.channels)
    timeSignal = cp.deepcopy(signal1.timeSignal)
    for inObj in signalObjects:
        if signal1.samplingRate != inObj.samplingRate:
            message = '\
            \n To merge signals they must have the same sampling rate!\
            \n SignalObj 1 and '+str(j+1)+' have different sampling rates.'
            raise AttributeError(message)
        if signal1.numSamples != inObj.numSamples:
            message = '\
            \n To merge signals they must have the same length!\
            \n SignalObj 1 and '+str(j+1)+' have different lengths.'
            raise AttributeError(message)
        comment = comment + ' / ' + inObj.comment
        for ch in inObj.channels._channels:
            channels.append(ch)
        timeSignal = np.hstack((timeSignal, inObj.timeSignal))
        j += 1
    newSignal = SignalObj(timeSignal, domain='time',
                          samplingRate=signal1.samplingRate,
                          freqMin=freqMin, freqMax=freqMax, comment=comment)
    channels.conform_to()
    newSignal.channels = channels
    return newSignal


def smooth_notches(signalObj, nthOct=3, threshold=3., squeeze=0.3):
    """
    Spectrum with its notches made shallower, for loudspeaker equalization,
    by the "compare and squeeze" of ita_smooth_notches (after S. Mueller):
    the magnitude is compared with its smoothing over 1 / nthOct octave
    (smooth_frequency), every notch deeper than threshold dB below it is
    taken from where the magnitude last crossed the smoothed one to where
    it crosses it again, and there the ratio r of the two is raised to
    squeeze, so that the notch keeps squeeze times its depth in dB. With
    threshold 0 every bin below the smoothed curve is squeezed. The phase
    is kept. Returns a SignalObj.
    """
    spectrum = np.array(signalObj.freqSignal, dtype=complex, ndmin=2)
    smooth = np.array(smooth_frequency(signalObj, nthOct).freqSignal, dtype=complex, ndmin=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.abs(spectrum) / np.abs(smooth)
    window = np.ones_like(ratio)
    for ch in range(ratio.shape[1]):
        r = ratio[:, ch]
        if not threshold:
            window[:, ch] = np.minimum(r, 1)
            continue
        notch = r < 10 ** (-abs(threshold) / 20)
        notch[0] = notch[1]
        edges = np.diff(notch.astype(int))
        for s, e in zip(np.flatnonzero(edges == 1) + 1, np.flatnonzero(edges == -1)):
            above = np.flatnonzero(r[:s] > 1)
            start = above[-1] + 1 if above.size else 0
            above = np.flatnonzero(r[e:] > 1)
            stop = e + above[0] if above.size else r.size
            window[start:stop, ch] = r[start:stop]
    with np.errstate(divide='ignore', invalid='ignore'):
        window = window ** -(1 - squeeze)
    window[~np.isfinite(window)] = 1
    newSignal = SignalObj(window * spectrum, 'freq',
                          samplingRate=signalObj.samplingRate,
                          numSamples=signalObj.numSamples,
                          signalType=signalObj.signalType,
                          freqMin=signalObj.freqMin,
                          freqMax=signalObj.freqMax,
                          comment=signalObj.comment)
    newSignal.channels = cp.deepcopy(signalObj.channels)
    return newSignal


def lifter(signal, n=None, alpha=0.):
    """
    Low-pass lifter, as ita_lifter: the complex cepstrum of each channel
    (cepstrum, which removes the integer delay) is multiplied by a Tukey
    window of 2 n + 1 quefrencies centred on zero (alpha 0 rectangular,
    1 Hann) and transformed back with the delay (icepstrum), which smooths
    the log magnitude and the phase of the spectrum together. n defaults
    to a quarter of the length; from half the length on nothing is done.

    >>> smooth = pytta.lifter(ir, 64, 0.5)
    """
    N = signal.numSamples
    n = int(np.ceil(N / 4)) if n is None else int(n)
    if n >= np.ceil(N / 2):
        warn('lifter: the window is too long, nothing is done.')
        return _cepstrum_signal(signal, np.asarray(signal.timeSignal, dtype=float))
    win = np.zeros(N)
    win[:2 * n + 1] = ss.windows.tukey(2 * n + 1, alpha)
    win = np.roll(win, -n)
    c, delay = cepstrum(signal)
    lifted = _cepstrum_signal(c, np.asarray(c.timeSignal, dtype=float) * win[:, None])
    return icepstrum(lifted, delay)


def plot_group_delay(*sigObjs, xLim=None, yLim=None, title=None):
    """
    Group delay of every channel of the SignalObjs, in ms, on a logarithmic
    frequency axis, as ita_plot_groupdelay. Returns the Figure.

    >>> pytta.plot_group_delay(ir1, ir2, xLim=[20, 20000])
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for sig in sigObjs:
        tau = group_delay(sig)
        for c in range(tau.shape[1]):
            ax.semilogx(sig.freqVector[1:], 1e3 * tau[1:, c])
    ax.set(xlabel='Frequency [Hz]', ylabel='Group delay [ms]', xlim=xLim, ylim=yLim, title=title)
    ax.grid(True, which='both')
    return fig


def plot_nyquist(*sigObjs, title=None):
    """
    Imaginary against real part of the spectrum of every channel of the
    SignalObjs, as ita_plot_nyquist. Returns the Figure.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for sig in sigObjs:
        X = np.asarray(sig.freqSignal)
        for c in range(X.shape[1]):
            ax.plot(X[:, c].real, X[:, c].imag)
    ax.set(xlabel='Real part', ylabel='Imaginary part', title=title)
    ax.set_aspect('equal', 'datalim')
    ax.grid(True)
    return fig


def plot_zplane(zeros, poles, title=None):
    """
    Zeros (o) and poles (x) of a discrete-time system on the z plane, with the
    unit circle, as ita_plot_zplane; scipy.signal.tf2zpk(b, a) gives them
    from the coefficients. Returns the Figure.

    >>> pytta.plot_zplane(*scipy.signal.tf2zpk(b, a)[:2])
    """
    import matplotlib.pyplot as plt
    zeros, poles = np.atleast_1d(zeros), np.atleast_1d(poles)
    fig, ax = plt.subplots()
    w = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(w), np.sin(w), 'k:', lw=0.8)
    ax.plot(zeros.real, zeros.imag, 'o', mfc='none', label='zeros')
    ax.plot(poles.real, poles.imag, 'x', label='poles')
    ax.set(xlabel='Real part', ylabel='Imaginary part', title=title)
    ax.set_aspect('equal', 'datalim')
    ax.grid(True)
    ax.legend()
    return fig


def plot_polar(angles, levels, labels=None, rLim=None, title=None):
    """
    Directivity in a polar plot, as ita_plot_polar: levels (dB) at the
    angles (degrees, 0 to the front, counterclockwise), one column per curve,
    closed at the first angle. Returns the Figure.

    >>> pytta.plot_polar(np.arange(0, 360, 15), L, labels=['1 kHz', '4 kHz'])
    """
    import matplotlib.pyplot as plt
    theta = np.deg2rad(np.r_[angles, angles[0]])
    levels = np.asarray(levels, dtype=float).reshape(len(angles), -1)
    levels = np.vstack([levels, levels[:1]])
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    for c in range(levels.shape[1]):
        ax.plot(theta, levels[:, c], label=None if labels is None else labels[c])
    if rLim is not None:
        ax.set_rlim(rLim)
    if labels is not None:
        ax.legend()
    ax.set_title(title)
    return fig


def split(*signalObjects,
          channels: list = None) -> list:
    """
    Split the provided SignalObjs' channels into several SignalObjs.

    If the 'channels' input argument is given, split the specified channel numbers of
    each SignalObj, otherwise split all channels.

    Arguments (default), (type):
    -----------------------------

        * non-keyword arguments (), (SignalObj)

        * channels (None), (list):
            specified channels to split from the provided SignalObjs;

    Return (type):
    --------------

        * spltdChs (list):
            a list containing SignalObjs for each split channel;

    """
    spltdChs = []

    for sigObj in signalObjects:
        moreSpltdChs = sigObj.split(channels=channels)
        spltdChs.extend(moreSpltdChs)

    return spltdChs


def fft_convolve(signal1, signal2):
    """
    Use scipy.signal.fftconvolve() to convolve two time domain signals.

        >>> convolution = pytta.fft_convolve(signal1,signal2)

    """
#    Fs = signal1.Fs
    conv = ss.fftconvolve(signal1.timeSignal, signal2.timeSignal)
    signal = SignalObj(conv, 'time', signal1.samplingRate)
    return signal


def find_delay(signal1, signal2):
    """
    Delay of signal2 relative to signal1, in samples, from the peak of the
    cross correlation of their first channels. Positive when signal2 comes
    later; the signals may have different lengths.

        >>> shift = pytta.find_delay(signal1, signal2)

    """
    x, y = signal1.timeSignal[:, 0], signal2.timeSignal[:, 0]
    lags = ss.correlation_lags(y.size, x.size)
    return int(lags[np.argmax(ss.correlate(y, x, method='fft'))])


def corr_coef(signal1, signal2):
    """Correlation coefficient between the first channels of two SignalObjs
    of the same length, by numpy.corrcoef."""
    return np.corrcoef(signal1.timeSignal[:, 0], signal2.timeSignal[:, 0])[0, 1]


def resample(signal, newSamplingRate):
    """
    Resample the timeSignal of the input SignalObj to the
    given sample rate using the scipy.signal.resample() function
    """
    newSignalSize = int(signal.timeLength*newSamplingRate)
    resampled = ss.resample(signal.timeSignal[:], newSignalSize)
    newSignal = SignalObj(resampled, "time", newSamplingRate)
    return newSignal


def peak_time(signal):
    """
    Return the time at signal's amplitude peak.
    """
    if not isinstance(signal, SignalObj):
        raise TypeError('Signal must be an SignalObj.')
    peaks_time = []
    for chindex in range(signal.numChannels):
        # the largest magnitude, positive or negative
        maxindex = np.argmax(np.abs(signal.timeSignal[:, chindex]))
        maxtime = signal.timeVector[maxindex]
        peaks_time.append(maxtime)
    if signal.numChannels > 1:
        return peaks_time
    else:
        return peaks_time[0]

def pitch_detection(signal, fmin=10., fmax=None, blockSize=None, overlap=0.,
                    overtones=5, maxPitches=1, minDistance=10.,
                    minConfidence=0.1, fftDegree=16):
    """
    Pitch of every block of the first channel by the cepstrum biased
    harmonic product spectrum of ita_pitch_detection: the magnitude
    spectrum of the Hann windowed block (zero padded to 2**fftDegree) is
    multiplied by itself decimated by 2 to overtones + 1 (the harmonic
    product spectrum, Noll 1969) and by the magnitude of the real cepstrum
    at the quefrency of each bin. The highest peak between fmin and fmax is
    the pitch; with maxPitches > 1 the band of minDistance Hz around each
    peak is cleared and the next is taken. The confidence of a pitch is the
    share of the energy of the harmonic product spectrum within that band,
    times the number of pitches found (at most 1); pitches below
    minConfidence are nan.

    Returns (pitch, confidence), arrays of shape (maxPitches, blocks), and
    the time (s) of the start of each block.
    """
    fs = signal.samplingRate
    x = np.asarray(signal.timeSignal[:, 0], dtype=float)
    size = blockSize or x.size
    hop = max(1, int(round(size * (1 - overlap))))
    n = max(2 ** fftDegree, size)
    f = np.fft.rfftfreq(n, 1 / fs)
    band = np.flatnonzero((f > fmin) & (f < (fmax or fs / 2)))
    half = max(1, int(np.ceil(minDistance / f[1] / 2)))
    starts = np.arange(0, x.size, hop)
    pitch = np.full((maxPitches, starts.size), np.nan)
    confidence = np.zeros((maxPitches, starts.size))
    for b, s in enumerate(starts):
        block = x[s:s + size]
        block = np.pad(block * np.hanning(block.size), (0, n - block.size))
        X = np.abs(np.fft.rfft(block))
        hps = X.copy()
        for a in range(2, overtones + 2):
            down = X[::a]
            hps[:down.size] *= down
            hps[down.size:] = 0
        with np.errstate(divide='ignore'):
            ceps = np.abs(np.fft.irfft(np.log(np.maximum(X, 1e-300)), n))
        bias = np.zeros_like(hps)
        bias[band] = ceps[np.minimum(n // band, n - 1)]
        score, energy = hps * bias, hps.copy()
        total = np.sum(energy[band] ** 2)
        found = 0
        for p in range(maxPitches):
            k = band[np.argmax(score[band])]
            if score[k] <= 0 or total == 0:
                break
            near = slice(max(band[0], k - half), min(band[-1], k + half) + 1)
            pitch[p, b] = f[k]
            confidence[p, b] = np.sum(energy[near] ** 2) / total
            score[near] = 0
            energy[near] = 0
            found += 1
        confidence[:, b] = np.minimum(confidence[:, b] * max(found, 1), 1)
    pitch[confidence < minConfidence] = np.nan
    confidence[confidence < minConfidence] = 0
    return pitch, confidence, starts / fs


def wiener_hopf(signal, invert=False):
    """
    Wiener-Hopf factorization of every channel into a causal and an
    anticausal part whose spectra multiply to the spectrum of signal, by
    splitting the complex cepstrum (log magnitude and unwrapped phase) at
    quefrency zero, half of the zero and middle quefrencies to each, as
    ita_wiener_hopf_factorization. With invert, the factors of the
    inverse spectrum are returned. Returns (plus, minus).
    """
    x = np.asarray(signal.timeSignal, dtype=float)
    n = x.shape[0]
    m = n // 2
    X = np.fft.fft(x, axis=0)
    phase = np.unwrap(np.angle(X[:m + 1]), axis=0)
    phase = np.vstack([phase, -phase[1:(n + 1) // 2][::-1]])
    c = np.fft.ifft((-1 if invert else 1) * (np.log(np.abs(X)) + 1j * phase), axis=0)
    plus, minus = np.zeros_like(c), np.zeros_like(c)
    plus[0] = minus[0] = c[0] / 2
    plus[1:m] = c[1:m]
    minus[n - m + 1:] = c[n - m + 1:]
    if n % 2 == 0:
        plus[m] = minus[m] = c[m] / 2
    out = [np.fft.ifft(np.exp(np.fft.fft(q, axis=0)), axis=0).real for q in (plus, minus)]
    return tuple(SignalObj(o, 'time', signal.samplingRate) for o in out)


def normxcorr2(template, image):
    """
    Normalized two-dimensional cross-correlation of template with image,
    as MATLAB's normxcorr2 that ita_normxcorr2 applies to array data: at
    every lag of the full correlation (shape image + template - 1), the
    correlation of the zero-mean template with the part of image under it,
    divided by both standard deviations (Lewis, Fast normalized
    cross-correlation, Vision Interface 1995). Values lie in [-1, 1]; lags
    where the image is constant give 0.
    """
    t = np.asarray(template, dtype=float)
    a = np.asarray(image, dtype=float)
    t = t - t.mean()
    ones = np.ones_like(t)
    num = ss.fftconvolve(a, t[::-1, ::-1])
    s1 = ss.fftconvolve(a, ones)
    s2 = ss.fftconvolve(a ** 2, ones)
    var_a = np.maximum(s2 - s1 ** 2 / t.size, 0)
    den = np.sqrt(var_a * np.sum(t ** 2))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(den > 1e-12 * den.max(), num / den, 0)
    return np.clip(r, -1, 1)


def multiple_sweep_excitation(sweep, numSources, delay, tail=0.):
    """
    Excitation of the multiple exponential sweep method (Majdak, Balazs and
    Laback, Multiple exponential sweep method for fast measurement of
    head-related transfer functions, J. Audio Eng. Soc. 55(7/8), 2007), the
    interleaved and MIMO measurements of the ITA-Toolbox: channel k plays
    sweep delayed by k delay seconds, with tail seconds of silence at the
    end for the last responses. delay must exceed the length of the
    impulse responses plus the advance of the harmonics that matter.
    """
    fs = sweep.samplingRate
    s = np.asarray(sweep.timeSignal[:, 0], dtype=float)
    d = int(round(delay * fs))
    x = np.zeros((s.size + (numSources - 1) * d + int(round(tail * fs)), numSources))
    for k in range(numSources):
        x[k * d:k * d + s.size, k] = s
    return SignalObj(x, 'time', fs, freqMin=sweep.freqMin, freqMax=sweep.freqMax)


def multiple_sweep_irs(recording, sweep, numSources, delay, irLength,
                       pretime=0.005, regularization=True):
    """
    Impulse responses of each source to every input channel from a
    recording of multiple_sweep_excitation: the recording is deconvolved
    with the single sweep by ImpulsiveResponse, regularized between the
    freqMin and freqMax of sweep unless regularization is False, and the
    response of source k is cut from pretime seconds before k delay,
    irLength seconds long, so that the onset of the band-limited impulse is
    kept. The regularization limits the band and so spreads each impulse
    in time, and part of it falls in the windows of the other sources
    (about 1 % with the default band); without it the division is exact
    but raises the noise outside the band of the sweep. Returns one
    SignalObj per source, with one channel per input channel.
    """
    fs = recording.samplingRate
    n = recording.timeSignal.shape[0]
    s = np.zeros(n)
    s[:sweep.timeSignal.shape[0]] = sweep.timeSignal[:, 0]
    excitation = SignalObj(s, 'time', fs, freqMin=sweep.freqMin, freqMax=sweep.freqMax)
    ir = ImpulsiveResponse(excitation, recording, 'linear', regularization=regularization)
    h = np.asarray(ir.IR.timeSignal, dtype=float)
    d, m = int(round(delay * fs)), int(round(irLength * fs))
    h = np.roll(h, int(round(pretime * fs)), axis=0)
    return [SignalObj(h[k * d:k * d + m], 'time', fs) for k in range(numSources)]


def triggered_average(recording, threshold, duration, pretime=0., channel=0):
    """
    Average of the events of a triggered measurement (e.g. impact hammer),
    as ita_measurement_trigger: an event starts where the trigger channel
    first exceeds threshold dB re full scale after the previous event, and
    is cut from pretime seconds before that sample, duration seconds long,
    in every channel. Returns (mean SignalObj, list of the events).
    """
    fs = recording.samplingRate
    x = np.asarray(recording.timeSignal, dtype=float)
    pre, m = int(round(pretime * fs)), int(round(duration * fs))
    above = np.flatnonzero(np.abs(x[:, channel]) > 10 ** (threshold / 20))
    runs, end = [], -1
    for i in above:
        if i >= end and i - pre >= 0 and i - pre + m <= x.shape[0]:
            runs.append(SignalObj(x[i - pre:i - pre + m], 'time', fs))
            end = i - pre + m
    if not runs:
        raise ValueError('No event above the threshold.')
    mean = np.mean([r.timeSignal for r in runs], axis=0)
    return SignalObj(mean, 'time', fs), runs


def dither(signal, bits=24, kind='tri', quiet=False, seed=None):
    """
    signal (full scale 1) with dither noise added before it is quantized to
    bits bits, so that the quantization error becomes noise independent of
    the signal, as ita_dither (Lipshitz, Wannamaker and Vanderkooy,
    Quantization and dither: a theoretical survey, J. Audio Eng. Soc.
    40(5), 1992). kind 'rect' is uniform over one step, 'tri' (triangular,
    two steps wide) makes the mean and the power of the error independent
    of the signal, and 'gauss' has a standard deviation of half a step. With
    quiet, samples equal to zero get no dither.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(signal.timeSignal, dtype=float)
    step = 2.0 ** (1 - bits)
    if kind == 'rect':
        d = rng.uniform(-step / 2, step / 2, x.shape)
    elif kind == 'tri':
        d = rng.uniform(-step / 2, step / 2, x.shape) + rng.uniform(-step / 2, step / 2, x.shape)
    elif kind == 'gauss':
        d = rng.normal(0, step / 2, x.shape)
    else:
        raise ValueError("kind must be 'rect', 'tri' or 'gauss'.")
    if quiet:
        d[x == 0] = 0
    return SignalObj(x + d, 'time', signal.samplingRate)


def impulsive_noise(ir, excitation, noise_start, threshold=20):
    """
    Detection of an impulsive event during a sweep measurement, as
    ita_impulsiveNoiseDetection: the part of the impulse response ir after
    noise_start seconds, where it holds only noise, is convolved again with
    the excitation (circularly, at the length of ir) to rebuild the
    background noise of the recording, and L_max_rms = 20 lg(max |n| /
    rms(n)) of that noise is compared with threshold (dB). Stationary noise
    gives 12 to 14 dB.

    Returns (detected, L_max_rms, background noise SignalObj).
    """
    h = np.asarray(ir.timeSignal[:, 0], dtype=float).copy()
    h[:int(round(noise_start * ir.samplingRate))] = 0
    x = np.asarray(excitation.timeSignal[:, 0], dtype=float)
    n = np.fft.irfft(np.fft.rfft(h) * np.fft.rfft(x, h.size), h.size)
    level = 20 * np.log10(np.abs(n).max() / np.sqrt(np.mean(n ** 2)))
    return level > threshold, level, SignalObj(n, 'time', ir.samplingRate)


def _log_crossfade(f, f0, f1):
    # 0 below f0, 1 above f1, raised cosine on a logarithmic axis between.
    with np.errstate(divide='ignore'):
        u = np.clip(np.log(f / f0) / np.log(f1 / f0), 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * np.nan_to_num(u))


def invert_spectrum(signal, freqRange, minimum_phase=False):
    """
    Regularized inverse of the spectrum of signal, C = conj(H) / (|H|^2 +
    eps) (Kirkeby, Rubak, Nelson and Farina, Design of cross-talk
    cancellation networks by using fast deconvolution, 1999), with eps
    negligible inside freqRange = [fmin, fmax] and half the largest |H|^2
    outside it, crossfaded over half an octave, the rule of the
    regularization of ImpulsiveResponse. With minimum_phase, the inverse
    gets the minimum phase of its magnitude, as
    ita_invert_spk_minphase_regularization.
    """
    x = np.asarray(signal.timeSignal, dtype=float)
    n = x.shape[0]
    H = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, 1 / signal.samplingRate)
    fmin, fmax = freqRange
    outside = 1 - _log_crossfade(f, fmin / np.sqrt(2), fmin) \
        + _log_crossfade(f, fmax, fmax * np.sqrt(2))
    eps = (1e-10 + outside[:, None]) * (np.abs(H) ** 2).max(axis=0) / 2
    C = np.conj(H) / (np.abs(H) ** 2 + eps)
    if minimum_phase:
        # Homomorphic: fold the real cepstrum of log |C| onto positive times.
        c = np.fft.irfft(np.log(np.abs(C)), n, axis=0)
        fold = np.zeros(n)
        fold[0] = 1
        fold[1:(n + 1) // 2] = 2
        if n % 2 == 0:
            fold[n // 2] = 1
        C = np.exp(np.fft.rfft(c * fold[:, None], axis=0))
    return SignalObj(np.fft.irfft(C, n, axis=0), 'time', signal.samplingRate)


def merge_near_far(near, far, crossover, order=4, port=None, port_ratio=1.):
    """
    Loudspeaker response from a near-field measurement below crossover (Hz)
    and a far-field one above it, as
    ita_add_nearfield_farfield_measurements (Keele, Low-frequency
    loudspeaker assessment by nearfield sound-pressure measurement, J.
    Audio Eng. Soc. 22(3), 1974). The near-field response, with the port
    one times port_ratio (port radius over diaphragm radius) added for a
    vented box, is delayed to the far-field one (find_delay) and scaled to
    its level over the third octave around the crossover, and the two are
    crossfaded with the power-complementary weights 1 / (1 + (f/fc)^(2
    order)) and its complement, which add up to 1.
    """
    fs = far.samplingRate
    m = np.asarray(near.timeSignal[:, 0], dtype=float)
    if port is not None:
        m = m + port_ratio * np.asarray(port.timeSignal[:, 0], dtype=float)
    x = np.asarray(far.timeSignal[:, 0], dtype=float)
    n = x.size
    f = np.fft.rfftfreq(n, 1 / fs)
    lag = np.argmax(ss.correlate(x, m, mode='full', method='fft')) - (m.size - 1)
    M = np.fft.rfft(m, n) * np.exp(-2j * np.pi * f * lag / fs)
    X = np.fft.rfft(x)
    band = (f >= crossover * 2 ** (-1 / 6)) & (f <= crossover * 2 ** (1 / 6))
    M *= np.sqrt(np.sum(np.abs(X[band]) ** 2) / np.sum(np.abs(M[band]) ** 2))
    low = 1 / (1 + (f / crossover) ** (2 * order))
    return SignalObj(np.fft.irfft(low * M + (1 - low) * X, n), 'time', fs)


def bark_warping(samplingRate):
    """
    Warping coefficient that makes the frequency axis approach the Bark
    scale, 0.8517 sqrt(arctan(0.06583 fs / 1000)) - 0.1916 (Smith and Abel,
    Bark and ERB bilinear transforms, IEEE Trans. Speech Audio Process. 7(6),
    1999), as the barkwarp of the ITA-Toolbox.
    """
    return 0.8517 * np.sqrt(np.arctan(0.06583 * samplingRate / 1000)) - 0.1916


def warp(signal, lam, numSamples=None):
    """
    Frequency warped signal, as ita_audio_warp: every delay z^-1 of the
    signal, taken as an FIR filter, is replaced by the allpass
    (z^-1 - lam) / (1 - lam z^-1), so that the new spectrum at omega is the
    old one at omega + 2 arctan(lam sin(omega) / (1 - lam cos(omega))).
    Positive lam stretches the low frequencies; warp with -lam undoes it.
    The result has numSamples samples (those of signal if None).

    >>> w = pytta.warp(ir, pytta.bark_warping(ir.samplingRate))
    """
    x = np.asarray(signal.timeSignal, dtype=float)
    n = numSamples or x.shape[0]
    y = np.zeros((n, x.shape[1]))
    g = np.zeros(n)
    g[0] = 1
    for k in range(x.shape[0]):
        y += g[:, None] * x[k]
        g = ss.lfilter([-lam, 1], [1, -lam], g)
    return SignalObj(y, 'time', signal.samplingRate)


def warp_zpk(z, p, k, lam):
    """
    Zeros, poles and gain of the filter H(D(z)) with z^-1 replaced by the
    allpass D(z) = (z^-1 - lam) / (1 - lam z^-1), as ita_zpk_warp: each root
    r moves to (r + lam) / (1 + lam r), and the difference between the
    numbers of zeros and poles leaves roots at lam. warp_zpk with -lam undoes
    it.
    """
    z, p = np.atleast_1d(z).astype(complex), np.atleast_1d(p).astype(complex)
    k = k * np.prod(1 + lam * z) / np.prod(1 + lam * p)
    extra = np.full(abs(z.size - p.size), lam, dtype=complex)
    zw, pw = (z + lam) / (1 + lam * z), (p + lam) / (1 + lam * p)
    if z.size > p.size:
        pw = np.r_[pw, extra]
    else:
        zw = np.r_[zw, extra]
    return zw, pw, k.real if abs(k.imag) <= 1e-12 * abs(k) else k


def prony(h, nb, na):
    """
    Coefficients b (nb + 1) and a (na + 1, a[0] = 1) of the IIR filter whose
    impulse response best matches h, by Prony's method (Parks and Burrus,
    Digital Filter Design, 1987), as the time domain of ita_prony_analysis:
    the denominator fits the samples after nb by least squares, and the
    numerator then matches the first nb + 1 samples exactly.
    scipy.signal.tf2zpk(b, a) gives the zeros and poles.

    >>> b, a = pytta.prony(ir.timeSignal[:, 0], 8, 8)
    """
    from scipy.linalg import toeplitz
    h = np.asarray(h)
    if h.size <= max(nb, na):
        h = np.pad(h, (0, max(nb, na) + 1 - h.size))
    H = toeplitz(h, np.r_[h[0], np.zeros(na)])
    a = np.r_[1, -np.linalg.lstsq(H[nb + 1:, 1:], H[nb + 1:, 0], rcond=None)[0]]
    return H[:nb + 1] @ a, a


def invfreqz(H, w, nb, na, weights=None):
    """
    Real coefficients b (nb + 1) and a (na + 1, a[0] = 1) of the IIR filter
    whose frequency response fits H at the normalized frequencies w (rad per
    sample), by the linear equation error method of Levi (Complex curve
    fitting, IRE Trans. Automatic Control 4(1), 1959), minimizing
    sum weights |B(w) - H(w) A(w)|^2, as MATLAB's invfreqz without
    iterations and the frequency domain of ita_prony_analysis.

    >>> b, a = pytta.invfreqz(H, 2 * np.pi * f / fs, 8, 8)
    """
    H, w = np.asarray(H), np.asarray(w)
    E = np.exp(-1j * np.outer(w, np.arange(max(nb, na) + 1)))
    M = np.hstack([E[:, :nb + 1], -H[:, None] * E[:, 1:na + 1]])
    sw = np.sqrt(np.ones(w.size) if weights is None else np.asarray(weights))[:, None]
    x = np.linalg.lstsq(np.vstack([(sw * M).real, (sw * M).imag]),
                        np.r_[(sw[:, 0] * H).real, (sw[:, 0] * H).imag], rcond=None)[0]
    return x[:nb + 1], np.r_[1, x[nb + 1:]]


def extrapolate(signal, freqRange, order=4):
    """
    Spectrum outside freqRange = [f1, f2] (Hz) replaced by IIR models of
    order order fitted with invfreqz near each end, as ita_extrapolate:
    the low model on f1 + (0.01 to 0.2)(f2 - f1), used below f1, and the
    high model on f1 + (0.9 to 0.99)(f2 - f1), used above f2, the bands
    of the ITA-Toolbox. A pure delay should be removed first (time_shift),
    since it is not a low-order model.
    """
    fs = signal.samplingRate
    x = np.asarray(signal.timeSignal, dtype=float)
    n = x.shape[0]
    H = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, 1 / fs)
    w = 2 * np.pi * f / fs
    f1, f2 = freqRange
    out = H.copy()
    for edge, (lo, hi) in (('low', (0.01, 0.2)), ('high', (0.9, 0.99))):
        band = (f >= f1 + lo * (f2 - f1)) & (f <= f1 + hi * (f2 - f1))
        outside = f < f1 if edge == 'low' else f > f2
        for c in range(H.shape[1]):
            b, a = invfreqz(H[band, c], w[band], order, order)
            out[outside, c] = ss.freqz(b, a, w[outside])[1]
    return SignalObj(np.fft.irfft(out, n, axis=0), 'time', fs)


def _harmonic_matrix(order, gain, phase):
    # Column k - 1: amplitudes of the harmonics 1 to order that x**k gives for
    # x = gain cos(wt), divided by gain as the sweep deconvolution does, with
    # harmonic m turned by (m - 1) phase (Dietrich 2013, sec. 3.3).
    T = np.zeros((order, order), dtype=complex)
    for k in range(1, order + 1):
        T[:k, k - 1] = np.polynomial.chebyshev.poly2cheb(np.eye(k + 1)[k])[1:] * gain ** (k - 1)
    return T * np.exp(1j * phase * np.arange(order))[:, None]


def power_to_harmonic_series(coefficients, gain=1, phase=0):
    """
    Harmonic series of a power series, as
    ita_nonlinear_power_to_harmonic_series: the amplitudes of the harmonics
    1 to N that y = sum_k a_k x**k gives for a cosine of amplitude gain,
    divided by gain, from the coefficients a_1 to a_N along the first axis
    (scalars, or arrays converted entry by entry). Harmonic m is turned by
    (m - 1) phase, the phase offset of an exponential sweep. The DC term is
    left out.

    >>> pytta.power_to_harmonic_series([1, 1, 1, 1], gain=0.5)  # 1.1875, 0.3125, 0.0625, 0.015625
    """
    a = np.asarray(coefficients)
    T = _harmonic_matrix(a.shape[0], gain, phase)
    h = np.tensordot(T, a, axes=1)
    return h.real if np.isrealobj(a) and phase == 0 else h


def harmonic_to_power_series(harmonics, gain=1, phase=0):
    """
    Power series coefficients a_1 to a_N of the harmonic series 1 to N, the
    inverse of power_to_harmonic_series, as
    ita_nonlinear_harmonic_to_power_series.
    """
    h = np.asarray(harmonics)
    T = _harmonic_matrix(h.shape[0], gain, phase)
    a = np.tensordot(np.linalg.inv(T), h, axes=1)
    return a.real if np.isrealobj(h) and phase == 0 else a


def crossfade_spectra(signal_low, signal_high, freqs):
    """
    Spectrum of signal_low below freqs[0] Hz and of signal_high above
    freqs[1] Hz, with a raised cosine crossfade between them, as
    ita_xfade_spk; a single frequency takes half of each at that bin.
    Both signals need the same sampling rate and number of samples.

    >>> combined = pytta.crossfade_spectra(ir_simulated, ir_measured, [200, 400])
    """
    fs = signal_low.samplingRate
    if fs != signal_high.samplingRate:
        raise ValueError('The signals must have the same sampling rate.')
    f0, f1 = np.broadcast_to(np.asarray(freqs, dtype=float), (2,))
    n = signal_low.timeSignal.shape[0]
    A = np.fft.rfft(signal_low.timeSignal, axis=0)
    if f0 >= fs / 2:
        return SignalObj(np.fft.irfft(A, n, axis=0), 'time', fs)
    B = np.fft.rfft(signal_high.timeSignal, axis=0)
    b0, b1 = (int(round(f * n / fs)) for f in (f0, min(f1, fs / 2)))
    if b1 > b0:
        rise = 0.5 - 0.5 * np.cos(np.pi * np.arange(b1 - b0 + 1) / (b1 - b0))
    else:
        rise = np.array([0.5])
    gain_low = np.zeros(A.shape[0])
    gain_low[:b0] = 1
    gain_low[b0:b1 + 1] = 1 - rise
    gain_high = np.zeros(A.shape[0])
    gain_high[b1 + 1:] = 1
    gain_high[b0:b1 + 1] = rise
    X = A * gain_low[:, None] + B * gain_high[:, None]
    return SignalObj(np.fft.irfft(X, n, axis=0), 'time', fs)


def cut_impulse_train(signal, threshold=0.7, length=1, start_before=0.004, ref_channel=0):
    """
    Impulses of an impulse train cut out of signal, as ita_cutImpulseTrain.
    A sample of ref_channel above the threshold, threshold * L_max +
    (1 - threshold) * L_rms dB, begins an impulse when it comes length
    seconds or more after the previous sample above it; each impulse is
    length seconds from start_before seconds before that sample, cut at the
    same samples in every channel.

    Returns a list with one SignalObj per channel, whose channels are the
    impulses, and the 0-based start samples.

    >>> impulses, starts = pytta.cut_impulse_train(take, length=0.5)
    """
    fs = signal.samplingRate
    x = np.asarray(signal.timeSignal)
    n = x.shape[0]
    nlen, nbefore = round(length * fs), round(start_before * fs)
    ref = x[:, ref_channel]
    with np.errstate(divide='ignore'):
        level = 20 * np.log10(np.abs(ref))
    thr = threshold * level.max() + (1 - threshold) * 10 * np.log10(np.mean(ref ** 2))
    idx = np.flatnonzero(level > thr)
    idx = idx[idx <= n - nlen + nbefore]
    idx = idx[np.r_[True, np.diff(idx) >= nlen]] if idx.size else idx
    starts = idx[idx >= nbefore] - nbefore
    take = starts[:, None] + np.arange(nlen)
    return [SignalObj(x[take, c].T, 'time', fs) for c in range(x.shape[1])], starts


def velocity_from_pressure_gradient(signal, distance, rho=1.2, c=343, normalized=False):
    """
    Particle velocity (m/s) midway between two microphones distance m apart
    from the pressures of channels 0 and 1 (Pa), by Euler's equation,
    v = (p0 - p1) / (j omega rho distance), as
    ita_velocity_from_pressure_gradient; times rho c if normalized. The DC
    bin is set to zero.

    >>> v = pytta.velocity_from_pressure_gradient(pp_probe, 0.012)
    """
    x = np.asarray(signal.timeSignal, dtype=float)
    if x.shape[1] != 2:
        raise ValueError('The signal must have two channels.')
    n = x.shape[0]
    P = np.fft.rfft(x[:, 0] - x[:, 1])
    omega = 2 * np.pi * np.fft.rfftfreq(n, 1 / signal.samplingRate)
    V = np.zeros_like(P)
    V[1:] = P[1:] / (1j * omega[1:] * rho * distance)
    if normalized:
        V *= rho * c
    return SignalObj(np.fft.irfft(V, n), 'time', signal.samplingRate)


def _soundfile():
    try:
        import soundfile
    except ImportError as err:          # pragma: no cover
        raise ImportError('read_audio, write_audio and read_blockwise need the '
                          'soundfile package (pip install soundfile).') from err
    return soundfile


def read_audio(fileName, dtype='float64'):
    """
    SignalObj from an audio file in any format that libsndfile reads (WAV,
    FLAC, CAF, AU, AIFF, OGG, ...), as ita_read_flac, ita_read_caf and
    ita_read_au; needs the soundfile package.

    >>> sig = pytta.read_audio('take.flac')
    """
    sf = _soundfile()
    data, fs = sf.read(fileName, dtype=dtype, always_2d=True)
    return SignalObj(data, 'time', fs)


def write_audio(fileName, signal, subtype=None):
    """
    signal to an audio file whose format follows the extension (WAV, FLAC,
    CAF, AU, AIFF, ...), with the libsndfile subtype (e.g. 'PCM_24',
    'FLOAT'; the default of the format if None); needs soundfile.

    >>> pytta.write_audio('take.flac', sig, 'PCM_24')
    """
    sf = _soundfile()
    sf.write(fileName, np.asarray(signal.timeSignal, dtype=float),
             int(signal.samplingRate), subtype=subtype)


def read_blockwise(fileName, block_size, overlap=0):
    """
    Generator of SignalObj blocks of block_size samples from an audio file,
    consecutive blocks sharing overlap samples, as ita_read_blockwise; the
    last block is shorter when the file ends. Needs soundfile.

    >>> for block in pytta.read_blockwise('long.wav', 48000): ...
    """
    sf = _soundfile()
    fs = sf.info(fileName).samplerate
    for block in sf.blocks(fileName, blocksize=block_size, overlap=overlap,
                           always_2d=True, dtype='float64'):
        yield SignalObj(block, 'time', fs)


def write_txt(fileName, signal, domain='time', delimiter=' '):
    """
    signal to a text file as ita_write_txt: the time (s) or frequency (Hz)
    in the first column, then each channel, in the time domain its samples
    and in the frequency domain the real and imaginary parts of its DFT
    (numpy.fft.rfft of timeSignal).

    >>> pytta.write_txt('response.txt', ir, domain='freq')
    """
    x = np.asarray(signal.timeSignal, dtype=float)
    if domain == 'time':
        cols = [np.arange(x.shape[0]) / signal.samplingRate] + [x[:, c] for c in range(x.shape[1])]
    elif domain == 'freq':
        X = np.fft.rfft(x, axis=0)
        cols = [np.fft.rfftfreq(x.shape[0], 1 / signal.samplingRate)]
        for c in range(x.shape[1]):
            cols += [X[:, c].real, X[:, c].imag]
    else:
        raise ValueError("domain must be 'time' or 'freq'.")
    np.savetxt(fileName, np.column_stack(cols), delimiter=delimiter, fmt='%.17g')


def read_txt(fileName, domain='time', samplingRate=None, numSamples=None, delimiter=None):
    """
    SignalObj from a text file of write_txt or ita_write_txt. In the time
    domain the sampling rate comes from the time column; in the frequency
    domain it is the last frequency times 2 (numSamples even, 2 (bins - 1),
    unless given).
    """
    data = np.loadtxt(fileName, delimiter=delimiter, ndmin=2)
    if domain == 'time':
        fs = samplingRate or round(1 / (data[1, 0] - data[0, 0]))
        return SignalObj(data[:, 1:], 'time', fs)
    if domain != 'freq':
        raise ValueError("domain must be 'time' or 'freq'.")
    f = data[:, 0]
    n = numSamples or 2 * (f.size - 1)
    fs = samplingRate or round(f[1] * n)
    X = data[:, 1::2] + 1j * data[:, 2::2]
    return SignalObj(np.fft.irfft(X, n, axis=0), 'time', fs)


def psd(signal, signal2=None, block_size=None, fft_size=None):
    """
    Power spectral density of each channel, or the cross power spectral
    density with the same channel of signal2, as ita_psd: Welch averaging
    with a symmetric Hann window of block_size samples (the whole signal by
    default) and half-block overlap, one-sided, in units^2/Hz, with the
    convention of MATLAB cpsd, X conj(Y) (the conjugate of
    scipy.signal.csd).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Frequencies (Hz) and the density, (bins, channels).

    >>> freq, P = pytta.psd(noise, block_size=1024)
    """
    x = np.asarray(signal.timeSignal, dtype=float)
    y = x if signal2 is None else np.asarray(signal2.timeSignal, dtype=float)
    if y.shape != x.shape:
        raise ValueError('both signals need the same length and channels.')
    block = x.shape[0] if block_size is None else int(round(block_size))
    block += block % 2
    if block > x.shape[0]:
        x = np.vstack([x, np.zeros((block - x.shape[0], x.shape[1]))])
        y = np.vstack([y, np.zeros((block - y.shape[0], y.shape[1]))])
    nfft = block if fft_size is None else int(fft_size)
    window = np.hanning(block + 2)[1:-1]          # MATLAB hanning(block)
    f, P = ss.csd(x, y, fs=signal.samplingRate, window=window, noverlap=block // 2,
                  nfft=nfft, axis=0, detrend=False, scaling='density')
    return f, np.conj(P) if signal2 is not None else P.real    # MATLAB cpsd: X conj(Y)


def interpolate_spectrum(signal, n_samples, method='spline', abs_phase=False):
    """
    signal with its spectrum interpolated onto the bins of n_samples, as
    ita_interpolate_spk: the DFT is interpolated between DC and the highest
    frequency (spline, as MATLAB interp1, or linear), on its real and
    imaginary parts or (abs_phase=True) on magnitude and unwrapped phase.
    Meant for energy signals such as impulse responses.

    >>> longer = pytta.interpolate_spectrum(ir, 2 ** 16)
    """
    from scipy.interpolate import CubicSpline
    x = np.asarray(signal.timeSignal, dtype=float)
    X = np.fft.rfft(x, axis=0)
    old = np.linspace(0, 1, X.shape[0])
    new = np.linspace(0, 1, n_samples // 2 + 1)
    if method == 'spline':
        interp = lambda v: CubicSpline(old, v, axis=0)(new)
    elif method == 'linear':
        interp = lambda v: np.column_stack([np.interp(new, old, v[:, c]) for c in range(v.shape[1])])
    else:
        raise ValueError("method must be 'spline' or 'linear'.")
    if abs_phase:
        Y = interp(np.abs(X)) * np.exp(1j * interp(np.unwrap(np.angle(X), axis=0)))
    else:
        Y = interp(X.real) + 1j * interp(X.imag)
    newSignal = SignalObj(np.fft.irfft(Y, n_samples, axis=0), 'time',
                          samplingRate=signal.samplingRate, signalType=signal.signalType)
    newSignal.channels = cp.deepcopy(signal.channels)
    return newSignal


def extrapolate_low(signal, freq_range, slope=0.):
    """
    Spectrum below freq_range replaced by an extrapolation, as the lower
    part of ita_spk_extrapolate: the mean magnitude of freq_range, changed
    by slope dB per decade towards low frequencies, and a linear phase from
    the mean group delay of freq_range, joined to the phase at its lower
    end.

    Parameters
    ----------
    freq_range : (float, float)
        Frequencies (Hz) whose magnitude and group delay are averaged.
    slope : float
        dB per decade below freq_range. Default 0.

    >>> fixed = pytta.extrapolate_low(ir, (80, 160))
    """
    x = np.asarray(signal.timeSignal, dtype=float)
    n = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, 1 / signal.samplingRate)
    lo = int(np.flatnonzero(f > min(freq_range))[0])
    hi = int(np.flatnonzero(f > max(freq_range))[0])
    df = f[1]
    band = X[lo:hi + 1]
    gdelay = -np.diff(np.unwrap(np.angle(X), axis=0), axis=0) / (2 * np.pi * df)
    tau = np.mean(gdelay[lo:hi + 1], axis=0)
    fl = f[:lo + 1, None]
    phase = -2 * np.pi * tau * fl
    phase = phase - (phase[-1] - np.angle(band[0]))
    with np.errstate(divide='ignore', invalid='ignore'):
        shape = 10 ** (slope * (np.log10(fl) - np.log10(fl[-1])) / 20)
    shape[0] = 0 if slope > 0 else shape[min(1, lo)]   # DC: limit, or held for slope <= 0
    X[:lo + 1] = np.mean(np.abs(band), axis=0) * shape * np.exp(1j * phase)
    X[0] = X[0].real
    newSignal = SignalObj(np.fft.irfft(X, n, axis=0), 'time',
                          samplingRate=signal.samplingRate, signalType=signal.signalType)
    newSignal.channels = cp.deepcopy(signal.channels)
    return newSignal


def _apply_response(signal, response_of):
    """signal filtered by the complex response response_of(f) at its rfft bins."""
    x = np.asarray(signal.timeSignal, dtype=float)
    n = x.shape[0]
    f = np.fft.rfftfreq(n, 1 / signal.samplingRate)
    y = np.fft.irfft(np.fft.rfft(x, axis=0) * response_of(f)[:, None], n, axis=0)
    newSignal = SignalObj(y, 'time', samplingRate=signal.samplingRate,
                          signalType=signal.signalType)
    newSignal.channels = cp.deepcopy(signal.channels)
    return newSignal


def butterworth(signal, cutoff, btype='low', order=8, linkwitz_riley=False,
                zero_phase=False):
    """
    signal filtered by a Butterworth or Linkwitz-Riley filter, applied to
    its spectrum (circular), as ita_butterFR, ita_liriFR and ita_filter_LiRi.

    Parameters
    ----------
    signal : SignalObj
    cutoff : float or (float, float)
        Cut-off frequency in Hz, or the two of a band-pass.
    btype : str
        'low', 'high' or 'band'.
    order : int
        Order of the filter; a Linkwitz-Riley filter of order 2k is a
        Butterworth filter of order k squared. Default 8.
    linkwitz_riley : bool
        Linkwitz-Riley instead of Butterworth. Default False.
    zero_phase : bool
        Keep only the magnitude of the response. Default False.

    The response is evaluated at the frequencies of the FFT bins of the
    signal (ita_butterFR samples it with MATLAB freqz(Hd, nBins), from 0 to
    pi without pi, so its bins sit slightly off the FFT bins).

    >>> lowpassed = pytta.butterworth(signalObj, 1000)
    """
    if btype not in ('low', 'high', 'band'):
        raise ValueError("btype must be 'low', 'high' or 'band'.")
    k = order // 2 if linkwitz_riley else order
    if linkwitz_riley and order % 2:
        raise ValueError('a Linkwitz-Riley filter has an even order.')
    sos = ss.butter(k, cutoff, btype={'band': 'bandpass'}.get(btype, btype),
                    fs=signal.samplingRate, output='sos')

    def response(f):
        _, h = ss.sosfreqz(sos, f, fs=signal.samplingRate)
        h = h * h if linkwitz_riley else h
        return np.abs(h) if zero_phase else h
    return _apply_response(signal, response)


def _biquad_response(b, a, fs):
    def response(f):
        return ss.freqz(b, a, f, fs=fs)[1]
    return response


def notch(signal, fc, Q=2.):
    """
    signal through a second-order notch at fc (Hz) with quality factor Q,
    applied to its spectrum, as ita_filter_notch (the biquad of the Audio
    EQ Cookbook of R. Bristow-Johnson).

    >>> pytta.notch(signalObj, 50, Q=10)
    """
    w0 = 2 * np.pi * fc / signal.samplingRate
    alpha = np.sin(w0) / (2 * Q)
    b = [1, -2 * np.cos(w0), 1]
    a = [1 + alpha, -2 * np.cos(w0), 1 - alpha]
    return _apply_response(signal, _biquad_response(b, a, signal.samplingRate))


def peak_eq(signal, fc, gain, Q=2.):
    """
    signal through a second-order peak (bell) filter at fc (Hz) with gain
    (dB) and quality factor Q, applied to its spectrum, as ita_filter_peak
    (bilinear design with K = tan(pi fc / fs) and V = 10^(gain / 20)).

    >>> pytta.peak_eq(signalObj, 1000, 6)
    """
    K = np.tan(np.pi * fc / signal.samplingRate)
    V = 10 ** (gain / 20)
    b = [1 + V * K / Q + K ** 2, 2 * (K ** 2 - 1), 1 - V * K / Q + K ** 2]
    a = [1 + K / Q + K ** 2, 2 * (K ** 2 - 1), 1 - K / Q + K ** 2]
    return _apply_response(signal, _biquad_response(b, a, signal.samplingRate))


def _new_signal(signal, data, domain='time'):
    """SignalObj like signal (sampling rate, channels) holding data."""
    newSignal = SignalObj(data, domain, samplingRate=signal.samplingRate,
                          signalType=signal.signalType)
    newSignal.channels = cp.deepcopy(signal.channels)
    return newSignal


def envelope(signal):
    """
    Envelope of each channel, the magnitude of the analytic signal
    (Hilbert transform), computed on twice the signal length to keep the
    circular wrap out of it, as ita_envelope.

    >>> env = pytta.envelope(signalObj)
    """
    x = np.asarray(signal.timeSignal, dtype=float)
    n = x.shape[0]
    return _new_signal(signal, np.abs(ss.hilbert(x, N=2 * n, axis=0)[:n]))


def start_ir(signal, threshold=20.):
    """
    Start of each channel of an impulse response, ISO 3382-1:2009, A.3: the
    last sample before the peak that lies more than threshold dB below the
    peak of the squared response, as ita_start_IR.

    If a sample 6 dB above that level comes earlier, the start moves to the
    last sample below the threshold before it (an oscillating onset). A
    channel whose peak is less than 20 dB above its noise (the mean of the
    last 10 %) or lies in that last 10 % gets 0.

    Returns
    -------
    numpy.ndarray
        0-based start sample of each channel.

    >>> pytta.start_ir(ir)
    """
    x2 = np.asarray(signal.timeSignal, dtype=float) ** 2
    n = x2.shape[0]
    noise = np.mean(x2[int(round(0.9 * n)) - 1:], axis=0)
    peak, at = np.max(x2, axis=0), np.argmax(x2, axis=0)
    thr = -abs(threshold)
    start = np.zeros(x2.shape[1], dtype=int)
    for c in range(x2.shape[1]):
        if peak[c] < 100 * noise[c] or at[c] + 1 > round(0.9 * n) or at[c] == 0:
            continue
        with np.errstate(divide='ignore'):
            level = 10 * np.log10(x2[:at[c] + 1, c]) - 10 * np.log10(peak[c])
        below = np.flatnonzero(level < thr)
        start[c] = below[-1] if below.size else 0
        above = np.flatnonzero(level[:start[c] + 1] > thr + 6)
        if above.size:
            before = np.flatnonzero(level[:above[0] + 1] < thr)
            start[c] = before[-1] if before.size else 0
    return start


def time_shift(signal, shift=None, unit='time'):
    """
    Cyclic shift of each channel, as ita_time_shift.

    Parameters
    ----------
    signal : SignalObj
    shift : float, array_like or None
        Shift in seconds (unit='time') or samples (unit='samples'), one for
        all channels or one per channel; a fraction of a sample is applied
        as a linear phase. None moves the earliest start of the impulse
        response (start_ir) to the first sample.
    unit : str
        'time' (default) or 'samples'.

    >>> shifted = pytta.time_shift(ir, 1e-3)
    """
    x = np.asarray(signal.timeSignal, dtype=float)
    fs = signal.samplingRate
    if shift is None:
        samples = np.full(x.shape[1], -np.min(start_ir(signal)), dtype=float)
    elif unit == 'samples':
        samples = np.broadcast_to(np.asarray(shift, dtype=float), (x.shape[1],))
    elif unit == 'time':
        samples = np.broadcast_to(np.round(np.asarray(shift, dtype=float) * fs), (x.shape[1],))
    else:
        raise ValueError("unit must be 'time' or 'samples'.")
    out = np.empty_like(x)
    n = x.shape[0]
    for c in range(x.shape[1]):
        k = samples[c]
        if k == round(k):
            out[:, c] = np.roll(x[:, c], int(k))
        else:
            f = np.fft.rfftfreq(n)
            out[:, c] = np.fft.irfft(np.fft.rfft(x[:, c]) * np.exp(-2j * np.pi * f * k), n)
    return _new_signal(signal, out)


def append(signal1, signal2):
    """
    signal2 after signal1 in time, as ita_append; both need the same
    sampling rate and number of channels.

    >>> longer = pytta.append(signalObj1, signalObj2)
    """
    if signal1.samplingRate != signal2.samplingRate:
        raise ValueError('the sampling rates differ.')
    if signal1.numChannels != signal2.numChannels:
        raise ValueError('the numbers of channels differ.')
    return _new_signal(signal1, np.vstack([np.asarray(signal1.timeSignal, dtype=float),
                                           np.asarray(signal2.timeSignal, dtype=float)]))


def normalize(signal, domain='time', per_channel=False):
    """
    Signal scaled so that its largest magnitude, in time or in the
    spectrum, is 1, as ita_normalize_dat and ita_normalize_spk.

    Returns
    -------
    (SignalObj, numpy.ndarray)
        The scaled signal and the gain removed (one value, or one per
        channel with per_channel=True).

    >>> normed, peak = pytta.normalize(signalObj)
    """
    x = np.asarray(signal.timeSignal, dtype=float)
    if domain == 'time':
        mag = np.abs(x)
    elif domain == 'freq':
        mag = np.abs(np.fft.rfft(x, axis=0))
    else:
        raise ValueError("domain must be 'time' or 'freq'.")
    gain = np.max(mag, axis=0) if per_channel else np.max(mag)
    return _new_signal(signal, x / gain), np.atleast_1d(gain)


def amplify_to(signal, level):
    """
    Signal scaled so that the mean RMS of its channels is level dB, re
    20 uPa for channels in Pa and re 1 otherwise, as ita_amplify_to.

    >>> at_70dB = pytta.amplify_to(signalObj, 70)
    """
    x = np.asarray(signal.timeSignal, dtype=float)
    ref = 2e-5 if signal.channels[signal.channels.mapping[0]].unit == 'Pa' else 1.
    rms = np.mean(np.sqrt(np.mean(x ** 2, axis=0)))
    return _new_signal(signal, x * 10 ** ((level - 20 * np.log10(rms / ref)) / 20))


def zero_phase(signal):
    """Signal with the magnitude spectrum of signal and zero phase (ita_zerophase)."""
    x = np.asarray(signal.timeSignal, dtype=float)
    return _new_signal(signal, np.fft.irfft(np.abs(np.fft.rfft(x, axis=0)), x.shape[0], axis=0))


def randomize_phase(signal, seed=None):
    """
    Signal with the magnitude spectrum of signal and a uniformly random
    phase (ita_randomize_phase); DC and Nyquist keep their sign so that the
    result is real.
    """
    x = np.asarray(signal.timeSignal, dtype=float)
    X = np.fft.rfft(x, axis=0)
    phase = np.random.default_rng(seed).uniform(0, 2 * np.pi, X.shape)
    phase[0] = np.angle(X[0])
    if x.shape[0] % 2 == 0:
        phase[-1] = np.angle(X[-1])
    return _new_signal(signal, np.fft.irfft(np.abs(X) * np.exp(1j * phase), x.shape[0], axis=0))


def quantize(signal, bits=24):
    """
    Signal rounded to 2^(bits - 1) + 1 intervals across its peak, halves
    away from zero, as ita_quantize.
    """
    x = np.asarray(signal.timeSignal, dtype=float)
    peak = np.max(np.abs(x))
    steps = 2 ** (bits - 1) + 1
    v = x / peak * steps / 2
    v = np.sign(v) * np.floor(np.abs(v) + 0.5)       # halves away from zero
    return _new_signal(signal, v / steps * 2 * peak)


def time_window(signalObj, interval, unit='time', window='hann',
                symmetric=False):
    """
    Apply a time window to a SignalObj and return a new SignalObj.

    The window is built from pairs of bounds. Each pair [start, end] with
    start <= end is a falling slope: ones before start, the falling half of
    the window between start and end (both included), zeros after end. A pair
    with start > end is the mirror image, a rising slope: zeros before end,
    rising half between end and start, ones after start. Several pairs
    multiply. The usual impulse response window is therefore
    [startFade, 0, endFadeStart, endFadeEnd]: a rise at the beginning and a
    fall at the end.

    Parameters (default), (type):
    -----------------------------
        * signalObj (), (SignalObj):
            signal to be windowed. It is not modified;
        * interval (), (list):
            an even number of bounds, in seconds or samples;
        * unit ('time'), (str):
            'time' for seconds (converted with round(t * fs), halves up) or
            'samples' for integer, zero based sample indices;
        * window ('hann'), (str):
            'hann', 'rect' or any window name scipy.signal.get_window
            accepts without parameters. The slope has abs(end - start) + 1
            samples and is one half of a symmetric window twice that long,
            so a 'hann' slope ends at exactly zero;
        * symmetric (False), (bool):
            mirror the window about the centre of the signal, for impulse
            responses that are slightly non causal.

    The behaviour follows ita_time_window of the ITA-Toolbox (RWTH Aachen),
    except that the bounds are not moved to even sample numbers.

    >>> ir = pytta.time_window(ir, [0.02, 0, 0.8, 1.0])
    """
    if not isinstance(signalObj, SignalObj):
        raise TypeError('signalObj must be a SignalObj.')
    bounds = np.asarray(interval, dtype=float).ravel()
    if bounds.size == 0 or bounds.size % 2:
        raise ValueError('interval must hold an even number of bounds.')
    if unit == 'time':
        bounds = np.floor(bounds * signalObj.samplingRate + 0.5)
    elif unit != 'samples':
        raise ValueError("unit must be 'time' or 'samples'.")
    numSamples = signalObj.numSamples
    if np.any(bounds < 0) or np.any(bounds >= numSamples):
        raise ValueError('interval bounds must lie within the signal.')
    bounds = bounds.astype(int)
    name = {'rect': 'boxcar', 'rectwin': 'boxcar'}.get(window, window)
    try:
        ss.get_window(name, 4, fftbins=False)
    except ValueError:
        raise ValueError(f"unknown window '{window}'.")
    fullWindow = np.ones(numSamples)
    for start, end in bounds.reshape(-1, 2):
        length = abs(end - start) + 1
        halves = ss.get_window(name, 2 * length, fftbins=False)
        slope = np.ones(numSamples)
        if start <= end:
            slope[start:end + 1] = halves[length:]
            slope[end + 1:] = 0
        else:
            slope[:end] = 0
            slope[end:start + 1] = halves[:length]
        fullWindow *= slope
    if symmetric:
        fullWindow = fullWindow + fullWindow[::-1]
        fullWindow -= fullWindow.min()
    newSignal = SignalObj(signalObj.timeSignal * fullWindow[:, np.newaxis],
                          'time',
                          samplingRate=signalObj.samplingRate,
                          signalType=signalObj.signalType,
                          freqMin=signalObj.freqMin,
                          freqMax=signalObj.freqMax,
                          comment=signalObj.comment)
    newSignal.channels = cp.deepcopy(signalObj.channels)
    return newSignal



def smooth_frequency(signalObj, nthOct=3):
    """
    Smooth the magnitude spectrum of a SignalObj over fractions of an octave
    and return a new SignalObj. The phase is kept.

    The magnitude of freqSignal is interpolated linearly onto a logarithmic
    frequency grid of 1 cent that spans the 20 octaves below Nyquist, box
    averaged over 1200 / nthOct cents (made odd; the box shrinks
    symmetrically at both ends of the grid), and interpolated back onto the
    frequency bins with a cubic spline. The DC bin is copied unchanged.

    Parameters (default), (type):
    -----------------------------
        * signalObj (), (SignalObj):
            signal whose spectrum is smoothed. It is not modified;
        * nthOct (3), (int or float):
            smoothing width of 1 / nthOct octave: 1 for one octave, 3 for a
            third, 12 for a twelfth.

    The smoothing acts on freqSignal as stored, so the DC and Nyquist bins,
    which PyTTa scales differently from the others, take part with that
    scale. This is the 'LogFreqOctave1' smoothing of ita_smooth in the
    ITA-Toolbox (RWTH Aachen).

    >>> smoothed = pytta.smooth_frequency(ir, nthOct=3)
    """
    if not isinstance(signalObj, SignalObj):
        raise TypeError('signalObj must be a SignalObj.')
    if isinstance(nthOct, bool) or not isinstance(nthOct, (int, float)) \
            or nthOct <= 0:
        raise ValueError('nthOct must be a positive number.')
    samplingRate = signalObj.samplingRate
    freq = signalObj.freqVector
    spectrum = np.array(signalObj.freqSignal, dtype=complex, ndmin=2)
    # logarithmic grid: 1 cent steps over the 20 octaves below Nyquist
    gridFreq = (samplingRate / 2) * 2 ** (np.arange(24001) / 1200 - 20)
    span = int(np.floor(1200 / nthOct + 0.5))
    width = min(span - 1 + span % 2, gridFreq.size)
    halfWidth = (width - 1) // 2
    index = np.arange(gridFreq.size)
    reach = np.minimum(halfWidth, np.minimum(index, index[::-1]))
    smoothed = np.empty_like(spectrum)
    for ch in range(spectrum.shape[1]):
        magnitude = np.interp(gridFreq, freq, np.abs(spectrum[:, ch]))
        summed = np.concatenate(([0.], np.cumsum(magnitude)))
        average = (summed[index + reach + 1] - summed[index - reach]) \
            / (2 * reach + 1)
        spline = CubicSpline(gridFreq[1:], average[1:], bc_type='not-a-knot')
        smoothed[0, ch] = spectrum[0, ch]
        smoothed[1:, ch] = spline(freq[1:]) \
            * np.exp(1j * np.angle(spectrum[1:, ch]))
    newSignal = SignalObj(smoothed, 'freq',
                          samplingRate=samplingRate,
                          numSamples=signalObj.numSamples,
                          signalType=signalObj.signalType,
                          freqMin=signalObj.freqMin,
                          freqMax=signalObj.freqMax,
                          comment=signalObj.comment)
    newSignal.channels = cp.deepcopy(signalObj.channels)
    return newSignal


def enhanced_spectrogram(signalObj, smoothing='readable', n_frames=2000,
                         f_min=20, bins_per_octave=192):
    """
    Enhanced spectrogram: a spectrogram with no window to choose.

    Nine Blackman-Harris windows from 8 to 512 ms (geometric spacing) are
    reassigned (Auger and Flandrin, 1995), the energy of each frame is held
    over the frame step of its window, each map is smoothed, and the maps are
    combined by geometric mean, so that only the energy that all the windows
    place at the same point of the time-frequency plane remains (the idea of
    Cheung and Lim, 1991, applied to reassigned maps).

    Parameters (default), (type):
    -----------------------------
        * signalObj (), (SignalObj):
            signal to analyse. It is not modified;
        * smoothing ('readable'), (str):
            'readable': Gaussian smoothing of 4 ms in time and 1.45 % of the
            frequency, for continuous lines; 'sharp': 1 ms and 1 Hz, for the
            thinnest lines;
        * n_frames (2000), (int):
            largest number of time columns. The time step is 1 ms or more;
        * f_min (20), (float):
            lowest frequency of the output, in Hz;
        * bins_per_octave (192), (int):
            frequency resolution of the logarithmic output grid.

    Return:
    -------
        * t: time of each column, in s, shape (M,);
        * f: frequency of each row, in Hz, shape (F,);
        * L: level of each cell in dB re the channel's dBRef (dB SPL for a
          channel in Pa), shape (F, M, numChannels). A cell holds the mean
          square value of its share of the signal, so the cells of a column
          add up to the level of that instant; a steady tone spread by the
          smoothing over several cells reads lower in each of them (about
          10 dB with 'readable'). Silence sits at -120 dB.

    Author: Sergio Aguirre, 2026. Released in PyTTa under the MIT license.
    The same method is in the SQAT toolbox (GPL-3.0) as
    gui/SQAT_GUI_enhanced_stft.m (https://github.com/ggrecow/SQAT).

    >>> t, f, L = pytta.enhanced_spectrogram(signalObj, smoothing='sharp')
    """
    if not isinstance(signalObj, SignalObj):
        raise TypeError('signalObj must be a SignalObj.')
    L = []
    for chIndex, chNum in enumerate(signalObj.channels.mapping):
        t, f, meanSquare = enhanced_map(signalObj.timeSignal[:, chIndex],
                                        signalObj.samplingRate, smoothing,
                                        n_frames, f_min, bins_per_octave)
        dBRef = signalObj.channels[chNum].dBRef
        L.append(10 * np.log10(meanSquare / dBRef ** 2 + 1e-12))
    return t, f, np.stack(L, axis=2)


def minimum_phase(signalObj):
    """
    Minimum phase version of a SignalObj: the same magnitude spectrum, with
    the phase that puts the energy of each channel as early as possible.
    Returns a new SignalObj of the same length.

    The phase comes from the real cepstrum (Oppenheim and Schafer, Discrete
    Time Signal Processing, ch. 13): the logarithm of the magnitude is taken
    to the time domain, folded onto positive times and brought back. The
    cepstrum is computed on the signal length, so the magnitude is kept
    exactly; its time aliasing leaves a small error in the phase, which
    zero padding the signal before the call reduces.

    >>> mp = pytta.minimum_phase(ir)
    """
    if not isinstance(signalObj, SignalObj):
        raise TypeError('signalObj must be a SignalObj.')
    N = signalObj.numSamples
    spectrum = np.fft.rfft(np.asarray(signalObj.timeSignal, dtype=float),
                           axis=0)
    magnitude = np.abs(spectrum)
    # ponytail: spectral zeros floored at -200 dB re the peak, so log is finite
    magnitude = np.maximum(magnitude, 1e-10 * magnitude.max(axis=0))
    cepstrum = np.fft.irfft(np.log(magnitude), n=N, axis=0)
    fold = np.zeros(N)
    fold[0] = 1
    fold[1:(N + 1) // 2] = 2
    if N % 2 == 0:
        fold[N // 2] = 1
    minPhase = np.fft.irfft(np.exp(np.fft.rfft(cepstrum * fold[:, None],
                                               axis=0)), n=N, axis=0)
    newSignal = SignalObj(minPhase, 'time',
                          samplingRate=signalObj.samplingRate,
                          signalType=signalObj.signalType,
                          freqMin=signalObj.freqMin,
                          freqMax=signalObj.freqMax,
                          comment=signalObj.comment)
    newSignal.channels = cp.deepcopy(signalObj.channels)
    return newSignal


def harmonic_distortion(ir, sweep, numHarmonics=5):
    """
    Harmonic impulse responses and total harmonic distortion (THD) from the
    impulse response of a system measured with an exponential sweep.

    Deconvolution by an exponential sweep puts the harmonic of order k of a
    nonlinear system L ln(k) seconds before the linear response, where L is
    the time the sweep takes to multiply its frequency by e (Farina, AES
    108th Convention, 2000). Each harmonic is cut with the same window: its
    length is the gap between harmonics numHarmonics and numHarmonics + 1,
    the shortest one, 10 % of it before the peak of the harmonic, with a
    Tukey taper of 10 % at each end.

    Parameters (default), (type):
    -----------------------------
        * ir (), (SignalObj):
            impulse response with one channel, from a circular
            deconvolution, so that the harmonics wrap to its end;
        * sweep (), (SignalObj or float):
            the exponential sweep used as excitation, whose rate is read
            from its group delay between 2 freqMin and freqMax / 2, or its
            rate in octaves per second;
        * numHarmonics (5), (int):
            highest harmonic order, 2 or more.

    Return:
    -------
        * harmonics: SignalObj with one channel per order, from 1 (the
          linear response) to numHarmonics;
        * f: excitation frequencies, in Hz, up to fs / (2 numHarmonics);
        * thd: total harmonic distortion at each frequency of f, as a
          ratio: sqrt(sum |H_k(k f)| ** 2 for k >= 2) / |H_1(f)|.

    >>> harmonics, f, thd = pytta.harmonic_distortion(ir, sweep, 3)
    """
    if not isinstance(ir, SignalObj):
        raise TypeError('ir must be a SignalObj.')
    if ir.numChannels != 1:
        raise ValueError('ir must have one channel; see SignalObj.split.')
    if numHarmonics < 2:
        raise ValueError('numHarmonics must be 2 or more.')
    rate = _sweep_rate(sweep) if isinstance(sweep, SignalObj) \
        else float(sweep)
    fs = ir.samplingRate
    L = 1 / (rate * np.log(2))  # s for the frequency to grow by e
    M = int(L * np.log((numHarmonics + 1) / numHarmonics) * fs)
    h = np.asarray(ir.timeSignal[:, 0], dtype=float)
    orders = np.arange(1, numHarmonics + 1)
    # ponytail: positions rounded to the sample; enough for magnitudes
    starts = np.argmax(np.abs(h)) - M // 10 \
        - np.round(L * np.log(orders) * fs).astype(int)
    harmonics = np.take(h, starts[None, :] + np.arange(M)[:, None],
                        mode='wrap') * ss.windows.tukey(M, 0.2)[:, None]
    H = np.abs(np.fft.rfft(harmonics, axis=0))
    i = np.arange((M // 2) // numHarmonics + 1)
    distortion = np.sqrt(sum(H[k * i, k - 1] ** 2
                             for k in orders[1:]))
    with np.errstate(divide='ignore', invalid='ignore'):
        thd = distortion / H[i, 0]
    harmonics = SignalObj(harmonics, 'time', samplingRate=fs,
                          signalType='energy')
    return harmonics, i * fs / M, thd


def _sweep_rate(sweep):
    """Rate of an exponential sweep in octaves per second, from the slope of
    its group delay over the octaves between 2 freqMin and freqMax / 2."""
    x = np.asarray(sweep.timeSignal[:, 0], dtype=float)
    f = np.fft.rfftfreq(x.size, 1 / sweep.samplingRate)
    # group delay with no phase unwrapping: Re{FFT(n x) / FFT(x)} / fs
    delay = np.real(np.fft.rfft(np.arange(x.size) * x)
                    / np.fft.rfft(x)) / sweep.samplingRate
    band = (f >= 2 * sweep.freqMin) & (f <= sweep.freqMax / 2)
    return 1 / np.polyfit(np.log2(f[band]), delay[band], 1)[0]


def sine_distortion(signalObj, freq, numHarmonics=5):
    """
    Total harmonic distortion (THD) and THD plus noise (THD+N) of a
    recorded sine, by channel, as ratios.

    The power of each band comes from a Blackman-Harris periodogram,
    integrated over 6 bins on each side of the fundamental and of its
    harmonics up to numHarmonics (and below Nyquist):

        * THD: rms of the harmonics re the rms of the fundamental;
        * THD+N: rms of everything except the fundamental and DC, re the
          total rms (as in AES17).

    The recording should hold a whole number of cycles or be long enough
    for the window to make that irrelevant (a second or more).

    >>> thd, thdn = pytta.sine_distortion(recording, 1000)
    """
    fs = signalObj.samplingRate
    if not 0 < freq < fs / 2:
        raise ValueError('freq must lie between 0 and fs / 2.')
    f, psd = ss.periodogram(signalObj.timeSignal, fs, window='blackmanharris',
                            scaling='density', axis=0)
    df = f[1]
    half = 6 * df

    def band(center):
        return np.sum(psd[np.abs(f - center) <= half], axis=0) * df

    fundamental = band(freq)
    harmonics = sum(band(k * freq) for k in range(2, numHarmonics + 1)
                    if k * freq + half < fs / 2)
    total = np.sum(psd, axis=0) * df
    rest = total - band(0) - fundamental
    return np.sqrt(harmonics / fundamental), np.sqrt(rest / total)


def snr(signal, noise):
    """
    Signal to noise ratio in dB, by channel: the rms of a recording of the
    signal re the rms of a recording of the noise alone.

    >>> pytta.snr(recording, backgroundNoise)
    """
    return 20 * np.log10(np.std(signal.timeSignal, axis=0)
                         / np.std(noise.timeSignal, axis=0))


def group_delay(signal):
    """
    Group delay of each channel, -d(phase)/d(omega), in seconds.

    Computed with the derivative of the spectrum, which needs no phase
    unwrapping (Oppenheim and Schafer, Discrete Time Signal Processing,
    ch. 5 and 13):

        tau = Re{ DFT(n x[n]) / DFT(x[n]) } / fs

    The DFT is circular, so the signal should decay within its length.
    Bins where the spectrum is zero give NaN.

    Parameters
    ----------
    signal : SignalObj

    Returns
    -------
    numpy.ndarray
        Group delay with shape (numBins, numChannels), at signal.freqVector.

    >>> tau = pytta.group_delay(ir)
    """
    if not isinstance(signal, SignalObj):
        raise TypeError('signal must be a SignalObj.')
    x = np.asarray(signal.timeSignal, dtype=float)
    n = np.arange(signal.numSamples)[:, None]
    X = np.fft.rfft(x, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = np.real(np.fft.rfft(n * x, axis=0) / X)
    tau[X == 0] = np.nan
    return tau / signal.samplingRate


def coherence(x, y, winSize=1024):
    """
    Magnitude squared coherence |Sxy|^2 / (Sxx Syy) between two SignalObj.

    Welch estimate from scipy.signal.coherence, Hann window of winSize
    samples with 50 % overlap. Channels are paired one to one; a signal
    with one channel is paired with every channel of the other.

    Parameters
    ----------
    x, y : SignalObj
        Same sampling rate and number of samples.
    winSize : int, optional
        Welch segment length in samples. The default is 1024.

    Returns
    -------
    freq : numpy.ndarray
        Frequencies in Hz, winSize // 2 + 1 bins.
    gamma2 : numpy.ndarray
        Coherence with shape (numBins, numChannels), between 0 and 1.

    >>> freq, gamma2 = pytta.coherence(excitation, response)
    """
    if not isinstance(x, SignalObj) or not isinstance(y, SignalObj):
        raise TypeError('x and y must be SignalObj.')
    if x.samplingRate != y.samplingRate or x.numSamples != y.numSamples:
        raise ValueError('x and y must have the same sampling rate and '
                         'number of samples.')
    return ss.coherence(np.asarray(x.timeSignal, dtype=float),
                        np.asarray(y.timeSignal, dtype=float),
                        fs=x.samplingRate, window='hann', nperseg=winSize,
                        axis=0)


def cepstrum(signal, kind='complex'):
    """
    Real or complex cepstrum of each channel (Oppenheim and Schafer,
    Discrete Time Signal Processing, ch. 13).

    The real cepstrum is IDFT(log|X|). The complex cepstrum is
    IDFT(log|X| + j arg X), with arg X unwrapped and its linear part, an
    integer delay in samples, removed first and returned so icepstrum can
    put it back. As in Oppenheim and Schafer, the sign of the gain is not
    represented: the input should have a positive gain at 0 Hz.

    Both are computed on the signal length, so they are time aliased;
    zero padding the signal before the call reduces it. Spectral zeros are
    floored at -200 dB re the peak so the logarithm is finite, as in
    minimum_phase.

    Parameters
    ----------
    signal : SignalObj
    kind : str, optional
        'complex' (default) or 'real'.

    Returns
    -------
    SignalObj
        The cepstrum, for kind='real'.
    (SignalObj, numpy.ndarray)
        The cepstrum and the delay removed from each channel, in samples,
        for kind='complex'.

    >>> c = pytta.cepstrum(ir, 'real')
    >>> chat, delay = pytta.cepstrum(ir)
    >>> ir2 = pytta.icepstrum(chat, delay)
    """
    if not isinstance(signal, SignalObj):
        raise TypeError('signal must be a SignalObj.')
    if kind not in ('complex', 'real'):
        raise ValueError("kind must be 'complex' or 'real'.")
    N = signal.numSamples
    X = np.fft.rfft(np.asarray(signal.timeSignal, dtype=float), axis=0)
    magnitude = np.abs(X)
    magnitude = np.maximum(magnitude, 1e-10 * magnitude.max(axis=0))
    if kind == 'real':
        return _cepstrum_signal(signal, np.fft.irfft(np.log(magnitude), n=N,
                                                     axis=0))
    w = 2 * np.pi * np.arange(X.shape[0])[:, None] / N
    phase = np.unwrap(np.angle(X), axis=0)
    delay = np.round(-phase[-1] / w[-1]).astype(int)
    logX = np.log(magnitude) + 1j * (phase + w * delay)
    return _cepstrum_signal(signal, np.fft.irfft(logX, n=N, axis=0)), delay


def icepstrum(cepstrumObj, delay=0):
    """
    Inverse complex cepstrum: the signal whose cepstrum and delay, from
    cepstrum(signal), are given.

    Parameters
    ----------
    cepstrumObj : SignalObj
        Complex cepstrum.
    delay : int or array_like, optional
        Delay in samples to put back, one per channel or one for all.
        The default is 0.

    Returns
    -------
    SignalObj

    >>> ir2 = pytta.icepstrum(*pytta.cepstrum(ir))
    """
    if not isinstance(cepstrumObj, SignalObj):
        raise TypeError('cepstrumObj must be a SignalObj.')
    N = cepstrumObj.numSamples
    logX = np.fft.rfft(np.asarray(cepstrumObj.timeSignal, dtype=float),
                       axis=0)
    w = 2 * np.pi * np.arange(logX.shape[0])[:, None] / N
    X = np.exp(logX - 1j * w * np.asarray(delay))
    return _cepstrum_signal(cepstrumObj, np.fft.irfft(X, n=N, axis=0))


def _cepstrum_signal(signal, data):
    newSignal = SignalObj(data, 'time', samplingRate=signal.samplingRate,
                          signalType='energy')
    newSignal.channels = cp.deepcopy(signal.channels)
    return newSignal


def mls_deconvolution(response, order):
    """
    Impulse response from the response to a periodic MLS, by periodic
    cross-correlation with the sequence of pytta.generate.mls(order).

    The response must start with the excitation. With two or more periods,
    the first one (the transient) is dropped and the others are averaged;
    a single period is taken as already periodic. The correlation is
    normalized by L + 1, L = 2^order - 1, which returns
    h - sum(h) / (L + 1): the impulse response plus a constant of order
    1 / L (Rife and Vanderkooy 1989; Peltonen 2000, eq. 2.22).

    Parameters
    ----------
    response : SignalObj
        Recorded response, at least L samples per channel.
    order : int
        MLS order used for the excitation.

    Returns
    -------
    SignalObj
        Impulse response, L samples per channel.

    >>> ir = pytta.mls_deconvolution(recording, 16)
    """
    if not isinstance(response, SignalObj):
        raise TypeError('response must be a SignalObj.')
    L = 2**order - 1
    periods = response.numSamples // L
    if periods < 1:
        raise ValueError('response is shorter than one MLS period.')
    y = np.asarray(response.timeSignal[:periods * L], dtype=float)
    y = y.reshape(periods, L, -1)[1 if periods > 1 else 0:].mean(axis=0)
    z = mls(order).timeSignal[:, :1].astype(float)
    ir = np.fft.irfft(np.fft.rfft(y, axis=0)
                      * np.conj(np.fft.rfft(z, axis=0)), n=L, axis=0)
    newSignal = SignalObj(ir / (L + 1), 'time',
                          samplingRate=response.samplingRate,
                          signalType='energy')
    newSignal.channels = cp.deepcopy(response.channels)
    return newSignal

def fft_degree(*args,**kwargs):
    """
    DEPRECATED
    ----------
        Being replaced by pytta.utils.maths.fft_degree on version 0.1.0.

    Power-of-two value that can be used to calculate the total number of samples of the signal.

        >>> numSamples = 2**fftDegree

    Parameters
    ----------
        * timeLength (float = 0):
            Value, in seconds, of the time duration of the signal or
            recording.

        * samplingRate (int = 1):
            Value, in samples per second, that the data will be captured
            or emitted.

    Returns
    -------
        fftDegree (float = 0):
            Power of 2 that can be used to calculate number of samples.

    """
    warn(DeprecationWarning("Function 'pytta.fft_degree' is DEPRECATED and " +
                            "being replaced by pytta.utils.maths.fft_degree" +
                            " on version 0.1.0"))
    return new_fft_degree(*args, **kwargs)

def plot_time(*sigObjs, xLabel:str=None, yLabel:str=None, yLim:list=None,
              xLim:list=None, title:str=None, decimalSep:str=',',
              timeUnit:str='s'):
    """Plot provided SignalObjs togheter in time domain.

    Saves xLabel, yLabel, and title when provided for the next plots.

    Parameters (default), (type):
    -----------

        * sigObjs (), (SignalObj):
            non-keyworded input arguments with N SignalObjs.

        * xLabel (None), (str):
            x axis label.

        * yLabel (None), (str):
            y axis label.

        * yLim (None), (list):
            inferior and superior limits.

            >>> yLim = [-100, 100]

        * xLim (None), (str):
            left and right limits.

            >>> xLim = [0, 15]

        * title (None), (str):
            plot title.

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

        * timeUnit ('s'), (str):
            'ms' or 's'.

    Return:
    --------

        matplotlib.figure.Figure object.
    """
    realSigObjs = _remove_non_(SignalObj, sigObjs, msgPrefix='plot_time:')
    if len(realSigObjs) > 0:
        fig = plot.time(realSigObjs, xLabel, yLabel, yLim, xLim, title,
                        decimalSep, timeUnit)
        return fig
    else:
        return

def plot_time_dB(*sigObjs, xLabel:str=None, yLabel:str=None, yLim:list=None,
              xLim:list=None, title:str=None, decimalSep:str=',',
              timeUnit:str='s'):
    """Plot provided SignalObjs togheter in decibels in time domain.

    Parameters (default), (type):
    -----------

        * sigObjs (), (SignalObj):
            non-keyworded input arguments with N SignalObjs.

        * xLabel ('Time [s]'), (str):
                x axis label.

        * yLabel ('Amplitude'), (str):
            y axis label.

        * yLim (), (list):
            inferior and superior limits.

            >>> yLim = [-100, 100]

        * xLim (), (list):
            left and right limits

            >>> xLim = [0, 15]

        * title (), (str):
            plot title

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

        * timeUnit ('s'), (str):
            'ms' or 's'.


    Return:
    --------

        matplotlib.figure.Figure object.
    """
    realSigObjs = \
        _remove_non_(SignalObj, sigObjs, msgPrefix='plot_time_dB:')
    if len(realSigObjs) > 0:
        fig = plot.time_dB(realSigObjs, xLabel, yLabel, yLim, xLim, title,
                           decimalSep, timeUnit)
        return fig
    else:
        return


def plot_freq(*sigObjs, smooth:bool=False, xLabel:str=None, yLabel:str=None,
              yLim:list=None, xLim:list=None, title:str=None,
              decimalSep:str=','):
    """Plot provided SignalObjs magnitudes togheter in frequency domain.

    Parameters (default), (type):
    -----------------------------

        * sigObjs (), (SignalObj):
            non-keyworded input arguments with N SignalObjs.

        * smooth (False), (bool or number):
            fractional octave smoothing of the magnitude by smooth_frequency:
            True for 1/3 octave, a number n for 1/n octave;
        * xLabel ('Time [s]'), (str):
            x axis label.

        * yLabel ('Amplitude'), (str):
            y axis label.

        * yLim (), (list):
            inferior and superior limits.

            >>> yLim = [-100, 100]

        * xLim (), (list):
            left and right limits

            >>> xLim = [15, 21000]

        * title (), (str):
            plot title

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

    Return:
    --------

        matplotlib.figure.Figure object.
    """
    realSigObjs = \
        _remove_non_(SignalObj, sigObjs, msgPrefix='plot_freq:')
    if len(realSigObjs) > 0:
        fig = plot.freq(_smoothed(realSigObjs, smooth), xLabel, yLabel, yLim,
                        xLim, title, decimalSep)
        return fig
    else:
        return

def _smoothed(sigObjs, smooth):
    """The signals, smoothed by smooth_frequency when smooth asks for it:
    True is 1/3 octave, a number n is 1/n octave."""
    if smooth is False or smooth is None:
        return list(sigObjs)
    nthOct = 3 if smooth is True else smooth
    return [smooth_frequency(sig, nthOct) for sig in sigObjs]


def plot_phase(*sigObjs, unwrap: bool = True, xLabel: str = None,
               yLabel: str = None, yLim: list = None, xLim: list = None,
               title: str = None, decimalSep: str = ','):
    """
    Plot the phase spectrum of the provided SignalObjs, in degrees, on a
    logarithmic frequency axis.

    Parameters (default), (type):
    -----------------------------
        * sigObjs (), (SignalObj): non-keyworded SignalObjs;
        * unwrap (True), (bool): unwrap the phase over frequency;
        * xLabel, yLabel, yLim, xLim, title, decimalSep: as in plot_freq.

    Return:
    --------
        matplotlib.figure.Figure object.
    """
    realSigObjs = _remove_non_(SignalObj, sigObjs, msgPrefix='plot_phase:')
    if len(realSigObjs) > 0:
        return plot.freq(realSigObjs, xLabel, yLabel, yLim, xLim, title,
                         decimalSep, phase=True, unwrap=unwrap)


def plot_bars(*analyses, xLabel:str=None, yLabel:str=None,
              yLim:list=None, xLim:list=None, title:str=None, decimalSep:str=',',
              barWidth:float=0.75, errorStyle:str=None,
              forceZeroCentering:bool=False, overlapBars:bool=False,
              color:list=None):
    """Plot the analysis data in fractinal octave bands.

    Parameters (default), (type):
    -----------------------------

        * analyses (), (SignalObj):
            non-keyworded input arguments with N SignalObjs.

        * xLabel ('Time [s]'), (str):
            x axis label.

        * yLabel ('Amplitude'), (str):
            y axis label.

        * yLim (), (list):
            inferior and superior limits.

            >>> yLim = [-100, 100]

        * xLim (), (list):
            bands limits.

            >>> xLim = [100, 10000]

        * title (), (str):
            plot title

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

        * barWidth (0.75), float:
            width of the bars from one fractional octave band.
            0 < barWidth < 1.

        * errorStyle ('standard'), str:
            error curve style. May be 'laza' or None/'standard'.

        * forceZeroCentering ('False'), bool:
            force centered bars at Y zero.

        * overlapBars ('False'), bool:
            overlap bars. No side by side bars of different data.

        * color (None), list:
            list containing the color of each Analysis.


    Return:
    --------

        matplotlib.figure.Figure object.
    """

    analyses = _remove_non_(Analysis, analyses, msgPrefix='plot_bars:')
    if len(analyses) > 0:
        fig = plot.bars(analyses, xLabel, yLabel, yLim, xLim, title,
            decimalSep, barWidth, errorStyle, forceZeroCentering, overlapBars,
            color)
        return fig
    else:
        return

def plot_spectrogram(*sigObjs, winType:str='hann', winSize:int=1024,
                     overlap:float=0.5, xLabel:str=None, yLabel:str=None,
                     yLim:list=None, xLim:list=None, title:str=None,
                     decimalSep:str=',',
                     enhanced:str=None):
    """
    Plots provided SignalObjs spectrogram.

    Parameters (default), (type):
    -----------------------------

        * sigObjs (), (SignalObj):
            non-keyworded input arguments with N SignalObjs.

        * winType ('hann'), (str):
            window type for the time slicing.

        * winSize (1024), (int):
            window size in samples

        * overlap (0.5), (float):
            overlapping fraction of consecutive windows, 0 to 1

        * xLabel ('Time [s]'), (str):
            x axis label.

        * yLabel ('Frequency [Hz]'), (str):
            y axis label.

        * yLim (), (list):
            inferior and superior frequency limits.

            >>> yLim = [20, 1000]

        * xLim (), (list):
            left and right time limits

            >>> xLim = [1, 3]

        * title (), (str):
            plot title

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil
        * enhanced (None), (str):
            'readable' or 'sharp' plots the enhanced spectrogram
            (pytta.enhanced_spectrogram) on a logarithmic frequency axis;
            winType, winSize and overlap are then not used.

    Return:
    --------

        List of matplotlib.figure.Figure objects for each item in curveData.
    """
    realSigObjs = \
        _remove_non_(SignalObj, sigObjs, msgPrefix='plot_spectrogram:')
    if len(realSigObjs) > 0:
        figs = plot.spectrogram(realSigObjs, winType, winSize,
                                overlap, xLabel, yLabel, xLim, yLim,
                                title, decimalSep, enhanced)
        return figs
    else:
        return

def plot_waterfall(*sigObjs, step=2 ** 9, n=2 ** 13, fmin=None, fmax=None,
                   pmin=None, pmax=None, tmax=None, xaxis='linear',
                   time_tick=None, freq_tick=None, mag_tick=None,
                   tick_fontsize=None, fpad=1, delta=60, dBref=2e-5,
                   fill_value='pmin', fill_below=True, overhead=3,
                   winAlpha=0, plots=['waterfall'], show=True, cmap='jet',
                   alpha=[1, 1], saveFig=False, figRatio=[1, 1, 1],
                   figsize=(950, 950), camera=[2, 1, 2]):
    """
    This function was gently sent by Rinaldi Polese Petrolli.

    # TO DO

    Keyword Arguments:
        step {int} -- [description] (default: {10})
        xLim {list} -- [description] (default: {None})
        Pmin {int} -- [description] (default: {20})
        Pmax {[type]} -- [description] (default: {None})
        tmin {int} -- [description] (default: {0})
        tmax {[type]} -- [description] (default: {None})
        azim {int} -- [description] (default: {-72})
        elev {int} -- [description] (default: {14})
        cmap {str} -- [description] (default: {'jet'})
        winPlot {bool} -- [description] (default: {False})
        waterfallPlot {bool} -- [description] (default: {True})
        fill {bool} -- [description] (default: {True})
        lines {bool} -- [description] (default: {False})
        alpha {int} -- [description] (default: {1})
        figsize {tuple} -- [description] (default: {(20, 8)})
        winAlpha {int} -- [description] (default: {0})
        removeGridLines {bool} -- [description] (default: {False})
        saveFig {bool} -- [description] (default: {False})
        bar {bool} -- [description] (default: {False})
        width {float} -- [description] (default: {0.70})
        size {int} -- [description] (default: {3})
        lcol {[type]} -- [description] (default: {None})
        filtered {bool} -- [description] (default: {True})

    Returns:
        [type] -- [description]
    """
    realSigObjs = \
        _remove_non_(SignalObj, sigObjs, msgPrefix='plot_waterfall:')
    if len(realSigObjs) > 0:
        figs = plot.waterfall(realSigObjs, step, n, fmin, fmax, pmin, pmax,
                              tmax, xaxis, time_tick, freq_tick, mag_tick,
                              tick_fontsize, fpad, delta, dBref, fill_value,
                              fill_below, overhead, winAlpha, plots, show,
                              cmap, alpha, saveFig, figRatio, figsize, camera)
        return figs
    else:
        return

def _remove_non_(dataType, dataSet,
                 msgPrefix:str='_remove_non_SignalObjs:'):
    if isinstance(dataSet, (list, tuple)):
        newDataSet = []
        for idx, item in enumerate(dataSet):
            if isinstance(item, dataType):
                newDataSet.append(item)
            elif isinstance(item, ImpulsiveResponse) and \
                dataType.__name__ == 'SignalObj':
                newDataSet.append(item.systemSignal)
            else:
                print("{}: skipping object {} as it isn't a {}."
                      .format(msgPrefix, idx+1, dataType.__name__))
        if isinstance(dataSet, tuple):
            newDataSet = tuple(newDataSet)
    return newDataSet

def save(fileName: str = time.ctime(time.time()), *PyTTaObjs):
    """
    Main save function for .hdf5 and .pytta files.

    The file format is chose by the extension applied to the fileName. If no
    extension is provided choose the default file format (.hdf5).

    For more information on saving PyTTa objects in .hdf5 format see
    pytta.functions._h5_save documentation.

    For more information on saving PyTTa objects in .pytta format see
    pytta.functions.pytta_save' documentation. (DEPRECATED)
    """
    # default file format
    defaultFormat = '.hdf5'
    # Checking the chosen file format
    if fileName.split('.')[-1] == 'hdf5':
        _h5_save(fileName, *PyTTaObjs)
    elif fileName.split('.')[-1] == 'pytta': # DEPRECATED
        warn(DeprecationWarning("'.pytta' format is DEPRECATED and being " +
                                "replaced by '.hdf5'."))
        pytta_save(fileName, *PyTTaObjs)
    else:
        print("File extension must be '.hdf5'.\n" +
              "Applying the default extension.")
        fileName += defaultFormat
        save(fileName, *PyTTaObjs)


def load(fileName: str):
    """
    Main save function for .pytta and .hdf5 files.
    """
    if fileName.split('.')[-1] == 'hdf5':
        output = _h5_load(fileName)
    elif fileName.split('.')[-1] == 'pytta':
        warn(DeprecationWarning("'.pytta' format is DEPRECATED and being " +
                                "replaced by '.hdf5'."))
        output = pytta_load(fileName)
    else:
        ValueError('pytta.load only works with *.hdf5 or *.pytta files.')
    return output

def pytta_save(fileName: str = time.ctime(time.time()), *PyTTaObjs):
    """
    Saves any number of PyTTaObj subclasses' objects to fileName.pytta file.

    Just calls .save() method of each class and packs them all into a major
    .pytta file along with a Meta.json file containing the fileName of each
    saved object.
    """
    if fileName.split('.')[-1] == 'pytta':
        fileName = fileName.replace('.pytta', '')
    meta = {}
    with zf.ZipFile(fileName + '.pytta', 'w') as zdir:
        for idx, obj in enumerate(PyTTaObjs):
            sobj = obj.pytta_save('obj' + str(idx))
            meta['obj' + str(idx)] = sobj
            zdir.write(sobj)
            os.remove(sobj)
        with open('Meta.json', 'w') as f:
            json.dump(meta, f, indent=4, default=lambda o: o.item())
        zdir.write('Meta.json')
        os.remove('Meta.json')
    return fileName + '.pytta'


def pytta_load(fileName: str):
    """
    Loads .pytta files and parses it's types to the correct objects.
    """
    if fileName.split('.')[-1] == 'pytta':
        with zf.ZipFile(fileName, 'r') as zdir:
            objects = zdir.namelist()
            for obj in objects:
                if obj.split('.')[-1] == 'json':
                    meta = obj
            zdir.extractall()
            output = __parse_load(meta)
    else:
        raise ValueError("pytta_load function only works with *.pytta files")
    return output


def __parse_load(className):
    name = className.split('.')[0]
    jsonFile = open(className, 'r')
    openJson = json.load(jsonFile)
    if name == 'SignalObj':
        openMat = sio.loadmat(openJson['timeSignalAddress'])
        out = SignalObj(openMat['timeSignal'], domain=openJson['lengthDomain'],
                        samplingRate=openJson['samplingRate'],
                        freqMin=openJson['freqLims'][0],
                        freqMax=openJson['freqLims'][1],
                        comment=openJson['comment'])
        out.channels = __parse_channels(openJson['channels'],
                                        out.channels)
        os.remove(openJson['timeSignalAddress'])

    elif name == 'ImpulsiveResponse':
        ir = pytta_load(openJson['SignalAddress']['ir'])
        out = ImpulsiveResponse(ir=ir, **openJson['methodInfo'])
        os.remove(openJson['SignalAddress']['ir'])

    elif name == 'RecMeasure':
        inch = list(1 + np.arange(len(openJson['inChannels'])))
        out = RecMeasure(device=openJson['device'], inChannels=inch,
                         lengthDomain=openJson['lengthDomain'],
                         fftDegree=openJson['fftDegree'],
                         timeLength=openJson['timeLength'],
                         samplingRate=openJson['samplingRate'],
                         freqMin=openJson['freqLims'][0],
                         freqMax=openJson['freqLims'][1],
                         comment=openJson['comment'])
        out.inChannels = __parse_channels(openJson['inChannels'],
                                          out.inChannels)

    elif name == 'PlayRecMeasure':
        inch = list(1 + np.arange(len(openJson['inChannels'])))
        outch = list(1 + np.arange(len(openJson['outChannels'])))
        excit = pytta_load(openJson['excitationAddress'])
        out = PlayRecMeasure(excitation=excit,
                             device=openJson['device'], inChannels=inch,
                             outChannels=outch, comment=openJson['comment'])
        out.inChannels = __parse_channels(openJson['inChannels'],
                                          out.inChannels)
        out.outChannels = __parse_channels(openJson['outChannels'],
                                           out.outChannels)
        os.remove(openJson['excitationAddress'])

    elif name == 'FRFMeasure':
        inch = list(1 + np.arange(len(openJson['inChannels'])))
        outch = list(1 + np.arange(len(openJson['outChannels'])))
        excit = pytta_load(openJson['excitationAddress'])
        out = FRFMeasure(excitation=excit, device=openJson['device'],
                         inChannels=inch,
                         outChannels=outch, comment=openJson['comment'])
        out.inChannels = __parse_channels(openJson['inChannels'],
                                          out.inChannels)
        out.outChannels = __parse_channels(openJson['outChannels'],
                                           out.outChannels)
        os.remove(openJson['excitationAddress'])

    elif name == 'Meta':
        out = []
        for val in openJson.values():
            out.append(pytta_load(val))
            os.remove(val)
    os.remove(className)
    jsonFile.close()
    return out


def __parse_channels(chDict, chList):
    # by position: renumbering a channel must not change which one is next
    for channel, key in zip(chList._channels, chDict.keys()):
        channel.num = key
        channel.unit = chDict[key]['unit']
        channel.name = chDict[key]['name']
        channel.code = chDict[key].get('code', 'Ch' + str(key))
        for name in ('coordinates', 'orientation'):
            if name in chDict[key]:
                setattr(channel, name, CoordinateObj(**chDict[key][name]))
        channel.CF = chDict[key]['calib'][0]
        channel.calibCheck = chDict[key]['calib'][1]
    return chList


def _h5_save(fileName: str, *PyTTaObjs):
    """
    Open an hdf5 file, create groups for each PyTTa object, pass it to
    the own object and it saves itself inside the group.

    >>> pytta._h5_save(fileName, PyTTaObj_1, PyTTaObj_2, ..., PyTTaObj_n)

    Dictionaries can also be passed as a PyTTa object. An hdf5 group will be
    created for each dictionary and its PyTTa objects will be saved. To ensure
    the diciontary name will be saved, create the key 'dictName' inside it with
    its name in a string as the value. This function will take this key and use
    as variable name for the dict.

    Lists can also be passed as a PyTTa object. An hdf5 group will be created
    for each list and its PyTTa objects will be saved. To ensure the list name
    will be saved, append to the list a string containing its name. This
    function will take the first string found and use it as variable name for
    the list.
    """
    # Checking if filename has .hdf5 extension
    if fileName.split('.')[-1] != 'hdf5':
        fileName += '.hdf5'
    with h5py.File(fileName, 'w') as f:
        # Save the version to the HDF5 file
        f.attrs['GENERATED_BY'] = 'PyTTa'
        f.attrs['LONG_DESCR'] = 'HDF5 file generated by the PyTTa toolbox'
        f.attrs['FILE_SYS_VERSION'] = 1
        # Dict for counting equal names for correctly renaming
        totalPObjCount = 0  # Counter for total groups
        savedPObjCount = 0  # Counter for loaded objects
        for idx, pObj in enumerate(PyTTaObjs):
            packTotalPObjCount, packSavedPObjCount = \
                __h5_pack(f, pObj, idx)
            totalPObjCount, savedPObjCount = \
                totalPObjCount + packTotalPObjCount, \
                    savedPObjCount + packSavedPObjCount
    # Final message
    plural1 = 's' if savedPObjCount > 1 else ''
    plural2 = 's' if totalPObjCount > 1 else ''
    print('Saved inside the hdf5 file {} PyTTa object{}'
          .format(savedPObjCount, plural1) +
          ' of {} object{} provided.'.format(totalPObjCount, plural2))
    return fileName


def __h5_pack(rootH5Group, pObj, objDesc):
    """
    __h5_pack packs a PyTTa object or dict into its respective HDF5 group.
    """
    if isinstance(pObj, (SignalObj,
                         ImpulsiveResponse,
                         RecMeasure,
                         PlayRecMeasure,
                         FRFMeasure,
                         Analysis)):
        # Creation name
        if isinstance(objDesc, str):
            creationName = objDesc
        else:
            creationName = pObj.creation_name
        # Check if creation_name was already used
        creationName = __h5_pack_count_and_rename(creationName, rootH5Group)
        # create obj's group
        objH5Group = rootH5Group.create_group(creationName)
        # save the obj inside its group
        pObj._h5_save(objH5Group)
        return (1, 1)

    elif isinstance(pObj, dict):
        # Creation name
        if 'dictName' in pObj:
            creationName = pObj.pop('dictName')
        elif isinstance(objDesc, str):
            creationName = objDesc
        else:
            creationName = 'noNameDict'
        creationName = __h5_pack_count_and_rename(creationName, rootH5Group)
        print("Saving the dict '{}'.".format(creationName))
        # create obj's group
        objH5Group = rootH5Group.create_group(creationName)
        objH5Group.attrs['class'] = 'dict'
        # Saving each key of the dict inside the hdf5 group
        totalPObjCount = 0
        savedPObjCount = 0
        for key, pObjFromDict in pObj.items():
            packTotalPObjCount, packSavedPObjCount = \
                __h5_pack(objH5Group, pObjFromDict, key)
            totalPObjCount, savedPObjCount = \
                totalPObjCount + packTotalPObjCount, \
                    savedPObjCount + packSavedPObjCount
        return (totalPObjCount, savedPObjCount)

    elif isinstance(pObj, list):
        # Creation name
        creationName = None
        for idx, item in enumerate(pObj):
            if isinstance(item, str):
                creationName = item
                pObj.pop(idx)
                continue
        if creationName is None:
            if isinstance(objDesc, str):
                creationName = objDesc
            else:
                creationName = 'noNameList'
        creationName = __h5_pack_count_and_rename(creationName, rootH5Group)
        print("Saving the list '{}'.".format(creationName))
        # create obj's group
        objH5Group = rootH5Group.create_group(creationName)
        objH5Group.attrs['class'] = 'list'
        # Saving each item of the list inside the hdf5 group
        totalPObjCount = 0
        savedPObjCount = 0
        for idx, pObjFromList in enumerate(pObj):
            packTotalPObjCount, packSavedPObjCount = \
                __h5_pack(objH5Group, pObjFromList, str(idx))
            totalPObjCount, savedPObjCount = \
                totalPObjCount + packTotalPObjCount, \
                    savedPObjCount + packSavedPObjCount
        return totalPObjCount, savedPObjCount

    else:
        print("Only PyTTa objects and dicts/lists with PyTTa objects " +
              "can be saved through this function. Skipping " +
              "object '" + str(objDesc) + "'.")
        return (1, 0)


def __h5_pack_count_and_rename(creationName, h5Group):
    # Check if creation_name was already used
    objNameCount = 1
    newCreationName = cp.copy(creationName)
    while newCreationName in h5Group:
        objNameCount += 1
        newCreationName = \
            creationName + '_' + str(objNameCount)
    creationName = newCreationName
    return creationName


def _h5_load(fileName: str):
    """
    Load an hdf5 file and recreate the PyTTa objects.
    """
    # Checking if the file is an hdf5 file
    if fileName.split('.')[-1] != 'hdf5':
        raise ValueError("_h5_load function only works with *.hdf5 files")
    f = h5py.File(fileName, 'r')

    # Check if it is a PyTTa-like hdf5 file
    try:
        if 'GENERATED_BY' not in f.attrs.keys() or  \
                                        f.attrs['GENERATED_BY'] != "PyTTa":
            raise NotImplementedError
    except:
        # raise NotImplementedError("Only PyTTa-like hdf5 files can be loaded.")
        warn(DeprecationWarning("'GENERATED_BY' tag couldn't be found in " +
                                "the .hdf5 file. Still trying to load " +
                                "because of legacy PyTTa HDF5 files."))

    loadedObjects = {}
    objCount = 0  # Counter for loaded objects
    totCount = 0  # Counter for total groups
    for PyTTaObjName, PyTTaObjH5Group in f.items():
        totCount += 1
        try:
            loadedObjects[PyTTaObjName] = __h5_unpack(PyTTaObjH5Group)
            objCount += 1
        except NotImplementedError:
            print("Skipping hdf5 group named {} as it ".format(PyTTaObjName) +
                  "isn't an PyTTa object group.")
    f.close()
    # Final message
    plural1 = 's' if objCount > 1 else ''
    plural2 = 's' if totCount > 1 else ''
    print('Imported {} PyTTa object-like group'.format(objCount) + plural1 +
          ' of {} group'.format(totCount) + plural2 +
          ' inside the hdf5 file.')
    return loadedObjects


def __h5_unpack(objH5Group):
    """
    Unpack an HDF5 group into its respective PyTTa object
    """
    if objH5Group.attrs['class'] == 'SignalObj':
        # PyTTaObj attrs unpacking
        samplingRate = objH5Group.attrs['samplingRate']
        freqMin = _h5.none_parser(objH5Group.attrs['freqMin'])
        freqMax = _h5.none_parser(objH5Group.attrs['freqMax'])
        lengthDomain = objH5Group.attrs['lengthDomain']
        comment = objH5Group.attrs['comment']
        # SignalObj attr unpacking
        channels = eval(objH5Group.attrs['channels'])
        # Added with an if for compatibility issues
        if 'signalType' in objH5Group.attrs:
            signalType = _h5.attr_parser(objH5Group.attrs['signalType'])
        else:
            signalType = 'power'
        # Creating and conforming SignalObj
        SigObj = SignalObj(signalArray=np.array(objH5Group['timeSignal']),
                           domain='time',
                           signalType=signalType,
                           samplingRate=samplingRate,
                           freqMin=freqMin,
                           freqMax=freqMax,
                           comment=comment)
        SigObj.channels = channels
        SigObj.lengthDomain = lengthDomain
        return SigObj

    elif objH5Group.attrs['class'] == 'ImpulsiveResponse':
        systemSignal = __h5_unpack(objH5Group['systemSignal'])
        method = objH5Group.attrs['method']
        winType = objH5Group.attrs['winType']
        winSize = objH5Group.attrs['winSize']
        overlap = objH5Group.attrs['overlap']
        IR = ImpulsiveResponse(method=method,
                               winType=winType,
                               winSize=winSize,
                               overlap=overlap,
                               ir=systemSignal)
        return IR

    elif objH5Group.attrs['class'] == 'RecMeasure':
        # PyTTaObj attrs unpacking
        samplingRate = objH5Group.attrs['samplingRate']
        freqMin = _h5.none_parser(objH5Group.attrs['freqMin'])
        freqMax = _h5.none_parser(objH5Group.attrs['freqMax'])
        comment = objH5Group.attrs['comment']
        lengthDomain = objH5Group.attrs['lengthDomain']
        fftDegree = objH5Group.attrs['fftDegree']
        timeLength = objH5Group.attrs['timeLength']
        # Measurement attrs unpacking
        device = _h5.list_w_int_parser(objH5Group.attrs['device'])
        inChannels = eval(objH5Group.attrs['inChannels'])
        blocking = objH5Group.attrs['blocking']
        # Recreating the object
        rObj = measurement(kind='rec',
                           device=device,
                           inChannels=inChannels,
                           blocking=blocking,
                           samplingRate=samplingRate,
                           freqMin=freqMin,
                           freqMax=freqMax,
                           comment=comment,
                           lengthDomain=lengthDomain,
                           fftDegree=fftDegree,
                           timeLength=timeLength)
        return rObj

    elif objH5Group.attrs['class'] == 'PlayRecMeasure':
        # PyTTaObj attrs unpacking
        samplingRate = objH5Group.attrs['samplingRate']
        freqMin = _h5.none_parser(objH5Group.attrs['freqMin'])
        freqMax = _h5.none_parser(objH5Group.attrs['freqMax'])
        comment = objH5Group.attrs['comment']
        lengthDomain = objH5Group.attrs['lengthDomain']
        fftDegree = objH5Group.attrs['fftDegree']
        timeLength =objH5Group.attrs['timeLength']
        # Measurement attrs unpacking
        device = _h5.list_w_int_parser(objH5Group.attrs['device'])
        inChannels = eval(objH5Group.attrs['inChannels'])
        outChannels = eval(objH5Group.attrs['outChannels'])
        blocking = objH5Group.attrs['blocking']
        # PlayRecMeasure attrs unpacking
        excitation = __h5_unpack(objH5Group['excitation'])
        outputAmplification = objH5Group.attrs['outputAmplification']
        # Recreating the object
        prObj = measurement(kind='playrec',
                            excitation=excitation,
                            outputAmplification=outputAmplification,
                            device=device,
                            inChannels=inChannels,
                            outChannels=outChannels,
                            blocking=blocking,
                            samplingRate=samplingRate,
                            freqMin=freqMin,
                            freqMax=freqMax,
                            comment=comment)
        return prObj

    elif objH5Group.attrs['class'] == 'FRFMeasure':
        # PyTTaObj attrs unpacking
        samplingRate = objH5Group.attrs['samplingRate']
        freqMin = _h5.none_parser(objH5Group.attrs['freqMin'])
        freqMax = _h5.none_parser(objH5Group.attrs['freqMax'])
        comment = objH5Group.attrs['comment']
        lengthDomain = objH5Group.attrs['lengthDomain']
        fftDegree = objH5Group.attrs['fftDegree']
        timeLength = objH5Group.attrs['timeLength']
        # Measurement attrs unpacking
        device = _h5.list_w_int_parser(objH5Group.attrs['device'])
        inChannels = eval(objH5Group.attrs['inChannels'])
        outChannels = eval(objH5Group.attrs['outChannels'])
        blocking = objH5Group.attrs['blocking']
        # PlayRecMeasure attrs unpacking
        excitation = __h5_unpack(objH5Group['excitation'])
        outputAmplification = objH5Group.attrs['outputAmplification']
        # FRFMeasure attrs unpacking
        method = _h5.none_parser(objH5Group.attrs['method'])
        winType = _h5.none_parser(objH5Group.attrs['winType'])
        winSize = _h5.none_parser(objH5Group.attrs['winSize'])
        overlap = _h5.none_parser(objH5Group.attrs['overlap'])
        # Recreating the object
        frfObj = measurement(kind='frf',
                             method=method,
                             winType=winType,
                             winSize=winSize,
                             overlap=overlap,
                             excitation=excitation,
                             outputAmplification=outputAmplification,
                             device=device,
                             inChannels=inChannels,
                             outChannels=outChannels,
                             blocking=blocking,
                             samplingRate=samplingRate,
                             freqMin=freqMin,
                             freqMax=freqMax,
                             comment=comment)
        return frfObj

    elif objH5Group.attrs['class'] == 'Analysis':
        # Analysis attrs unpacking
        anType = _h5.attr_parser(objH5Group.attrs['anType'])
        nthOct = _h5.attr_parser(objH5Group.attrs['nthOct'])
        minBand = _h5.attr_parser(objH5Group.attrs['minBand'])
        maxBand = _h5.attr_parser(objH5Group.attrs['maxBand'])
        comment = _h5.attr_parser(objH5Group.attrs['comment'])
        title = _h5.attr_parser(objH5Group.attrs['title'])
        dataLabel = _h5.attr_parser(objH5Group.attrs['dataLabel'])
        errorLabel = _h5.attr_parser(objH5Group.attrs['errorLabel'])
        xLabel = _h5.attr_parser(objH5Group.attrs['xLabel'])
        yLabel = _h5.attr_parser(objH5Group.attrs['yLabel'])
        # Analysis data unpacking
        data = np.array(objH5Group['data'])
        # If error in save moment was None no group was created for it
        if 'error' in objH5Group:
            error = np.array(objH5Group['error'])
        else:
            error = None
        # Recreating the object
        anObject = Analysis(anType=anType,
                            nthOct=nthOct,
                            minBand=minBand,
                            maxBand=maxBand,
                            data=data,
                            dataLabel=dataLabel,
                            error=error,
                            errorLabel=errorLabel,
                            comment=comment,
                            xLabel=xLabel,
                            yLabel=yLabel,
                            title=title)
        return anObject

    elif objH5Group.attrs['class'] == 'dict':
        dictObj = {}
        for PyTTaObjName, PyTTaObjH5Group in objH5Group.items():
            dictObj[PyTTaObjName] = __h5_unpack(PyTTaObjH5Group)
        return dictObj

    elif objH5Group.attrs['class']  == 'list':
        dictObj = {}
        for idx, PyTTaObjH5Group in objH5Group.items():
            dictObj[int(idx)] = __h5_unpack(PyTTaObjH5Group)
        idxs = [int(item) for item in list(dictObj.keys())]
        maxIdx = max(idxs)
        listObj = []
        for idx in range(maxIdx+1):
            listObj.append(dictObj[idx])
        return listObj

    else:
        raise NotImplementedError

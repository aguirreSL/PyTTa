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
        maxamp = max(np.abs(signal.timeSignal[:, chindex]))
        maxindex = np.where(signal.timeSignal[:, chindex] == np.abs(maxamp))[0]
        maxtime = signal.timeVector[maxindex][0]
        peaks_time.append(maxtime)
    if signal.numChannels > 1:
        return peaks_time
    else:
        return peaks_time[0]

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
def _new_signal(signal, data, domain='time'):
    """SignalObj like signal (sampling rate, channels) holding data."""
    newSignal = SignalObj(data, domain, samplingRate=signal.samplingRate,
                          signalType=signal.signalType)
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

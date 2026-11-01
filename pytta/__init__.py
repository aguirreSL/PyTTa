# -*- coding: utf-8 -*-
"""
This is a package developed to perform acoustic and vibration measurements
and signal analysis.

To get started, try:

    >>> import pytta

    >>> pytta.default()
    >>> pytta.list_devices()

    >>> mySignal = pytta.generate.sweep()
    >>> mySignal.plot_freq()  # same as pytta.plot_freq(mySignal)

Our current dependencies are:

    * Numpy
    * Scipy
    * Matplotlib
    * Numba
    * h5py
    * SoundDevice
    * SoundFile

PyTTa is now a multi-paradigm toolbox, which may change in the future. We aim
from now on to be more Pythonic, and therefore more object-oriented. For now,
we have classes and functions operating together to provide a working
environment for acoustical measurements and post-processing.

The toolbox' structure is presented next.


Sub-packages:
-------------

    * pytta.classes:
        main classes intended to do measurements (Rec, PlayRec and
        FRFMeasurement), handle signals/processed data (SignalObj and
        Analysis), handle streaming functionalities (Monitor and Streaming),
        filter (OctFilter), communicate with some hardware (LJU3EI1050),
        and also intended for new features whose should have an
        object-oriented implementation. The main classes are called from the
        toolbox's top-level (e.g. pytta.SignalObj, pytta.Analysis, ...);

    * pytta.utils:
        contains simple tools which help to keep things modularized and make
        more accessible the reuse of some operations. Intended to hold tools
        (classes and functions) whose operate built-in python classes, NumPy
        arrays, and other stuff not contained by the pytta.classes subpackage;

    * pytta.apps:
        applications built from the toolbox's functionalities. The apps
        are called from the top-level (e.g. pytta.roomir);


Modules:
--------

    * pytta.functions:
        assistant functions intended to manipulate and visualize multiple
        signals and analyses stored as SignalObj and Analysis objects. These
        functions are called from the toolbox's top-level (e.g.
        pytta.plot_time, pytta.plot_waterfall, ...);

    * pytta.generate:
        functions for signal synthesis and measurement configuration;

    * pytta.rooms (DEPRECATED):
        room acoustics parameters calculation according to ISO 3382-1;

    * pytta.iso3741:
        calculations according to the standard;

    * pytta.psychoacoustics:
        loudness (ISO 532-1), sharpness (DIN 45692), roughness (Daniel
        and Weber 1997), fluctuation strength (Osses et al. 2016) and
        tonality (Aures 1985) of a sound pressure signal;
    * pytta.ecma418:
        prominent discrete tones according to ECMA-418-1 (tone-to-noise
        ratio and prominence ratio);
    * pytta.loudspeaker:
        Thiele-Small parameters from the electrical impedance;
    * pytta.sofa:
        read and write SOFA files (AES69-2022) with impulse responses;
    * pytta.din45681:
        tonal audibility of tones in noise (DIN 45681);
    * pytta.vbap:
        vector base amplitude panning for 2D and 3D loudspeaker layouts;

    * pytta.ambisonics:
        higher-order Ambisonics encoding, decoding and rotation;
    * pytta.iso532_2:
        loudness of stationary sounds by the Moore-Glasberg method (ISO 532-2);
    * pytta.absorbers:
        surface impedance and absorption of layered absorbers;
    * pytta.analytic:
        analytic sound fields of point sources, plane waves and a sphere;
    * pytta.fourpole:
        four-pole matrices of lumped elements;
    * pytta.e2611:
        normal incidence sound transmission in a four-microphone tube
        (ASTM E2611);
    * pytta.fem:
        finite elements for the Helmholtz equation in air;
    * pytta.bem:
        boundary elements for radiation into free field;
    * pytta.rays:
        stochastic ray tracing in convex rooms;
    * pytta.mesh:
        meshes for finite and boundary elements;
    * pytta.iso11654:
        rating of sound absorption (ISO 11654);
    * pytta.iso3382_3:
        open-plan offices (ISO 3382-3);


You can find usage examples in the toolbox's examples folder.

For further information, check the sub-packages' and modules' documentation.

The documentation is also available at:

    https://pytta.readthedocs.io/


Created on Fri May 25 2018

@authors:
    João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br
    Matheus Lazarin, matheus.lazarin@eac.ufsm.br

"""

# Importing .py files as submodules
from ._properties import default

# Instantiate the Default parameters to be loaded by other
# methods and function calls

from .classes import SignalObj, ImpulsiveResponse,\
    Measurement, RecMeasure, PlayRecMeasure, FRFMeasure,\
    Streaming, Monitor,\
    OctFilter, weighting,\
    Analysis, RoomAnalysis

from . import _h5utils
from . import _plot
from . import generate
from . import utils
from . import rooms
from . import iso3741
from . import iec60268
from . import iso3382
from . import iso717
from . import bs1770
from . import iec61672
from . import descriptors
from . import response
from . import vibration
from . import iso10534
from . import iso9613
from . import iso354
from . import psychoacoustics
from . import ecma418
from . import loudspeaker
from . import sofa
from . import din45681
from . import vbap
from . import ambisonics
from . import iso532_2
from . import absorbers
from . import analytic
from . import fourpole
from . import e2611
from . import fem
from . import bem
from . import rays
from . import mesh
from . import iso11654
from . import iso3382_3

from .functions import (
    read_wav, write_wav, merge, split, list_devices, print_devices,
    get_device_from_user, fft_convolve, find_delay, corr_coef, resample,
    peak_time, save, load, plot_time, plot_time_dB, plot_freq, plot_bars,
    plot_spectrogram, plot_waterfall, SPL, time_window, smooth_frequency,
    enhanced_spectrogram, minimum_phase, harmonic_distortion, sine_distortion,
    snr, plot_phase, group_delay, coherence, cepstrum, icepstrum,
    mls_deconvolution, envelope, start_ir, time_shift, append, normalize,
    amplify_to, zero_phase, randomize_phase, quantize, butterworth, notch,
    peak_eq, psd, interpolate_spectrum, extrapolate_low, read_audio,
    write_audio, read_blockwise, write_txt, read_txt, crossfade_spectra,
    cut_impulse_train, velocity_from_pressure_gradient, plot_group_delay,
    plot_nyquist, plot_zplane, plot_polar, power_to_harmonic_series,
    harmonic_to_power_series, prony, invfreqz, bark_warping, warp, warp_zpk,
    dither, impulsive_noise, invert_spectrum, merge_near_far,
    multiple_sweep_excitation, multiple_sweep_irs, triggered_average,
    pitch_detection, wiener_hopf, normxcorr2, lifter, smooth_notches,
    extrapolate)

# Must go on v0.1.0 (DEPRECATED)
from .functions import fft_degree

from .apps import roomir

__version__ = '0.1.2a'  # package version

# package submodules and scripts to be called as pytta.something
__all__ = [  # Apps
           'roomir',

           # Submodules
           'generate',
           'utils',
           'iso3741',
           'iso717',
           'bs1770',
           'iec61672',
           'descriptors',
           'response',
           'vibration',
           'iso10534',
           'iso9613',
           'iso354',
           'psychoacoustics',
           'ecma418',
           'loudspeaker',
           'sofa',
           'din45681',
           'vbap',
           'ambisonics',
           'iso532_2',
           'absorbers',
           'analytic',
           'fourpole',
           'e2611',
           'fem',
           'bem',
           'rays',
           'mesh',
           'iso11654',
           'iso3382_3',
           'rooms',

           # Functions
           'merge',
           'split',
           'fft_convolve',
           'read_wav',
           'write_wav',
           'SPL',
           'list_devices',
           'print_devices',
           'get_device_from_user',
           'find_delay',
           'resample',
           'corr_coef',
           'peak_time',
           'plot_phase',
           'iec60268',
           'iso3382',
           'snr',
           'sine_distortion',
           'harmonic_distortion',
           'minimum_phase',
           'enhanced_spectrogram',
           'smooth_frequency',
           'time_window',
           'group_delay',
           'coherence',
           'cepstrum',
           'icepstrum',
           'mls_deconvolution',
           'envelope',
           'start_ir',
           'time_shift',
           'append',
           'normalize',
           'amplify_to',
           'zero_phase',
           'randomize_phase',
           'quantize',
           'butterworth',
           'notch',
           'peak_eq',
           'psd',
           'interpolate_spectrum',
           'extrapolate_low',
           'read_audio',
           'write_audio',
           'read_blockwise',
           'write_txt',
           'read_txt',
           'crossfade_spectra',
           'cut_impulse_train',
           'velocity_from_pressure_gradient',
           'plot_group_delay',
           'plot_nyquist',
           'plot_zplane',
           'plot_polar',
           'power_to_harmonic_series',
           'harmonic_to_power_series',
           'prony',
           'invfreqz',
           'bark_warping',
           'warp',
           'warp_zpk',
           'dither',
           'impulsive_noise',
           'invert_spectrum',
           'merge_near_far',
           'multiple_sweep_excitation',
           'multiple_sweep_irs',
           'triggered_average',
           'pitch_detection',
           'wiener_hopf',
           'normxcorr2',
           'lifter',
           'smooth_notches',
           'extrapolate',
           'save',
           'load',
           'weighting',
           'plot_time',
           'plot_time_dB',
           'plot_freq',
           'plot_bars',
           'plot_spectrogram',
           'plot_waterfall',

           # Classes
           'RecMeasure',
           'PlayRecMeasure',
           'FRFMeasure',
           'SignalObj',
           'ImpulsiveResponse',
           'Analysis',
           'RoomAnalysis',
           'OctFilter',
           'Monitor',
           'Streaming',

           # Objects
           'default']

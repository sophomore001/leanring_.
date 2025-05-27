#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines,unnecessary-lambda-assignment
"""
LISA Instrument module.

Authors:
    Jean-Baptiste Bayle <j2b.bayle@gmail.com>
"""

import re
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
import importlib_metadata

from h5py import File
from scipy.signal import lfilter, kaiserord, firwin
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import cumulative_trapezoid
from packaging.version import Version
from packaging.specifiers import SpecifierSet

from lisaconstants import c

from .containers import ForEachSC
from .containers import ForEachMOSA

from . import dsp
from . import noises

logger = logging.getLogger(__name__)


class Instrument:
    """Represents an instrumental simulation."""
    # pylint: disable=attribute-defined-outside-init

    # Indexing conventions
    SCS = ForEachSC.indices()
    MOSAS = ForEachMOSA.indices()

    # Supported laser locking topologies (L12 as primary)
    # pylint: disable=line-too-long
    LOCK_TOPOLOGIES = {
        'N1': {'12': 'cavity', '23': 'adjacent', '31': 'distant',  '13': 'adjacent', '32': 'adjacent', '21': 'distant'},
        'N2': {'12': 'cavity', '23': 'adjacent', '31': 'distant',  '13': 'adjacent', '32': 'distant',  '21': 'distant'},
        'N3': {'12': 'cavity', '23': 'adjacent', '31': 'adjacent', '13': 'adjacent', '32': 'distant',  '21': 'distant'},
        'N4': {'12': 'cavity', '23': 'distant',  '31': 'distant',  '13': 'adjacent', '32': 'adjacent', '21': 'distant'},
        'N5': {'12': 'cavity', '23': 'distant',  '31': 'distant',  '13': 'adjacent', '32': 'adjacent', '21': 'adjacent'},
        'N6': {'12': 'cavity', '23': 'adjacent', '31': 'adjacent', '13': 'distant',  '32': 'distant',  '21': 'distant'},
    }

    # Index cycles for locking topologies
    INDEX_CYCLES = {
        '12': {'12': '12', '23': '23', '31': '31', '13': '13', '32': '32', '21': '21'},
        '21': {'12': '21', '23': '13', '31': '32', '13': '23', '32': '31', '21': '12'},
        '31': {'12': '23', '23': '31', '31': '12', '13': '21', '32': '13', '21': '32'},
        '32': {'12': '32', '23': '21', '31': '13', '13': '31', '32': '12', '21': '23'},
        '23': {'12': '31', '23': '12', '31': '23', '13': '32', '32': '21', '21': '13'},
        '13': {'12': '13', '23': '32', '31': '21', '13': '12', '32': '23', '21': '31'},
    }

    def __init__(self,
                 # Sampling parameters
                 size=1200, dt=1/4, t0='orbits',
                 # Physics simulation sampling and filtering
                 physics_upsampling=4, aafilter=('kaiser', 240, 1.1, 2.9),
                 # Telemetry sampling
                 telemetry_downsampling=86400 * 4, initial_telemetry_size=0,
                 # Inter-spacecraft propagation
                 orbits='static', orbit_dataset='tps/ppr',
                 gws=None, interpolation=('lagrange', 31),
                 # Artifacts
                 glitches=None,
                 # Laser locking and frequency plan
                 lock='N1-12', fplan='static',
                 laser_asds=30, laser_shape='white+infrared',
                 central_freq=2.816E14, offset_freqs='default',
                 # Laser phase modulation
                 modulation_asds='default', modulation_freqs='default', tdir_modulations=None,
                 # Clocks
                 clock_asds=6.32E-14, clock_offsets=0, clock_freqoffsets='default',
                 clock_freqlindrifts='default', clock_freqquaddrifts='default',
                 # Clock inversion
                 clockinv_tolerance=1E-10, clockinv_maxiter=5,
                 # Optical pathlength noises
                 backlink_asds=3E-12, backlink_fknees=2E-3,
                 testmass_asds=2.4E-15, testmass_fknees=0.4E-3, testmass_fbreak=8E-3,
                 testmass_shape='original', testmass_frelax=0.8E-4,
                 oms_asds=(6.35E-12, 1.25E-11, 1.42E-12, 3.38E-12, 3.32E-12, 7.90E-12), oms_fknees=2E-3,
                 # MOC time correlation
                 moc_time_correlation_asds=0.42,
                 # Tilt-to-length (TTL)
                 ttl_coeffs='default',
                 sc_jitter_asds=(5E-9, 5E-9, 5E-9), sc_jitter_fknees=(8E-4, 8E-4, 8E-4),
                 mosa_jitter_asds=(5E-9, 1e-9), mosa_jitter_fknees=(8E-4, 8E-4), mosa_angles='default',
                 dws_asds=7E-8/335,
                 # Pseudo-ranging
                 ranging_biases=0, ranging_asds=3E-9, prn_ambiguity=None,
                 # Electronic delays
                 electro_delays=(0, 0, 0),
                 # Concurrency
                 concurrent=False):
        """Initialize an instrumental simulation.

        Args:
            size: number of measurement samples to generate
            dt: sampling period [s]
            t0: initial time [s], or 'orbits' to match that of the orbits
            physics_upsampling: ratio of sampling frequencies for physics vs. measurement simulation
            aafilter: antialiasing filter function, list of coefficients, filter design method,
                or None for no filter; to design a filter from a Kaiser window, use a tuple
                ('kaiser', attenuation [dB], f1 [Hz], f2 [Hz]) with f1 < f2 the frequencies defining
                the transition band
            telemetry_downsampling: ratio of sampling frequencies for measurements vs. telemetry events
            initial_telemetry_size: number of telemetry samples before :attr:`lisainstrument.Instrument.t0`
            orbits: path to orbit file, dictionary of constant PPRs for static arms, 'static'
                for a set of static PPRs corresponding to a fit of Keplerian orbits around t = 0,
                or dictionary of PPR time series
            orbit_dataset: datasets to read from orbit file, must be 'tps/ppr' or 'tcb/ltt';
                if set to 'tps/ppr', read proper pseudo-ranges (PPRs) in TPSs (proper times),
                if set to 'tcb/ltt', read light travel times (LTTs) in TCB (coordinate time);
                ignored if no orbit files are used
            gws: path to gravitational-wave file, or dictionary of gravitational-wave responses;
                if ``orbit_dataset`` is ``'tps/ppr'``, we try to read link responses as functions
                of the TPS instead of link responses in the TCB (fallback behavior)
            interpolation: interpolation function or interpolation method and parameters;
                use a tuple ('lagrange', order) with `order` the odd Lagrange interpolation order;
                an arbitrary function should take (x, shift [number of samples]) as parameter
            glitches: path to glitch file, or dictionary of glitch signals per injection point
            lock: pre-defined laser locking configuration ('N1-12' non-swap N1 with 12 primary laser),
                or 'six' for 6 lasers locked on cavities, or a dictionary of locking conditions
            fplan: path to frequency-plan file, dictionary of locking beatnote frequencies [Hz],
                or 'static' for a default set of constant locking beatnote frequencies
            laser_asds: dictionary of amplitude spectral densities for laser noise [Hz/sqrt(Hz)]
            laser_shape: laser noise spectral shape, either 'white' or 'white+infrared'
            central_freq: laser central frequency from which all offsets are computed [Hz]
            offset_freqs: dictionary of laser frequency offsets for unlocked lasers [Hz],
                defined with respect to :attr:`lisainstrument.Instrument.central_freq`,
                or 'default' for a default set of frequency offsets that yield valid beatnote
                frequencies for 'six' lasers locked on cavity and default set of constant PPRs
            modulation_asds: dictionary of amplitude spectral densities for modulation noise
                on each MOSA [s/sqrt(Hz)], or 'default' for a default set of levels with a factor
                10 higher on right-sided MOSAs to account for the frequency distribution system
            modulation_freqs: dictionary of modulation frequencies [Hz], or 'default'
            tdir_modulations: dictionary of callable generators of TDIR assistance modulations
                that take an array of TPS times as argument, or None
            clock_asds: dictionary of clock noise amplitude spectral densities [/sqrt(Hz)]
            clock_offsets: dictionary of clock offsets from TPS [s]
            clock_freqoffsets: dictionary of clock frequency offsets [s/s], or 'default'
            clock_freqlindrifts: dictionary of clock frequency linear drifts [s/s^2], or 'default'
            clock_freqquaddrifts: dictionary of clock frequency quadratic drifts [s/s^3], or 'default'
            clockinv_tolerance: convergence tolerance for clock noise inversion [s]
            clockinv_maxiter: maximum number of iterations for clock noise inversion
            backlink_asds: dictionary of amplitude spectral densities for backlink noise [m/sqrt(Hz)]
            backlink_fknees: dictionary of cutoff frequencied for backlink noise [Hz]
            testmass_asds: dictionary of amplitude spectral densities for test-mass noise [ms^(-2)/sqrt(Hz)]
            testmass_fknees: dictionary of low-frequency cutoff frequencies for test-mass noise [Hz]
            testmass_fbreak: dictionary of high-frequency break frequencies for test-mass noise [Hz]
            testmass_shape: test-mass noise spectral shape, either 'original' or 'lowfreq-relax'
            testmass_frelax: dictionary of low-frequency relaxation frequencies for test-mass noise [Hz]
            oms_asds: tuple of dictionaries of amplitude spectral densities for OMS noise [m/sqrt(Hz)],
                ordered as (isi_carrier, isi_usb, tmi_carrier, tmi_usb, rfi_carrier, rfi_usb)
            moc_time_correlation_asds: dictionary of amplitude spectral densities for MOC time
                correlation noise [s/sqrt(Hz)], the default ASD seems rather high, this is due
                to the small sampling rate (default 1 / 86400s)
            oms_fknees: dictionary of cutoff frequencies for OMS noise
            ttl_coeffs: tuple (local_phi, distant_phi, local_eta, distant_eta) of dictionaries of
                tilt-to-length coefficients on each MOSA [m/rad], 'default' for a default set of
                coefficients, or 'random' to draw a set of coefficients from uniform distributions
                (LISA-UKOB-INST-ML-0001-i2 LISA TTL STOP Model, summary table, 2.4 mm/rad and
                2.2mm/rad for distant and local coefficients, respectively)
            sc_jitter_asds: tuple of dictionaries of angular jitter amplitude spectral densities
                for spacecraft, ordered as (yaw, pitch, roll) [rad/sqrt(Hz)]
            sc_jitter_fknees: tuple of dictionaries of cutoff frequencies for spacecraft angular jitter,
                ordered as (yaw, pitch, roll) [Hz]
            mosa_jitter_asds: tuple of dictionaries of angular jitter amplitude spectral densities
                for MOSA, ordered as (yaw, pitch) [rad/sqrt(Hz)]
            mosa_jitter_fknees: tuple of dictionaries of cutoff frequencies for MOSA angular jitter,
                ordered as (yaw, pitch) [Hz]
            mosa_angles: dictionary of oriented MOSA opening angles [deg], or 'default'
            dws_asds: dictionary of amplitude spectral densities for DWS measurement noise [rad/sqrt(Hz)]
            ranging_biases: dictionary of ranging noise bias [s]
            ranging_asds: dictionary of ranging noise amplitude spectral densities [s/sqrt(Hz)]
            prn_ambiguity: distance after which PRN code repeats itself [m] (reasonable value is 300 km),
                None or 0 for no ambiguities
            electro_delays: tuple (isi, tmi, rfi) of dictionaries for electronic delays [s]
            concurrent (bool): whether to use multiprocessing
        """
        # pylint: disable=too-many-arguments,too-many-statements,too-many-locals,too-many-branches
        logger.info("Initializing instrumental simulation")
        self.git_url = 'https://gitlab.in2p3.fr/lisa-simulation/instrument'
        self.version = importlib_metadata.version('lisainstrument')
        self.simulated = False
        self.concurrent = bool(concurrent)

        # Check orbit dataset
        if orbit_dataset not in ['tcb/ltt', 'tps/ppr']:
            raise ValueError(f"invalid orbit dataset '{orbit_dataset}'")

        # Measurement sampling
        if t0 == 'orbits':
            if isinstance(orbits, str) and orbits != 'static':
                logger.debug("Reading initial time from orbit file '%s'", orbits)
                with File(orbits, 'r') as orbitf:
                    version = Version(orbitf.attrs['version'])
                    logger.debug("Using orbit file version %s", version)
                    # Switch between various orbit file standards
                    if version in SpecifierSet('== 1.*', True):
                        self.t0 = float(orbitf.attrs['t0' if orbit_dataset == 'tcb/ltt' else 'tau0'])
                    elif version in SpecifierSet('== 2.*', True):
                        self.t0 = float(orbitf.attrs['t0'])
                    else:
                        raise ValueError(f"unsupported orbit file version '{version}'")
            else:
                self.t0 = 0.0
        else:
            self.t0 = float(t0)
        self.size = int(size)
        self.dt = float(dt)
        self.fs = 1 / self.dt
        self.duration = self.dt * self.size
        logger.info("Computing measurement time vector (size=%s, dt=%s)", self.size, self.dt)
        self.t = self.t0 + np.arange(self.size, dtype=np.float64) * self.dt

        # Physics sampling
        self.physics_upsampling = int(physics_upsampling)
        self.physics_size = self.size * self.physics_upsampling
        self.physics_dt = self.dt / self.physics_upsampling
        self.physics_fs = self.fs * self.physics_upsampling
        logger.info("Computing physics time vector (size=%s, dt=%s)", self.physics_size, self.physics_dt)
        self.physics_et = np.arange(self.physics_size, dtype=np.float64) * self.physics_dt # elapsed time
        self.physics_t = self.t0 + self.physics_et

        # Telemetry sampling
        self.telemetry_downsampling = int(telemetry_downsampling)
        self.telemetry_dt = self.dt * self.telemetry_downsampling
        self.telemetry_fs = self.fs / self.telemetry_downsampling
        # Extra telemetry samples before t0
        self.initial_telemetry_size = int(initial_telemetry_size)
        self.telemetry_t0 = self.t0 - self.initial_telemetry_size * self.telemetry_dt
        # Total telemetry size, includes initial telemetry samples
        # plus telemetry samples covering the entire measurement time vector,
        # hence the use of ``math.ceil`` -- the +1 is for the sample at t0
        self.telemetry_size = self.initial_telemetry_size + 1 \
            + math.ceil(self.size / self.telemetry_downsampling)
        logger.info("Computing telemetry time vector (size=%s, dt=%s)", self.telemetry_size, self.telemetry_dt)
        self.telemetry_t = self.telemetry_t0 \
            + np.arange(self.telemetry_size, dtype=np.float64) * self.telemetry_dt
        # Physics time vector covering telemetry samples
        self.telemetry_to_physics_dt = self.telemetry_downsampling * self.physics_upsampling
        self.physics_size_covering_telemetry = self.telemetry_size * self.telemetry_to_physics_dt
        self.physics_et_covering_telemetry = self.telemetry_t0 - self.t0 + \
            np.arange(self.physics_size_covering_telemetry, dtype=np.float64) * self.physics_dt
        self.physics_t_covering_telemetry = self.t0 + self.physics_et_covering_telemetry
        # How to cut physics data covering telemetry to regular size
        first_sample = self.initial_telemetry_size * self.telemetry_to_physics_dt
        self.telemetry_physics_slice = slice(first_sample, first_sample + self.physics_size)

        # Orbits, gravitational waves, glitches
        self.init_orbits(orbits, orbit_dataset)
        self.init_gws(gws)
        self.init_glitches(glitches)

        # Instrument topology
        self.central_freq = float(central_freq)
        self.init_lock(lock)
        self.init_fplan(fplan)
        if offset_freqs == 'default':
            # Default set yields valid beatnote frequencies for all lasers ('six')
            # with default 'static' set of MPRs
            self.offset_freqs = ForEachMOSA({
                '12': 0.0, '23': 11E6, '31': 7.5E6,
                '13': 16E6, '32': -12E6, '21': -7E6,
            })
        else:
            self.offset_freqs = ForEachMOSA(offset_freqs)

        # Laser and modulation noise
        self.laser_asds = ForEachMOSA(laser_asds)
        if laser_shape not in ['white', 'white+infrared']:
            raise ValueError(f"invalid laser noise spectral shape '{laser_shape}'")
        self.laser_shape = laser_shape

        if modulation_asds == 'default':
            # Default based on the performance model, with 10x amplification for right-sided
            # MOSAs, to account for errors in the frequency distribution system
            self.modulation_asds = ForEachMOSA({
                '12': 5.2E-14, '23': 5.2E-14, '31': 5.2E-14,
                '13': 5.2E-13, '32': 5.2E-13, '21': 5.2E-13,
            })
        elif modulation_asds is None:
            self.modulation_asds = ForEachMOSA(0)
        else:
            self.modulation_asds = ForEachMOSA(modulation_asds)

        if modulation_freqs == 'default':
            # Default based on mission baseline 2.4 MHz/2.401 MHz for left and right MOSAs
            self.modulation_freqs = ForEachMOSA({
                '12': 2.4E9, '23': 2.4E9, '31': 2.4E9,
                '13': 2.401E9, '32': 2.401E9, '21': 2.401E9,
            })
        else:
            self.modulation_freqs = ForEachMOSA(modulation_freqs)

        if tdir_modulations is None:
            self.tdir_modulations = {mosa: lambda times: 0 for mosa in ForEachMOSA.indices()}
        else:
            self.tdir_modulations = tdir_modulations

        # Clocks
        self.clock_asds = ForEachSC(clock_asds)
        self.clock_offsets = ForEachSC(clock_offsets)
        if clock_freqoffsets == 'default':
            # Default based on LISANode
            self.clock_freqoffsets = ForEachSC({'1': 5E-8, '2': 6.25E-7, '3': -3.75E-7})
        else:
            self.clock_freqoffsets = ForEachSC(clock_freqoffsets)
        if clock_freqlindrifts == 'default':
            # Default based on LISANode
            self.clock_freqlindrifts = ForEachSC({'1': 1.6E-15, '2': 2E-14, '3': -1.2E-14})
        else:
            self.clock_freqlindrifts = ForEachSC(clock_freqlindrifts)
        if clock_freqquaddrifts == 'default':
            # Default based on LISANode
            self.clock_freqquaddrifts = ForEachSC({'1': 9E-24, '2': 6.75E-23, '3': -1.125e-22})
        else:
            self.clock_freqquaddrifts = ForEachSC(clock_freqquaddrifts)

        # MOC time correlation
        self.moc_time_correlation_asds = ForEachSC(moc_time_correlation_asds)

        # Clock-noise inversion
        self.clockinv_tolerance = float(clockinv_tolerance)
        self.clockinv_maxiter = int(clockinv_maxiter)

        # Ranging noise
        self.ranging_biases = ForEachMOSA(ranging_biases)
        self.ranging_asds = ForEachMOSA(ranging_asds)
        if prn_ambiguity in [None, 0]:
            self.prn_ambiguity = None
        else:
            self.prn_ambiguity = float(prn_ambiguity)

        # Backlink, OMS and test-mass acceleration noise
        self.backlink_asds = ForEachMOSA(backlink_asds)
        self.backlink_fknees = ForEachMOSA(backlink_fknees)
        self.oms_isi_carrier_asds = ForEachMOSA(oms_asds[0])
        self.oms_isi_usb_asds = ForEachMOSA(oms_asds[1])
        self.oms_tmi_carrier_asds = ForEachMOSA(oms_asds[2])
        self.oms_tmi_usb_asds = ForEachMOSA(oms_asds[3])
        self.oms_rfi_carrier_asds = ForEachMOSA(oms_asds[4])
        self.oms_rfi_usb_asds = ForEachMOSA(oms_asds[5])
        self.oms_fknees = ForEachMOSA(oms_fknees)

        # Test-mass noise
        if testmass_shape not in ['original', 'lowfreq-relax']:
            raise ValueError(f"invalid test-mass noise spectral shape '{testmass_shape}'")
        self.testmass_shape = testmass_shape
        self.testmass_asds = ForEachMOSA(testmass_asds)
        self.testmass_fknees = ForEachMOSA(testmass_fknees)
        self.testmass_fbreak = ForEachMOSA(testmass_fbreak)
        self.testmass_frelax = ForEachMOSA(testmass_frelax)

        # Tilt-to-length
        if ttl_coeffs == 'default':
            # Default values drawn from distributions set in 'random'
            self.ttl_coeffs_local_phis = ForEachMOSA({
                '12': 2.005835e-03, '23': 2.105403e-04, '31': -1.815399e-03,
                '13': -2.865050e-04, '32': -1.986657e-03, '21': 9.368319e-04,
            })
            self.ttl_coeffs_distant_phis = ForEachMOSA({
                '12': 1.623910e-03, '23': 1.522873e-04, '31': -1.842871e-03,
                '13': -2.091585e-03, '32': 1.300866e-03, '21': -8.445374e-04,
            })
            self.ttl_coeffs_local_etas = ForEachMOSA({
                '12': -1.670389e-03, '23': 1.460681e-03, '31': -1.039064e-03,
                '13': 1.640473e-04, '32': 1.205353e-03, '21': -9.205764e-04,
            })
            self.ttl_coeffs_distant_etas = ForEachMOSA({
                '12': -1.076470e-03, '23': 5.228848e-04, '31': -5.662766e-05,
                '13': 1.960050e-03, '32': 9.021890e-04, '21': 1.908239e-03,
            })
        elif ttl_coeffs == 'random':
            self.ttl_coeffs_local_phis = ForEachMOSA(lambda _: np.random.uniform(-2.2E-3, 2.2E-3))
            self.ttl_coeffs_distant_phis = ForEachMOSA(lambda _: np.random.uniform(-2.4E-3, 2.4E-3))
            self.ttl_coeffs_local_etas = ForEachMOSA(lambda _: np.random.uniform(-2.2E-3, 2.2E-3))
            self.ttl_coeffs_distant_etas = ForEachMOSA(lambda _: np.random.uniform(-2.4E-3, 2.4E-3))
        else:
            self.ttl_coeffs_local_phis = ForEachMOSA(ttl_coeffs[0])
            self.ttl_coeffs_distant_phis = ForEachMOSA(ttl_coeffs[1])
            self.ttl_coeffs_local_etas = ForEachMOSA(ttl_coeffs[2])
            self.ttl_coeffs_distant_etas = ForEachMOSA(ttl_coeffs[3])
        self.sc_jitter_phi_asds = ForEachSC(sc_jitter_asds[0])
        self.sc_jitter_eta_asds = ForEachSC(sc_jitter_asds[1])
        self.sc_jitter_theta_asds = ForEachSC(sc_jitter_asds[2])
        self.sc_jitter_phi_fknees = ForEachSC(sc_jitter_fknees[0])
        self.sc_jitter_eta_fknees = ForEachSC(sc_jitter_fknees[1])
        self.sc_jitter_theta_fknees = ForEachSC(sc_jitter_fknees[2])
        self.mosa_jitter_phi_asds = ForEachMOSA(mosa_jitter_asds[0])
        self.mosa_jitter_eta_asds = ForEachMOSA(mosa_jitter_asds[1])
        self.mosa_jitter_phi_fknees = ForEachMOSA(mosa_jitter_fknees[0])
        self.mosa_jitter_eta_fknees = ForEachMOSA(mosa_jitter_fknees[1])
        self.dws_asds = ForEachMOSA(dws_asds)

        # MOSA opening angles
        if mosa_angles == 'default':
            # Default MOSA at +/- 30 deg
            self.mosa_angles = ForEachMOSA({
                '12': 30, '23': 30, '31': 30,
                '13': -30, '32': -30, '21': -30,
            })
        else:
            self.mosa_angles = ForEachMOSA(mosa_angles)

        # Interpolation and antialiasing filter
        self.init_interpolation(interpolation)
        self.init_aafilter(aafilter)

        # Electronic delays
        self.electro_delays_isis = ForEachMOSA(electro_delays[0])
        self.electro_delays_tmis = ForEachMOSA(electro_delays[1])
        self.electro_delays_rfis = ForEachMOSA(electro_delays[2])

    def init_interpolation(self, interpolation):
        """Initialize or design the interpolation function.

        We support no interpolation, a custom interpolation function, or Lagrange interpolation.

        Args:
            parameters: see `interpolation` docstring in `__init__()`
        """
        if interpolation is None:
            logger.info("Disabling interpolation")
            self.interpolation_order = None
            self.interpolate = lambda x, _: x
        elif callable(interpolation):
            logger.info("Using user-provided interpolation function")
            self.interpolation_order = None
            self.interpolate = lambda x, shift: x if np.isscalar(x) else \
                interpolation(x, shift * self.physics_fs)
        else:
            method = str(interpolation[0])
            if method == 'lagrange':
                self.interpolation_order = int(interpolation[1])
                logger.debug("Using Lagrange interpolation of order %s", self.interpolation_order)
                self.interpolate = lambda x, shift: x if np.isscalar(x) else \
                    dsp.timeshift(x, shift * self.physics_fs, self.interpolation_order)
            else:
                raise ValueError(f"invalid interpolation parameters '{interpolation}'")

    def init_aafilter(self, aafilter):
        """Initialize antialiasing filter and downsampling."""
        self.downsampled = lambda _, x: x if np.isscalar(x) else x[::self.physics_upsampling]

        if aafilter is None:
            logger.info("Disabling antialiasing filter")
            self.aafilter_coeffs = None
            self.aafilter = lambda _, x: x
        elif isinstance(aafilter, (list, np.ndarray)):
            logger.info("Using user-provided antialiasing filter coefficients")
            self.aafilter_coeffs = aafilter
            self.aafilter = lambda _, x: x if np.isscalar(x) else \
                lfilter(self.aafilter_coeffs, 1, x)
        elif callable(aafilter):
            logger.info("Using user-provided antialiasing filter function")
            self.aafilter_coeffs = None
            self.aafilter = lambda _, x: x if np.isscalar(x) else aafilter(x)
        else:
            logger.info("Designing antialiasing filter %s", aafilter)
            self.aafilter_coeffs = self.design_aafilter(aafilter)
            self.aafilter = lambda _, x: x if np.isscalar(x) else \
                lfilter(self.aafilter_coeffs, 1, x)

    def design_aafilter(self, parameters):
        """Design the antialiasing filter.

        We currently support finite-impulse response filter designed from a Kaiser window.
        The order and beta parameters of the Kaiser window are deduced from the desired attenuation
        and transition bandwidth of the filter.

        Args:
            parameters: see `aafilter` docstring in `__init__()`

        Returns:
            A function that filters data.
        """
        method = parameters[0]
        nyquist = self.physics_fs / 2

        if method == 'kaiser':
            logger.debug("Designing finite-impulse response filter from Kaiser window")
            attenuation, freq1, freq2 = parameters[1], parameters[2], parameters[3]
            if attenuation == 0:
                logger.debug("Vanishing filter attenuation, disabling filtering")
                return lambda x: x
            logger.debug("Filter attenuation is %s dB", attenuation)
            logger.debug("Filter transition band is [%s Hz, %s Hz]", freq1, freq2)
            numtaps, beta = kaiserord(attenuation, (freq2 - freq1) / nyquist)
            logger.debug("Kaiser window has %s taps and beta is %s", numtaps, beta)
            taps = firwin(numtaps, (freq1 + freq2) / (2 * nyquist), window=('kaiser', beta))
            logger.debug("Filter taps are %s", taps)
            return taps

        raise ValueError(f"invalid filter parameters '{parameters}'")

    def init_lock(self, lock):
        """Initialize laser locking configuration."""
        if lock == 'six':
            logger.info("Using pre-defined locking configuration 'six'")
            self.lock_config = None # not a standard lock config
            self.lock = {'12': 'cavity', '23': 'cavity', '31': 'cavity', '13': 'cavity', '32': 'cavity', '21': 'cavity'}
        elif isinstance(lock, str):
            logger.info("Using pre-defined locking configuration '%s'", lock)
            self.lock_config = lock
            match = re.match(r'^(N[1-6])-(12|23|31|13|32|21)$', lock)
            if match:
                topology, primary = match.group(1), match.group(2)
                lock_12 = self.LOCK_TOPOLOGIES[topology] # with 12 as primary
                cycle = self.INDEX_CYCLES[primary] # correspondance to lock_12
                self.lock = {mosa: lock_12[cycle[mosa]] for mosa in self.MOSAS}
            else:
                raise ValueError(f"unsupported pre-defined locking configuration '{lock}'")
        elif isinstance(lock, dict):
            logger.info("Using explicit locking configuration '%s'", lock)
            if (set(lock.keys()) != set(self.MOSAS) or
                set(lock.values()) != set(['cavity', 'distant', 'adjacent'])):
                raise ValueError(f"invalid locking configuration '{lock}'")
            self.lock_config = None
            self.lock = lock
        else:
            raise ValueError(f"invalid locking configuration '{lock}'")

    def init_fplan(self, fplan):
        """Initialize frequency plan.

        Args:
            fplan: `fplan` parameter, c.f. `__init__()`
        """
        # pylint: disable=too-many-branches
        if fplan == 'static':
            logger.warning(
                "Using default set of locking beatnote frequencies; this "
                "might cause interferometric beatnote frequencies to fall "
                "outside the requirement range of 5..25 MHz"
            )
            self.fplan_file = None
            self.fplan = ForEachMOSA({
                '12': 8E6, '23': 9E6, '31': 10E6,
                '13': -8.2E6, '32': -8.5E6, '21': -8.7E6,
            })
        elif isinstance(fplan, str):
            logger.info("Using frequency-plan file '%s'", fplan)
            self.fplan_file = fplan
            # Refuse to use a fplan file if no orbit files are used
            if self.orbit_file is None:
                raise ValueError("cannot use frequency-plan for non orbit files")
            # Without a standard lock config, there is no dataset
            # in the frequency-plan file and therefore we cannot use it
            if self.lock_config is None:
                raise ValueError("cannot use frequency-plan for non standard lock configuration")
            with File(self.fplan_file, 'r') as fplanf:
                version = Version(fplanf.attrs['version'])
                logger.debug("Using frequency-plan file version %s", version)
                # Warn for frequency-plan file development version
                if version.is_devrelease:
                    logger.warning("You are using an frequency-plan file in a development version")
                # Switch between various fplan file standards
                if version in SpecifierSet('== 1.1.*', True):
                    logger.debug("Interpolating locking beatnote frequencies with piecewise linear functions")
                    times = self.orbit_t0 + np.arange(fplanf.attrs['size']) * fplanf.attrs['dt']
                    interpolate = lambda x: InterpolatedUnivariateSpline(times, x, k=1, ext='raise')(self.physics_t)
                    lock_beatnotes = {}
                    # Go through all MOSAs and pick locking beatnotes
                    try:
                        for mosa in self.MOSAS:
                            if self.lock[mosa] == 'cavity':
                                # No offset for primary laser
                                lock_beatnotes[mosa] = 0.0
                            elif self.lock[mosa] == 'distant':
                                lock_beatnotes[mosa] = 1E6 * interpolate(fplanf[self.lock_config][f'isi_{mosa}'])
                            elif self.lock[mosa] == 'adjacent':
                                # Fplan files only contain the one (left) RFI beatnote
                                left_mosa = ForEachSC.left_mosa(ForEachMOSA.sc(mosa))
                                sign = +1 if left_mosa == mosa else -1
                                lock_beatnotes[mosa] = 1E6 * sign * interpolate(fplanf[self.lock_config][f'rfi_{left_mosa}'])
                    except ValueError as error:
                        logger.error("Missing frequency-plan information at \n%s")
                        raise ValueError("missing frequency-plan information, use longer file or adjust sampling") from error
                    self.fplan = ForEachMOSA(lock_beatnotes)
                else:
                    raise ValueError(f"unsupported frequency-plan file version '{version}'")
        else:
            logger.info("Using user-provided locking beatnote frequencies")
            self.fplan_file = None
            self.fplan = ForEachMOSA(fplan)

    def init_orbits(self, orbits, orbit_dataset):
        """Initialize orbits.

        Args:
            orbits: `orbits` parameter, c.f. `__init__()`
            orbit_dataset: `orbit_dataset` parameter, c.f. `__init__()`
        """
        if orbits == 'static':
            logger.info("Using default set of static proper pseudo-ranges")
            self.orbit_file = None
            self.orbit_t0 = self.t0
            self.pprs = ForEachMOSA({
                # Default PPRs based on first samples of Keplerian orbits (v2.0.dev)
                '12': 8.33242295, '23': 8.30282196, '31': 8.33242298,
                '13': 8.33159404, '32': 8.30446786, '21': 8.33159402,
            })
            self.d_pprs = ForEachMOSA(0)
            self.tps_wrt_tcb = ForEachSC(0)
            self.orbit_dataset = None
        elif isinstance(orbits, str):
            logger.info("Using orbit file '%s'", orbits)
            self.orbit_file = orbits
            self.orbit_dataset = orbit_dataset
            with File(self.orbit_file, 'r') as orbitf:
                version = Version(orbitf.attrs['version'])
                logger.debug("Using orbit file version %s", version)
                # Warn for orbit file development version
                if version.is_devrelease:
                    logger.warning("You are using an orbit file in a development version")
                # Switch between various orbit file standards
                if version in SpecifierSet('== 1.*', True):
                    self.init_orbits_file_1_0(orbitf)
                elif version in SpecifierSet('== 2.*', True):
                    self.init_orbits_file_2_0(orbitf)
                else:
                    raise ValueError(f"unsupported orbit file version '{version}'")
        else:
            logger.info("Using user-provided proper pseudo-ranges and derivatives")
            self.orbit_file = None
            self.orbit_t0 = self.t0
            self.pprs = ForEachMOSA(orbits)
            self.d_pprs = self.pprs.transformed(lambda _, x:
                0 if np.isscalar(x) else np.gradient(x, self.physics_dt)
            )
            self.tps_wrt_tcb = ForEachSC(0)
            self.orbit_dataset = None

    def init_orbits_file_1_0(self, orbitf):
        """Initialize orbits from an orbit file version == 1.*."""

        def pprs_const(mosa):
            if self.orbit_dataset == 'tcb/ltt':
                return orbitf[f'tcb/l_{mosa}']['tt'][0]
            if self.orbit_dataset == 'tps/ppr':
                return orbitf[f'tps/l_{mosa}']['ppr'][0]
            raise ValueError(f"invalid orbit dataset '{self.orbit_dataset}'")

        def d_pprs(mosa):
            if self.orbit_dataset == 'tcb/ltt':
                times = orbitf['tcb']['t'][:]
                values = orbitf[f'tcb/l_{mosa}']['d_tt']
            elif self.orbit_dataset == 'tps/ppr':
                times = orbitf['tps']['tau'][:]
                values = orbitf[f'tps/l_{mosa}']['d_ppr']
            else:
                raise ValueError(f"invalid orbit dataset '{self.orbit_dataset}'")
            return InterpolatedUnivariateSpline(times, values, k=5, ext='raise')

        def tps_wrt_tcb(sc):
            if self.orbit_dataset == 'tcb/ltt':
                return lambda x: 0
            if self.orbit_dataset == 'tps/ppr':
                times = orbitf['tcb']['t'][:]
                values = orbitf[f'tcb/sc_{sc}']['tau']
                return InterpolatedUnivariateSpline(times, values, k=5, ext='raise')
            raise ValueError(f"invalid orbit dataset '{self.orbit_dataset}'")

        try:
            logger.debug("Reading orbit's t0")
            if self.orbit_dataset == 'tcb/ltt':
                self.orbit_t0 = orbitf['tcb']['t'][0]
            elif self.orbit_dataset == 'tps/ppr':
                self.orbit_t0 = orbitf['tps']['tau'][0]
            else:
                raise ValueError(f"invalid orbit dataset '{self.orbit_dataset}'")
            logger.debug("Interpolating proper pseudo-ranges")
            self.pprs = ForEachMOSA(lambda mosa: \
                pprs_const(mosa) + d_pprs(mosa).antiderivative()(self.physics_t) \
                - d_pprs(mosa).antiderivative()(self.orbit_t0)
            )
            logger.debug("Interpolating proper pseudo-range derivatives")
            self.d_pprs = ForEachMOSA(lambda mosa: d_pprs(mosa)(self.physics_t))
            logger.debug("Interpolating TPSs with respect to TCB")
            self.tps_wrt_tcb = ForEachSC(lambda sc: tps_wrt_tcb(sc)(self.physics_t_covering_telemetry))
        except ValueError as error:
            logger.error("Missing orbit information at \n%s", self.physics_t)
            raise ValueError("missing orbit information, use longer orbit file or adjust sampling") from error

    def init_orbits_file_2_0(self, orbitf):
        """Initialize orbits from an orbit file version == 2.*."""

        # Prepare common interpolation method
        times = orbitf.attrs['t0'] + np.arange(orbitf.attrs['size']) * orbitf.attrs['dt']
        interpolate = lambda data, t: InterpolatedUnivariateSpline(times, data, k=5, ext='raise')(t)
        int_interpolate = lambda data, t: InterpolatedUnivariateSpline(times, data, k=5, ext='raise').antiderivative()(t)
        link_index = {'12': 0, '23': 1, '31': 2, '13': 3, '32': 4, '21': 5}
        sc_index = {'1': 0, '2': 1, '3': 2}

        # Interpolate necessary orbital quantities,
        # show a helpful error message if orbit file is too short
        try:
            logger.debug("Reading orbit's t0")
            self.orbit_t0 = orbitf.attrs['t0']
            logger.debug("Interpolating proper pseudo-ranges")
            ppr_dataset = orbitf['tcb/ltt'] if self.orbit_dataset == 'tcb/ltt' else orbitf['tps/ppr']
            d_ppr_dataset = orbitf['tcb/d_ltt'] if self.orbit_dataset == 'tcb/ltt' else orbitf['tps/d_ppr']
            self.pprs = ForEachMOSA(lambda mosa: \
                ppr_dataset[0, link_index[mosa]] \
                + int_interpolate(d_ppr_dataset[:, link_index[mosa]], self.physics_t) \
                - int_interpolate(d_ppr_dataset[:, link_index[mosa]], self.orbit_t0)
            )
            logger.debug("Interpolating proper pseudo-range derivatives")
            self.d_pprs = ForEachMOSA(lambda mosa: \
                interpolate(d_ppr_dataset[:, link_index[mosa]], self.physics_t)
            )
        except ValueError as error:
            logger.error("Missing orbit information at \n%s", self.physics_t)
            raise ValueError("missing orbit information, use longer orbit file or adjust sampling") from error
        try:
            logger.debug("Interpolating TPSs with respect to TCB")
            if self.orbit_dataset == 'tcb/ltt':
                self.tps_wrt_tcb = ForEachSC(lambda sc: 0)
            else:
                dataset = orbitf['tcb/delta_tau']
                self.tps_wrt_tcb = ForEachSC(lambda sc:
                    interpolate(dataset[:, sc_index[sc]], self.physics_t_covering_telemetry))
        except ValueError as error:
            logger.error("Missing orbit information at \n%s", self.physics_t_covering_telemetry)
            raise ValueError("missing orbit information, use longer orbit file or adjust sampling") from error

    def init_gws(self, gws):
        """Initialize gravitational-wave responses."""
        if isinstance(gws, str):
            logger.info("Interpolating gravitational-wave responses from GW file '%s'", gws)
            self.gw_file = gws
            with File(self.gw_file, 'r') as gwf:
                version = Version(gwf.attrs['version'])
                logger.debug("Using GW file version %s", version)
                if version.is_devrelease:
                    logger.warning("You are using a GW file in a development version")
                if version in SpecifierSet('< 1.1', True):
                    self._init_gw_file_before_1_1(gwf)
                elif version in SpecifierSet('== 1.1', True):
                    self._init_gw_file_1_1(gwf)
                elif version in SpecifierSet('== 2.*', True):
                    self._init_gw_file_2(gwf)
                else:
                    raise ValueError(f"unsupported GW file version '{version}'")
        elif gws is None:
            logger.debug("No gravitational-wave responses")
            self.gw_file = None
            self.gw_group = None
            self.gws = ForEachMOSA(0)
        else:
            logger.info("Using user-provided gravitational-wave responses")
            self.gw_file = None
            self.gw_group = None
            self.gws = ForEachMOSA(gws)

    def _init_gw_file_before_1_1(self, gwf):
        """Initialize GW responses from GW file version < 1.1.

        Args:
            gwf (:obj:`h5py.File`): GW file object
        """
        self.gw_group = None
        interpolate = lambda data, t: InterpolatedUnivariateSpline(gwf['t'][:], data, k=5, ext='zeros')(t)
        self.gws = ForEachMOSA(lambda mosa: interpolate(gwf[f'l_{mosa}'][:], self.physics_t))

    def _init_gw_file_1_1(self, gwf):
        """Initialize GW responses from GW file version == 1.1.

        Args:
            gwf (:obj:`h5py.File`): GW file object
        """
        if self.orbit_dataset == 'tps/ppr' and 'tps' in gwf:
            logger.debug("Using link responses in TPS (following orbit dataset)")
            self.gw_group = 'tps'
        elif self.orbit_dataset == 'tps/ppr' and 'tcb' in gwf:
            logger.warning("TPS link responses not found on '%s', fall back to TCB responses", self.gw_file)
            logger.debug("Using link responses in TCB (inconsistent with orbit dataset)")
            self.gw_group = 'tcb'
        else:
            logger.debug("Using link responses in TCB (following orbit dataset)")
            self.gw_group = 'tcb'

        interpolate = lambda data, t: InterpolatedUnivariateSpline(gwf['t'][:], data, k=5, ext='zeros')(t)
        self.gws = ForEachMOSA(lambda mosa: interpolate(gwf[f'{self.gw_group}/l_{mosa}'][:], self.physics_t))

    def _init_gw_file_2(self, gwf):
        """Initialize GW responses from GW file version == 2.*.

        Args:
            gwf (:obj:`h5py.File`): GW file object
        """
        if self.orbit_dataset == 'tps/ppr' and 'tps' in gwf:
            logger.debug("Using link responses in TPS (following orbit dataset)")
            self.gw_group = 'tps'
        elif self.orbit_dataset == 'tps/ppr' and 'tcb' in gwf:
            logger.warning("TPS link responses not found on '%s', fall back to TCB responses", self.gw_file)
            logger.debug("Using link responses in TCB (inconsistent with orbit dataset)")
            self.gw_group = 'tcb'
        else:
            logger.debug("Using link responses in TCB (following orbit dataset)")
            self.gw_group = 'tcb'

        link_index = {'12': 0, '23': 1, '31': 2, '13': 3, '32': 4, '21': 5}
        times = gwf.attrs['t0'] + np.arange(gwf.attrs['size']) * gwf.attrs['dt']
        interpolate = lambda data, t: InterpolatedUnivariateSpline(times, data, k=5, ext='zeros')(t)
        self.gws = ForEachMOSA(lambda mosa: interpolate(gwf[f'{self.gw_group}/y'][:, link_index[mosa]], self.physics_t))

    def init_glitches(self, glitches):
        """Initialize glitches.

        According to https://gitlab.in2p3.fr/lisa-simulation/glitch, we have

            * test-mass glitches `tm_ij` [m/s]
            * laser glitches `laser_ij` [Hz]
            * readout glitches in the carrier ISIs `readout_isi_carrier_ij` [Hz]
            * readout glitches in the upper sideband ISIs `readout_isi_usb_ij` [Hz]
            * readout glitches in the carrier TMIs `readout_tmi_carrier_ij` [Hz]
            * readout glitches in the upper sideband TMIs `readout_tmi_usb_ij` [Hz]
            * readout glitches in the carrier RFIs `readout_rfi_carrier_ij` [Hz]
            * readout glitches in the upper sideband RFIs `readout_rfi_usb_ij` [Hz]

        """
        if isinstance(glitches, str):
            interpolate = lambda x, y, newx: InterpolatedUnivariateSpline(x, y, k=5, ext='const')(newx)

            self.glitch_file = glitches
            logger.info("Interpolating glitch signals from glitch file '%s'", self.glitch_file)
            with File(self.glitch_file, 'r') as glitchf:
                version = Version(glitchf.attrs['version'])
                logger.debug("Using glitch file version %s", version)
                if version.is_devrelease or version.local is not None:
                    logger.warning("You are using a glitch file in a development version")
                if version >= Version('1.3.dev'):
                    logger.warning(
                        "You are using a glitch file in a version that might not be fully supported")
                if version >= Version('1.3.dev'):
                    # Readout glitches
                    self.glitch_readout_isi_carriers = ForEachMOSA(lambda mosa:
                        0 if f'readout_isi_carrier_{mosa}' not in glitchf else \
                        interpolate(glitchf['t'][:], glitchf[f'readout_isi_carrier_{mosa}'][:], self.physics_t)
                    )
                    self.glitch_readout_isi_usbs = ForEachMOSA(lambda mosa:
                        0 if f'readout_isi_usb_{mosa}' not in glitchf else \
                        interpolate(glitchf['t'][:], glitchf[f'readout_isi_usb_{mosa}'][:], self.physics_t)
                    )
                    self.glitch_readout_tmi_carriers = ForEachMOSA(lambda mosa:
                        0 if f'readout_tmi_carrier_{mosa}' not in glitchf else \
                        interpolate(glitchf['t'][:], glitchf[f'readout_tmi_carrier_{mosa}'][:], self.physics_t)
                    )
                    self.glitch_readout_tmi_usbs = ForEachMOSA(lambda mosa:
                        0 if f'readout_tmi_usb_{mosa}' not in glitchf else \
                        interpolate(glitchf['t'][:], glitchf[f'readout_tmi_usb_{mosa}'][:], self.physics_t)
                    )
                    self.glitch_readout_rfi_carriers = ForEachMOSA(lambda mosa:
                        0 if f'readout_rfi_carrier_{mosa}' not in glitchf else \
                        interpolate(glitchf['t'][:], glitchf[f'readout_rfi_carrier_{mosa}'][:], self.physics_t)
                    )
                    self.glitch_readout_rfi_usbs = ForEachMOSA(lambda mosa:
                        0 if f'readout_rfi_usb_{mosa}' not in glitchf else \
                        interpolate(glitchf['t'][:], glitchf[f'readout_rfi_usb_{mosa}'][:], self.physics_t)
                    )
                    # Test-mass glitches
                    self.glitch_tms = ForEachMOSA(lambda mosa:
                        0 if f'tm_{mosa}' not in glitchf else \
                        interpolate(glitchf['t'][:], glitchf[f'tm_{mosa}'][:], self.physics_t)
                    )
                    # Laser glitches
                    self.glitch_lasers = ForEachMOSA(lambda mosa:
                        0 if f'laser_{mosa}' not in glitchf else \
                        interpolate(glitchf['t'][:], glitchf[f'laser_{mosa}'][:], self.physics_t)
                    )
                elif version >= Version('1.0'):
                    # Readout glitches
                    self.glitch_readout_isi_carriers = ForEachMOSA(0)
                    self.glitch_readout_isi_usbs = ForEachMOSA(0)
                    self.glitch_readout_tmi_carriers = ForEachMOSA(0)
                    self.glitch_readout_tmi_usbs = ForEachMOSA(0)
                    self.glitch_readout_rfi_carriers = ForEachMOSA(0)
                    self.glitch_readout_rfi_usbs = ForEachMOSA(0)
                    # Test-mass glitches
                    self.glitch_tms = ForEachMOSA(lambda mosa:
                        0 if f'tm_{mosa}' not in glitchf else \
                        interpolate(glitchf['t'][:], glitchf[f'tm_{mosa}'][:], self.physics_t)
                    )
                    # Laser glitches
                    self.glitch_lasers = ForEachMOSA(lambda mosa:
                        0 if f'laser_{mosa}' not in glitchf else \
                        interpolate(glitchf['t'][:], glitchf[f'laser_{mosa}'][:], self.physics_t)
                    )
        elif glitches is None:
            logger.debug("No glitches")
            self.glitch_file = None
            self.glitch_readout_isi_carriers = ForEachMOSA(0)
            self.glitch_readout_isi_usbs = ForEachMOSA(0)
            self.glitch_readout_tmi_carriers = ForEachMOSA(0)
            self.glitch_readout_tmi_usbs = ForEachMOSA(0)
            self.glitch_readout_rfi_carriers = ForEachMOSA(0)
            self.glitch_readout_rfi_usbs = ForEachMOSA(0)
            self.glitch_tms = ForEachMOSA(0)
            self.glitch_lasers = ForEachMOSA(0)
        else:
            raise ValueError(f"invalid value '{glitches}' for glitches")

    def disable_all_noises(self, but=None):
        """Turn off all instrumental noises.

        Args:
            but: optional category of noises to keep on ['laser', 'modulation',
                'clock', 'pathlength', 'ranging', 'jitters', 'dws',
                'moc-time-correlation']
        """
        valid_noises = [
            'laser', 'modulation', 'clock', 'pathlength',
            'ranging', 'jitters', 'dws', 'moc-time-correlation']
        if but is not None and but not in valid_noises:
            raise ValueError(f"unknown noise '{but}'")

        if but != 'laser':
            self.laser_asds = ForEachMOSA(0)
        if but != 'modulation':
            self.modulation_asds = ForEachMOSA(0)
        if but != 'clock':
            self.disable_clock_noises()
        if but != 'pathlength':
            self.disable_pathlength_noises()
        if but != 'ranging':
            self.disable_ranging_noises()
        if but != 'jitters':
            self.disable_jitters()
        if but != 'dws':
            self.dws_asds = ForEachMOSA(0)
        if but != 'moc-time-correlation':
            self.moc_time_correlation_asds = ForEachSC(0)

    def disable_clock_noises(self):
        """Turn off all imperfections on clocks."""
        self.clock_asds = ForEachSC(0)
        self.clock_offsets = ForEachSC(0)
        self.clock_freqoffsets = ForEachSC(0)
        self.clock_freqlindrifts = ForEachSC(0)
        self.clock_freqquaddrifts = ForEachSC(0)

    def disable_pathlength_noises(self):
        """Turn off all optical pathlength noises."""
        self.backlink_asds = ForEachMOSA(0)
        self.testmass_asds = ForEachMOSA(0)
        self.oms_isi_carrier_asds = ForEachMOSA(0)
        self.oms_isi_usb_asds = ForEachMOSA(0)
        self.oms_tmi_carrier_asds = ForEachMOSA(0)
        self.oms_tmi_usb_asds = ForEachMOSA(0)
        self.oms_rfi_carrier_asds = ForEachMOSA(0)
        self.oms_rfi_usb_asds = ForEachMOSA(0)

    def disable_ranging_noises(self):
        """Turn off all pseudo-ranging noises."""
        self.ranging_biases = ForEachMOSA(0)
        self.ranging_asds = ForEachMOSA(0)

    def disable_jitters(self):
        """Turn off all angular jitters."""
        self.sc_jitter_phi_asds = ForEachSC(0)
        self.sc_jitter_eta_asds = ForEachSC(0)
        self.sc_jitter_theta_asds = ForEachSC(0)
        self.mosa_jitter_phi_asds = ForEachMOSA(0)
        self.mosa_jitter_eta_asds = ForEachMOSA(0)

    def disable_dopplers(self):
        """Set proper pseudo-range derivatives to zero to turn off Doppler effects."""
        self.d_pprs = ForEachMOSA(0)

    def simulate(self, keep_all=False):
        """Run a simulation, and generate all intermediary signals.

        Args:
            keep_all: whether to keep all quantities in memory
        """
        # pylint: disable=too-many-locals,too-many-statements,too-many-branches

        logger.info("Starting simulation")
        self.keep_all = keep_all
        self.simulated = True

        self.simulate_noises()

        ## MOC time correlations

        logger.debug("Computing local SCET with respect to TPS")
        t = self.physics_et
        self.scet_wrt_tps_local = \
            self.clock_offsets \
            + self.clock_freqoffsets * t \
            + self.clock_freqlindrifts * t**2 / 2 \
            + self.clock_freqquaddrifts * t**3 / 3 \
            + self.integrated_clock_noise_fluctuations

        logger.debug("Computing SCET with respect to TCB")
        t = self.physics_et_covering_telemetry
        self.scet_wrt_tcb_withinitial = \
            self.tps_wrt_tcb \
            + self.clock_offsets \
            + self.clock_freqoffsets * (t + self.tps_wrt_tcb) \
            + self.clock_freqlindrifts * (t + self.tps_wrt_tcb)**2 / 2 \
            + self.clock_freqquaddrifts * (t + self.tps_wrt_tcb)**3  / 3 \
            + self.tps_wrt_tcb * self.clock_noise_fluctuations_covering_telemetry \
            + self.integrated_clock_noise_fluctuations_covering_telemetry

        logger.debug("Computing MOC time correlations")
        physics_to_telemetry = lambda _, x: x[::self.telemetry_to_physics_dt]
        self.moc_time_correlations = self.moc_time_correlation_noises \
            + self.scet_wrt_tcb_withinitial.transformed(physics_to_telemetry)

        ## TDIR modulations

        logger.debug("Forming TDIR assistance modulations")
        self.tdir_modulations_tseries = ForEachMOSA(lambda mosa:
            self.tdir_modulations[mosa](self.physics_et + self.scet_wrt_tps_local[mosa[0]])
        )

        ## Local beams

        logger.info("Simulating local beams")
        self.simulate_locking()

        ## Simulate sidebands

        logger.debug("Computing upper sideband offsets for primary local beam")
        self.local_usb_offsets = self.local_carrier_offsets \
            + self.modulation_freqs * (1 + self.clock_noise_offsets)

        logger.debug("Computing upper sideband fluctuations for primary local beam")
        self.local_usb_fluctuations = \
            self.local_carrier_fluctuations \
            + self.modulation_freqs * (self.clock_noise_fluctuations + self.modulation_noises)

        ## Propagation to distant MOSA

        logger.info("Propagating local beams to distant MOSAs")

        logger.debug("Propagating carrier offsets to distant MOSAs")
        delayed_distant_carrier_offsets = self.local_carrier_offsets.distant() \
            .transformed(lambda mosa, x: self.interpolate(x, -self.pprs[mosa]), concurrent=self.concurrent)
        self.distant_carrier_offsets = \
            -self.d_pprs * self.central_freq \
            + (1 - self.d_pprs) * delayed_distant_carrier_offsets

        logger.debug("Propagating carrier fluctuations to distant MOSAs")
        carrier_fluctuations = \
            self.local_carrier_fluctuations \
            - (self.central_freq + self.local_carrier_offsets) * self.distant_ttls / c
        propagated_carrier_fluctuations = \
            (1 - self.d_pprs) * carrier_fluctuations.distant() \
            .transformed(lambda mosa, x: self.interpolate(x, -self.pprs[mosa]), concurrent=self.concurrent)
        self.distant_carrier_fluctuations = \
            propagated_carrier_fluctuations \
            - (self.central_freq + delayed_distant_carrier_offsets) * self.gws \
            - (self.central_freq + delayed_distant_carrier_offsets) * self.local_ttls / c

        logger.debug("Propagating upper sideband offsets to distant MOSAs")
        delayed_distant_usb_offsets = self.local_usb_offsets.distant() \
            .transformed(lambda mosa, x: self.interpolate(x, -self.pprs[mosa]), concurrent=self.concurrent)
        self.distant_usb_offsets = \
            -self.d_pprs * self.central_freq \
            + (1 - self.d_pprs) * delayed_distant_usb_offsets

        logger.debug("Propagating upper sideband fluctuations to distant MOSAs")
        usb_fluctuations = \
            self.local_usb_fluctuations \
            - (self.central_freq + self.local_usb_offsets) * self.distant_ttls / c
        propagated_usb_fluctuations = \
            (1 - self.d_pprs) * usb_fluctuations.distant() \
            .transformed(lambda mosa, x: self.interpolate(x, -self.pprs[mosa]), concurrent=self.concurrent)
        self.distant_usb_fluctuations = \
            propagated_usb_fluctuations \
            - (self.central_freq + delayed_distant_usb_offsets) * self.gws \
            - (self.central_freq + delayed_distant_usb_offsets) * self.local_ttls / c

        logger.debug("Propagating local SCETs with respect to TPS to distant MOSAs")
        self.scet_wrt_tps_distant = \
            self.scet_wrt_tps_local.for_each_mosa().distant() \
            .transformed(lambda mosa, x: self.interpolate(x, -self.pprs[mosa])
            - self.pprs[mosa]
        )

        ## Propagation to adjacent MOSA

        logger.info("Propagating local beams to adjacent MOSAs")

        logger.debug("Propagating carrier offsets to adjacent MOSAs")
        self.adjacent_carrier_offsets = self.local_carrier_offsets.adjacent()

        logger.debug("Propagating carrier fluctuations to adjacent MOSAs")
        self.adjacent_carrier_fluctuations = \
            self.local_carrier_fluctuations.adjacent() \
            + self.central_freq * self.backlink_noises

        logger.debug("Propagating upper sideband offsets to adjacent MOSAs")
        self.adjacent_usb_offsets = self.local_usb_offsets.adjacent()

        logger.debug("Propagating upper sideband fluctuations to adjacent MOSAs")
        self.adjacent_usb_fluctuations = \
            self.local_usb_fluctuations.adjacent() \
            + self.central_freq * self.backlink_noises

        ## Inter-spacecraft interferometer local beams

        logger.info("Propagating local beams to inter-spacecraft interferometers")

        logger.debug("Propagating local carrier offsets to inter-spacecraft interferometer")
        self.local_isi_carrier_offsets = self.local_carrier_offsets

        logger.debug("Propagating local carrier fluctuations to inter-spacecraft interferometer")
        self.local_isi_carrier_fluctuations = self.local_carrier_fluctuations

        logger.debug("Propagating local upper sideband offsets to inter-spacecraft interferometer")
        self.local_isi_usb_offsets = self.local_usb_offsets

        logger.debug("Propagating local upper sideband fluctuations to inter-spacecraft interferometer")
        self.local_isi_usb_fluctuations = self.local_usb_fluctuations

        ## Inter-spacecraft interferometer distant beams

        logger.info("Propagating distant beams to inter-spacecraft interferometers")

        logger.debug("Propagating distant carrier offsets to inter-spacecraft interferometer")
        self.distant_isi_carrier_offsets = self.distant_carrier_offsets

        logger.debug("Propagating distant carrier fluctuations to inter-spacecraft interferometer")
        self.distant_isi_carrier_fluctuations = self.distant_carrier_fluctuations

        logger.debug("Propagating distant upper sideband offsets to inter-spacecraft interferometer")
        self.distant_isi_usb_offsets = self.distant_usb_offsets

        logger.debug("Propagating distant upper sideband fluctuations to inter-spacecraft interferometer")
        self.distant_isi_usb_fluctuations = self.distant_usb_fluctuations

        ## Inter-spacecraft interferometer beatnotes on TPS (high-frequency)

        logger.info("Computing inter-spacecraft beatnotes on TPS")

        logger.debug("Computing inter-spacecraft carrier beatnote offsets on TPS")
        self.tps_isi_carrier_offsets = \
            self.distant_isi_carrier_offsets - self.local_isi_carrier_offsets

        logger.debug("Computing inter-spacecraft carrier beatnote fluctuations on TPS")
        self.tps_isi_carrier_fluctuations = \
            self.distant_isi_carrier_fluctuations - self.local_isi_carrier_fluctuations \
            + self.central_freq * self.oms_isi_carrier_noises \
            + self.glitch_readout_isi_carriers

        logger.debug("Computing inter-spacecraft upper sideband beatnote offsets on TPS")
        self.tps_isi_usb_offsets = \
            self.distant_isi_usb_offsets - self.local_isi_usb_offsets

        logger.debug("Computing inter-spacecraft upper sideband beatnote fluctuations on TPS")
        self.tps_isi_usb_fluctuations = \
            self.distant_isi_usb_fluctuations - self.local_isi_usb_fluctuations \
            + self.central_freq * self.oms_isi_usb_noises \
            + self.glitch_readout_isi_usbs

        ## Inter-spacecraft DWS measurements on TPS (high-frequency)

        logger.info("Computing inter-spacecraft DWS measurements on TPS")
        self.tps_isi_dws_phis = self.mosa_total_jitter_phis + self.dws_phi_noises
        self.tps_isi_dws_etas = self.mosa_total_jitter_etas + self.dws_eta_noises

        ## Measured pseudo-ranging on TPS grid (high-frequency)

        logger.info("Computing measured pseudo-ranges on TPS")
        self.tps_mprs = self.scet_wrt_tps_local - self.scet_wrt_tps_distant \
            + self.ranging_noises


        ## Test-mass interferometer local beams

        logger.info("Propagating local beams to test-mass interferometers")

        logger.debug("Propagating local carrier offsets to test-mass interferometer")
        self.local_tmi_carrier_offsets = self.local_carrier_offsets

        logger.debug("Propagating local carrier fluctuations to test-mass interferometer")
        self.local_tmi_carrier_fluctuations = \
            self.local_carrier_fluctuations \
            + 2 * (self.testmass_noises + self.glitch_tms) / c \
            * (self.central_freq + self.local_tmi_carrier_offsets)

        logger.debug("Propagating local upper sideband offsets to test-mass interferometer")
        self.local_tmi_usb_offsets = self.local_usb_offsets

        logger.debug("Propagating local upper sideband fluctuations to test-mass interferometer")
        self.local_tmi_usb_fluctuations = \
            self.local_usb_fluctuations \
            + 2 * (self.testmass_noises + self.glitch_tms) / c \
            * (self.central_freq + self.local_tmi_usb_offsets)

        ## Test-mass interferometer adjacent beams

        logger.info("Propagating adjacent beams to test-mass interferometers")

        logger.debug("Propagating adjacent carrier offsets to test-mass interferometer")
        self.adjacent_tmi_carrier_offsets = self.adjacent_carrier_offsets

        logger.debug("Propagating adjacent carrier fluctuations to test-mass interferometer")
        self.adjacent_tmi_carrier_fluctuations = self.adjacent_carrier_fluctuations

        logger.debug("Propagating adjacent upper sideband offsets to test-mass interferometer")
        self.adjacent_tmi_usb_offsets = self.adjacent_usb_offsets

        logger.debug("Propagating adjacent upper sideband fluctuations to test-mass interferometer")
        self.adjacent_tmi_usb_fluctuations = self.adjacent_usb_fluctuations

        ## Test-mass interferometer beatnotes on TPS (high-frequency)

        logger.info("Computing test-mass beatnotes on TPS")

        logger.debug("Computing test-mass carrier beatnote offsets on TPS")
        self.tps_tmi_carrier_offsets = \
            self.adjacent_tmi_carrier_offsets - self.local_tmi_carrier_offsets

        logger.debug("Computing test-mass carrier beatnote fluctuations on TPS")
        self.tps_tmi_carrier_fluctuations = \
            self.adjacent_tmi_carrier_fluctuations - self.local_tmi_carrier_fluctuations \
            + self.central_freq * self.oms_tmi_carrier_noises \
            + self.glitch_readout_tmi_carriers

        logger.debug("Computing test-mass upper sideband beatnote offsets on TPS")
        self.tps_tmi_usb_offsets = \
            self.adjacent_tmi_usb_offsets - self.local_tmi_usb_offsets

        logger.debug("Computing test-mass upper sideband beatnote fluctuations on TPS")
        self.tps_tmi_usb_fluctuations = \
            self.adjacent_tmi_usb_fluctuations - self.local_tmi_usb_fluctuations \
            + self.central_freq * self.oms_tmi_usb_noises \
            + self.glitch_readout_tmi_usbs

        ## Reference interferometer local beams

        logger.info("Propagating local beams to reference interferometers")

        logger.debug("Propagating local carrier offsets to reference interferometer")
        self.local_rfi_carrier_offsets = self.local_carrier_offsets

        logger.debug("Propagating local carrier fluctuations to reference interferometer")
        self.local_rfi_carrier_fluctuations = self.local_carrier_fluctuations

        logger.debug("Propagating local upper sideband offsets to reference interferometer")
        self.local_rfi_usb_offsets = self.local_usb_offsets

        logger.debug("Propagating local upper sideband fluctuations to reference interferometer")
        self.local_rfi_usb_fluctuations = self.local_usb_fluctuations

        ## Reference interferometer adjacent beams

        logger.info("Propagating adjacent beams to reference interferometers")

        logger.debug("Propagating adjacent carrier offsets to reference interferometer")
        self.adjacent_rfi_carrier_offsets = self.adjacent_carrier_offsets

        logger.debug("Propagating adjacent carrier fluctuations to reference interferometer")
        self.adjacent_rfi_carrier_fluctuations = self.adjacent_carrier_fluctuations

        logger.debug("Propagating adjacent upper sideband offsets to reference interferometer")
        self.adjacent_rfi_usb_offsets = self.adjacent_usb_offsets

        logger.debug("Propagating adjacent upper sideband fluctuations to reference interferometer")
        self.adjacent_rfi_usb_fluctuations = self.adjacent_usb_fluctuations

        ## Reference interferometer beatnotes on TPS (high-frequency)

        logger.info("Computing reference beatnotes on TPS")

        logger.debug("Computing reference carrier beatnote offsets on TPS")
        self.tps_rfi_carrier_offsets = \
            self.adjacent_rfi_carrier_offsets - self.local_rfi_carrier_offsets

        logger.debug("Computing reference carrier beatnote fluctuations on TPS")
        self.tps_rfi_carrier_fluctuations = \
            self.adjacent_rfi_carrier_fluctuations - self.local_rfi_carrier_fluctuations \
            + self.central_freq * self.oms_rfi_carrier_noises \
            + self.glitch_readout_rfi_carriers

        logger.debug("Computing reference upper sideband beatnote offsets on TPS")
        self.tps_rfi_usb_offsets = \
            self.adjacent_rfi_usb_offsets - self.local_rfi_usb_offsets

        logger.debug("Computing reference upper sideband beatnote fluctuations on TPS")
        self.tps_rfi_usb_fluctuations = \
            self.adjacent_rfi_usb_fluctuations - self.local_rfi_usb_fluctuations \
            + self.central_freq * self.oms_rfi_usb_noises \
            + self.glitch_readout_rfi_usbs

        ## Sampling beatnotes, DWS measurements, and measured pseudo-ranges to SCET grid

        logger.info("Inverting SCET with respect to TPS")
        self.tps_wrt_scet = self.scet_wrt_tps_local \
            .transformed(lambda sc, x: self.invert_scet_wrt_tps(x, sc), concurrent=self.concurrent)

        self.timestamped = \
            lambda mosa, x: self.interpolate(x, -self.tps_wrt_scet.for_each_mosa()[mosa])

        logger.info("Sampling inter-spacecraft beatnotes to SCET grid")

        logger.debug("Sampling inter-spacecraft carrier beatnote fluctuations to SCET grid")
        self.scet_isi_carrier_offsets = (
            self.tps_isi_carrier_offsets / (1 + self.clock_noise_offsets)
        ).transformed(self.timestamped, concurrent=self.concurrent)

        logger.debug("Sampling inter-spacecraft carrier beatnote fluctuations to SCET grid")
        self.scet_isi_carrier_fluctuations = (
            self.tps_isi_carrier_fluctuations / (1 + self.clock_noise_offsets)
                - self.tps_isi_carrier_offsets * self.clock_noise_fluctuations
                / (1 + self.clock_noise_offsets)**2
        ).transformed(self.timestamped, concurrent=self.concurrent)

        logger.debug("Sampling inter-spacecraft upper sideband beatnote offsets to SCET grid")
        self.scet_isi_usb_offsets = (
            self.tps_isi_usb_offsets / (1 + self.clock_noise_offsets)
        ).transformed(self.timestamped, concurrent=self.concurrent)

        logger.debug("Sampling inter-spacecraft upper sideband beatnote fluctuations to SCET grid")
        self.scet_isi_usb_fluctuations = (
            self.tps_isi_usb_fluctuations / (1 + self.clock_noise_offsets)
                - self.tps_isi_usb_offsets * self.clock_noise_fluctuations
                / (1 + self.clock_noise_offsets)**2
        ).transformed(self.timestamped, concurrent=self.concurrent)

        logger.debug("Sampling inter-spacecraft DWS measurements to SCET grid")
        self.scet_isi_dws_phis = self.tps_isi_dws_phis.transformed(self.timestamped, concurrent=self.concurrent)
        self.scet_isi_dws_etas = self.tps_isi_dws_etas.transformed(self.timestamped, concurrent=self.concurrent)

        logger.info("Sampling measured pseudo-ranges to SCET grid")
        self.scet_mprs = self.tps_mprs.transformed(self.timestamped, concurrent=self.concurrent)

        logger.info("Sampling test-mass beatnotes to SCET grid")

        logger.debug("Sampling test-mass carrier beatnote offsets to SCET grid")
        self.scet_tmi_carrier_offsets = (
            self.tps_tmi_carrier_offsets / (1 + self.clock_noise_offsets)
        ).transformed(self.timestamped, concurrent=self.concurrent)

        logger.debug("Sampling test-mass carrier beatnote fluctuations to SCET grid")
        self.scet_tmi_carrier_fluctuations = (
            self.tps_tmi_carrier_fluctuations / (1 + self.clock_noise_offsets)
                - self.tps_tmi_carrier_offsets * self.clock_noise_fluctuations
                / (1 + self.clock_noise_offsets)**2
        ).transformed(self.timestamped, concurrent=self.concurrent)

        logger.debug("Sampling test-mass upper sideband beatnote offsets to SCET grid")
        self.scet_tmi_usb_offsets = (
            self.tps_tmi_usb_offsets / (1 + self.clock_noise_offsets)
        ).transformed(self.timestamped, concurrent=self.concurrent)

        logger.debug("Sampling test-mass upper sideband beatnote fluctuations to SCET grid")
        self.scet_tmi_usb_fluctuations = (
            self.tps_tmi_usb_fluctuations / (1 + self.clock_noise_offsets)
                - self.tps_tmi_usb_offsets * self.clock_noise_fluctuations
                / (1 + self.clock_noise_offsets)**2
        ).transformed(self.timestamped, concurrent=self.concurrent)

        logger.info("Sampling reference beatnotes to SCET grid")

        logger.debug("Sampling reference carrier beatnote offsets to SCET grid")
        self.scet_rfi_carrier_offsets = (
            self.tps_rfi_carrier_offsets / (1 + self.clock_noise_offsets)
        ).transformed(self.timestamped, concurrent=self.concurrent)

        logger.debug("Sampling reference carrier beatnote fluctuations to SCET grid")
        self.scet_rfi_carrier_fluctuations = (
            self.tps_rfi_carrier_fluctuations / (1 + self.clock_noise_offsets)
                - self.tps_rfi_carrier_offsets * self.clock_noise_fluctuations
                / (1 + self.clock_noise_offsets)**2
        ).transformed(self.timestamped, concurrent=self.concurrent)

        logger.debug("Sampling reference upper sideband beatnote offsets to SCET grid")
        self.scet_rfi_usb_offsets = (
            self.tps_rfi_usb_offsets / (1 + self.clock_noise_offsets)
        ).transformed(self.timestamped, concurrent=self.concurrent)

        logger.debug("Sampling reference upper sideband beatnote fluctuations to SCET grid")
        self.scet_rfi_usb_fluctuations = (
            self.tps_rfi_usb_fluctuations / (1 + self.clock_noise_offsets)
                - self.tps_rfi_usb_offsets * self.clock_noise_fluctuations
                / (1 + self.clock_noise_offsets)**2
        ).transformed(self.timestamped, concurrent=self.concurrent)

        # Electronic delays

        logger.info("Applying electronic delays")
        self.electro_isi = lambda mosa, x: self.interpolate(x, -self.electro_delays_isis[mosa])
        self.electro_tmi = lambda mosa, x: self.interpolate(x, -self.electro_delays_tmis[mosa])
        self.electro_rfi = lambda mosa, x: self.interpolate(x, -self.electro_delays_rfis[mosa])

        logger.debug("Applying electronic delays to inter-spacecraft beatnotes")
        self.electro_isi_carrier_offsets = self.scet_isi_carrier_offsets \
            .transformed(self.electro_isi, concurrent=self.concurrent)
        self.electro_isi_carrier_fluctuations = self.scet_isi_carrier_fluctuations \
            .transformed(self.electro_isi, concurrent=self.concurrent)
        self.electro_isi_usb_offsets = self.scet_isi_usb_offsets \
            .transformed(self.electro_isi, concurrent=self.concurrent)
        self.electro_isi_usb_fluctuations = self.scet_isi_usb_fluctuations \
            .transformed(self.electro_isi, concurrent=self.concurrent)

        logger.debug("Applying electronic delays to test-mass beatnotes")
        self.electro_tmi_carrier_offsets = self.scet_tmi_carrier_offsets \
            .transformed(self.electro_tmi, concurrent=self.concurrent)
        self.electro_tmi_carrier_fluctuations = self.scet_tmi_carrier_fluctuations \
            .transformed(self.electro_tmi, concurrent=self.concurrent)
        self.electro_tmi_usb_offsets = self.scet_tmi_usb_offsets \
            .transformed(self.electro_tmi, concurrent=self.concurrent)
        self.electro_tmi_usb_fluctuations = self.scet_tmi_usb_fluctuations \
            .transformed(self.electro_tmi, concurrent=self.concurrent)

        logger.debug("Applying electronic delays to reference beatnotes")
        self.electro_rfi_carrier_offsets = self.scet_rfi_carrier_offsets \
            .transformed(self.electro_rfi, concurrent=self.concurrent)
        self.electro_rfi_carrier_fluctuations = self.scet_rfi_carrier_fluctuations \
            .transformed(self.electro_rfi, concurrent=self.concurrent)
        self.electro_rfi_usb_offsets = self.scet_rfi_usb_offsets \
            .transformed(self.electro_rfi, concurrent=self.concurrent)
        self.electro_rfi_usb_fluctuations = self.scet_rfi_usb_fluctuations \
            .transformed(self.electro_rfi, concurrent=self.concurrent)

        ## Antialiasing filtering

        logger.info("Filtering beatnotes")

        logger.debug("Filtering inter-spacecraft beatnotes")
        self.filtered_isi_carrier_offsets = self.electro_isi_carrier_offsets \
            .transformed(self.aafilter, concurrent=self.concurrent)
        self.filtered_isi_carrier_fluctuations = self.electro_isi_carrier_fluctuations \
            .transformed(self.aafilter, concurrent=self.concurrent)
        self.filtered_isi_usb_offsets = self.electro_isi_usb_offsets \
            .transformed(self.aafilter, concurrent=self.concurrent)
        self.filtered_isi_usb_fluctuations = self.electro_isi_usb_fluctuations \
            .transformed(self.aafilter, concurrent=self.concurrent)

        logger.debug("Filtering inter-spacecraft DWS measurements")
        self.filtered_isi_dws_phis = self.scet_isi_dws_phis \
            .transformed(self.aafilter, concurrent=self.concurrent)
        self.filtered_isi_dws_etas = self.scet_isi_dws_etas \
            .transformed(self.aafilter, concurrent=self.concurrent)

        logger.debug("Filtering measured pseudo-ranges")
        self.filtered_mprs = self.scet_mprs \
            .transformed(self.aafilter, concurrent=self.concurrent)

        logger.debug("Filtering test-mass beatnotes")
        self.filtered_tmi_carrier_offsets = self.electro_tmi_carrier_offsets \
            .transformed(self.aafilter, concurrent=self.concurrent)
        self.filtered_tmi_carrier_fluctuations = self.electro_tmi_carrier_fluctuations \
            .transformed(self.aafilter, concurrent=self.concurrent)
        self.filtered_tmi_usb_offsets = self.electro_tmi_usb_offsets \
            .transformed(self.aafilter, concurrent=self.concurrent)
        self.filtered_tmi_usb_fluctuations = self.electro_tmi_usb_fluctuations \
            .transformed(self.aafilter, concurrent=self.concurrent)

        logger.debug("Filtering reference beatnotes")
        self.filtered_rfi_carrier_offsets = self.electro_rfi_carrier_offsets \
            .transformed(self.aafilter, concurrent=self.concurrent)
        self.filtered_rfi_carrier_fluctuations = self.electro_rfi_carrier_fluctuations \
            .transformed(self.aafilter, concurrent=self.concurrent)
        self.filtered_rfi_usb_offsets = self.electro_rfi_usb_offsets \
            .transformed(self.aafilter, concurrent=self.concurrent)
        self.filtered_rfi_usb_fluctuations = self.electro_rfi_usb_fluctuations \
            .transformed(self.aafilter, concurrent=self.concurrent)

        ## Downsampling filtering

        logger.info("Downsampling beatnotes")

        logger.debug("Downsampling inter-spacecraft beatnotes")
        self.isi_carrier_offsets = self.filtered_isi_carrier_offsets.transformed(self.downsampled)
        self.isi_carrier_fluctuations = self.filtered_isi_carrier_fluctuations.transformed(self.downsampled)
        self.isi_usb_offsets = self.filtered_isi_usb_offsets.transformed(self.downsampled)
        self.isi_usb_fluctuations = self.filtered_isi_usb_fluctuations.transformed(self.downsampled)

        logger.debug("Downsampling inter-spacecraft DWS measurements")
        self.isi_dws_phis = self.filtered_isi_dws_phis.transformed(self.downsampled)
        self.isi_dws_etas = self.filtered_isi_dws_etas.transformed(self.downsampled)

        logger.debug("Downsampling measured pseudo-ranges")
        self.mprs_unambiguous = self.filtered_mprs.transformed(self.downsampled)
        if self.prn_ambiguity is None:
            self.mprs = self.mprs_unambiguous
        else:
            self.mprs = self.mprs_unambiguous.transformed(lambda _, x: np.mod(x, self.prn_ambiguity / c))

        logger.debug("Downsampling test-mass beatnotes")
        self.tmi_carrier_offsets = self.filtered_tmi_carrier_offsets.transformed(self.downsampled)
        self.tmi_carrier_fluctuations = self.filtered_tmi_carrier_fluctuations.transformed(self.downsampled)
        self.tmi_usb_offsets = self.filtered_tmi_usb_offsets.transformed(self.downsampled)
        self.tmi_usb_fluctuations = self.filtered_tmi_usb_fluctuations.transformed(self.downsampled)

        logger.debug("Downsampling reference beatnotes")
        self.rfi_carrier_offsets = self.filtered_rfi_carrier_offsets.transformed(self.downsampled)
        self.rfi_carrier_fluctuations = self.filtered_rfi_carrier_fluctuations.transformed(self.downsampled)
        self.rfi_usb_offsets = self.filtered_rfi_usb_offsets.transformed(self.downsampled)
        self.rfi_usb_fluctuations = self.filtered_rfi_usb_fluctuations.transformed(self.downsampled)

        ## Total frequencies

        logger.info("Computing total beatnote frequencies")

        logger.debug("Computing total inter-spacecraft carrier beatnotes")
        self.isi_carriers = \
            self.isi_carrier_offsets + self.isi_carrier_fluctuations

        logger.debug("Computing total inter-spacecraft upper sideband beatnotes")
        self.isi_usbs = \
            self.isi_usb_offsets + self.isi_usb_fluctuations

        logger.debug("Computing total test-mass carrier beatnotes")
        self.tmi_carriers = \
            self.tmi_carrier_offsets + self.tmi_carrier_fluctuations

        logger.debug("Computing total test-mass upper sideband beatnotes")
        self.tmi_usbs = \
            self.tmi_usb_offsets + self.tmi_usb_fluctuations

        logger.debug("Computing total reference carrier beatnotes")
        self.rfi_carriers = \
            self.rfi_carrier_offsets + self.rfi_carrier_fluctuations

        logger.debug("Computing total reference upper sideband beatnotes")
        self.rfi_usbs = \
            self.rfi_usb_offsets + self.rfi_usb_fluctuations

        ## Closing simulation
        logger.info("Simulation complete")

    def simulate_noises(self): # pylint: disable=too-many-statements
        """Generate noise time series."""

        ## Laser noise
        # Laser noise are only generated for lasers locked on cavities,
        # in `simulate_locking.lock_on_cavity()`

        self.laser_noises = ForEachMOSA(0)

        ## Clock noise

        logger.info("Generating clock noise")

        if self.clock_freqlindrifts == self.clock_freqquaddrifts == 0:
            # Optimize to use a scalar if we only have a constant frequency offset
            logger.debug("Generating clock noise offsets as constant frequency offsets")
            self.clock_noise_offsets = self.clock_freqoffsets
        else:
            logger.debug("Generating clock noise offsets")
            t = self.physics_et
            self.clock_noise_offsets = \
                self.clock_freqoffsets \
                + self.clock_freqlindrifts * t \
                + self.clock_freqquaddrifts * t**2

        logger.debug("Generating clock noise fluctuations")

        # Include initial telemetry time period
        self.clock_noise_fluctuations_covering_telemetry = ForEachSC(lambda sc:
            noises.clock(
                self.physics_fs,
                self.physics_size_covering_telemetry,
                self.clock_asds[sc]),
            concurrent=self.concurrent
        )

        # Slice to only select physics period
        self.clock_noise_fluctuations = \
            self.clock_noise_fluctuations_covering_telemetry.transformed(
                lambda _, x: x if np.isscalar(x) else x[self.telemetry_physics_slice]
            )

        logger.debug("Integrating clock noise fluctuations")

        # Include initial telemetry time period
        self.integrated_clock_noise_fluctuations_covering_telemetry = \
            ForEachSC(lambda sc:
                cumulative_trapezoid(np.broadcast_to(
                    self.clock_noise_fluctuations_covering_telemetry[sc],
                    self.physics_size_covering_telemetry),
                dx=self.physics_dt, initial=0),
                concurrent=self.concurrent
            )

        # Slice to only select physics period
        self.integrated_clock_noise_fluctuations = \
            self.integrated_clock_noise_fluctuations_covering_telemetry.transformed(
                lambda _, x: x if np.isscalar(x) else x[self.telemetry_physics_slice]
            )

        ## Modulation noise

        logger.info("Generating modulation noise")
        self.modulation_noises = ForEachMOSA(lambda mosa:
            noises.modulation(self.physics_fs, self.physics_size, self.modulation_asds[mosa]),
            concurrent=self.concurrent
        )

        ## Backlink noise

        logger.info("Generating backlink noise")
        self.backlink_noises = ForEachMOSA(lambda mosa:
            noises.backlink(self.physics_fs, self.physics_size,
                self.backlink_asds[mosa], self.backlink_fknees[mosa]),
            concurrent=self.concurrent
        )

        ## Test-mass acceleration noise

        logger.info("Generating test-mass acceleration noise")
        self.testmass_noises = ForEachMOSA(lambda mosa:
            noises.testmass(
                self.physics_fs,
                self.physics_size,
                self.testmass_asds[mosa],
                self.testmass_fknees[mosa],
                self.testmass_fbreak[mosa],
                self.testmass_frelax[mosa],
                self.testmass_shape,
            ),
            concurrent=self.concurrent
        )

        ## Ranging noise

        logger.info("Generating ranging noise")
        self.ranging_noises = ForEachMOSA(lambda mosa:
            self.ranging_biases[mosa] + noises.ranging(self.physics_fs,
                self.physics_size, self.ranging_asds[mosa]),
            concurrent=self.concurrent
        )

        ## OMS noise

        logger.info("Generating OMS noise")

        self.oms_isi_carrier_noises = ForEachMOSA(lambda mosa:
            noises.oms(self.physics_fs, self.physics_size,
                self.oms_isi_carrier_asds[mosa], self.oms_fknees[mosa]),
            concurrent=self.concurrent
        )

        self.oms_isi_usb_noises = ForEachMOSA(lambda mosa:
            noises.oms(self.physics_fs, self.physics_size,
                self.oms_isi_usb_asds[mosa], self.oms_fknees[mosa]),
            concurrent=self.concurrent
        )

        self.oms_tmi_carrier_noises = ForEachMOSA(lambda mosa:
            noises.oms(self.physics_fs, self.physics_size,
                self.oms_tmi_carrier_asds[mosa], self.oms_fknees[mosa]),
            concurrent=self.concurrent
        )

        self.oms_tmi_usb_noises = ForEachMOSA(lambda mosa:
            noises.oms(self.physics_fs, self.physics_size,
                self.oms_tmi_usb_asds[mosa], self.oms_fknees[mosa]),
            concurrent=self.concurrent
        )

        self.oms_rfi_carrier_noises = ForEachMOSA(lambda mosa:
            noises.oms(self.physics_fs, self.physics_size,
                self.oms_rfi_carrier_asds[mosa], self.oms_fknees[mosa]),
            concurrent=self.concurrent
        )

        self.oms_rfi_usb_noises = ForEachMOSA(lambda mosa:
            noises.oms(self.physics_fs, self.physics_size,
                self.oms_rfi_usb_asds[mosa], self.oms_fknees[mosa]),
            concurrent=self.concurrent
        )

        ## DWS measurement noise

        logger.info("Generating DWS measurement noise")

        self.dws_phi_noises = ForEachMOSA(lambda mosa:
            noises.dws(self.physics_fs, self.physics_size, self.dws_asds[mosa]),
            concurrent=self.concurrent
        )
        self.dws_eta_noises = ForEachMOSA(lambda mosa:
            noises.dws(self.physics_fs, self.physics_size, self.dws_asds[mosa]),
            concurrent=self.concurrent
        )

        ## MOC time correlation noise

        self.moc_time_correlation_noises = ForEachSC(lambda sc:
            noises.moc_time_correlation(
                self.telemetry_fs,
                self.telemetry_size,
                self.moc_time_correlation_asds[sc]),
            concurrent=self.concurrent
        )

        ## Angular jitters

        logger.info("Generating spacecraft angular jitters")

        self.sc_jitter_phis = ForEachSC(lambda sc:
            noises.jitter(self.physics_fs, self.physics_size,
                self.sc_jitter_phi_asds[sc], self.sc_jitter_phi_fknees[sc]),
            concurrent=self.concurrent
        )
        self.sc_jitter_etas = ForEachSC(lambda sc:
            noises.jitter(self.physics_fs, self.physics_size,
                self.sc_jitter_eta_asds[sc], self.sc_jitter_eta_fknees[sc]),
            concurrent=self.concurrent
        )
        self.sc_jitter_thetas = ForEachSC(lambda sc:
            noises.jitter(self.physics_fs, self.physics_size,
                self.sc_jitter_theta_asds[sc], self.sc_jitter_theta_fknees[sc]),
            concurrent=self.concurrent
        )

        logger.info("Generating MOSA angular jitters")

        self.mosa_jitter_phis = ForEachMOSA(lambda mosa:
            noises.jitter(self.physics_fs, self.physics_size,
                self.mosa_jitter_phi_asds[mosa], self.mosa_jitter_phi_fknees[mosa]),
            concurrent=self.concurrent
        )
        self.mosa_jitter_etas = ForEachMOSA(lambda mosa:
            noises.jitter(self.physics_fs, self.physics_size,
                self.mosa_jitter_eta_asds[mosa], self.mosa_jitter_eta_fknees[mosa]),
            concurrent=self.concurrent
        )

        logger.info("Computing MOSA total angular jitters")

        self.mosa_total_jitter_phis = self.sc_jitter_phis.for_each_mosa() + self.mosa_jitter_phis
        cos_mosa_angles = (self.mosa_angles * np.pi / 180).transformed(lambda _, x: np.cos(x))
        sin_mosa_angles = (self.mosa_angles * np.pi / 180).transformed(lambda _, x: np.sin(x))
        self.mosa_total_jitter_etas = \
             cos_mosa_angles * self.sc_jitter_etas.for_each_mosa() \
             + sin_mosa_angles * self.sc_jitter_thetas.for_each_mosa() \
             + self.mosa_jitter_etas

        ## Tilt-to-length coupling
        ## TTL couplings are defined as velocities [m/s]

        logger.info("Computing tilt-to-length couplings")

        logger.debug("Computing local tilt-to-length couplings")
        self.local_ttls = \
         self.ttl_coeffs_local_phis * self.mosa_total_jitter_phis \
         + self.ttl_coeffs_local_etas * self.mosa_total_jitter_etas

        logger.debug("Computing unpropagated distant tilt-to-length couplings")
        self.distant_ttls = \
         self.ttl_coeffs_distant_phis * self.mosa_total_jitter_phis \
         + self.ttl_coeffs_distant_etas * self.mosa_total_jitter_etas

    def lock_on_cavity(self, mosa):
        """Compute carrier and upper sideband offsets and fluctuations for laser locked on cavity.

        We generate laser noises for lasers locked on cavities here.

        Args:
            mosa: laser index
        """
        logger.info("Generating laser noise for laser %s", mosa)
        self.laser_noises[mosa] = noises.laser(
            fs=self.physics_fs,
            size=self.physics_size,
            asd=self.laser_asds[mosa],
            shape=self.laser_shape)

        logger.debug("Computing carrier offsets for primary local beam %s", mosa)
        self.local_carrier_offsets[mosa] = self.offset_freqs[mosa]

        logger.debug("Computing carrier fluctuations for primary local beam %s", mosa)
        self.local_carrier_fluctuations[mosa] = \
            self.laser_noises[mosa] + self.glitch_lasers[mosa] + self.tdir_modulations_tseries[mosa]


    def lock_on_adjacent(self, mosa):
        """Compute carrier and upper sideband offsets and fluctuations for laser locked to adjacent beam.

        Args:
            mosa: laser index
        """
        sc = ForEachMOSA.sc
        adjacent = ForEachMOSA.adjacent_mosa

        logger.debug("Computing carrier offsets for local beam %s "
                     "locked on adjacent beam %s", mosa, adjacent(mosa))
        self.local_carrier_offsets[mosa] = \
            self.local_carrier_offsets[adjacent(mosa)] \
            - self.fplan[mosa] * (1 + self.clock_noise_offsets[sc(mosa)])

        logger.debug("Computing carrier fluctuations for local beam %s "
                     "locked on adjacent beam %s", mosa, adjacent(mosa))
        adjacent_carrier_fluctuations = self.local_carrier_fluctuations[adjacent(mosa)] \
            + self.central_freq * self.backlink_noises[mosa]
        self.local_carrier_fluctuations[mosa] = adjacent_carrier_fluctuations \
            - self.fplan[mosa] * self.clock_noise_fluctuations[sc(mosa)] \
            + self.central_freq * self.oms_rfi_carrier_noises[mosa] \
            + self.tdir_modulations_tseries[mosa]


    def lock_on_distant(self, mosa):
        """Compute carrier and upper sideband offsets and fluctuations for locked laser to distant beam.

        Args:
            mosa: laser index
        """
        sc = ForEachMOSA.sc
        distant = ForEachMOSA.distant_mosa

        logger.debug("Computing carrier offsets for local beam %s "
                     "locked on distant beam %s", mosa, distant(mosa))
        carrier_offsets = self.local_carrier_offsets[distant(mosa)]
        distant_carrier_offsets = \
            -self.d_pprs[mosa] * self.central_freq \
            + (1 - self.d_pprs[mosa]) * self.interpolate(carrier_offsets, -self.pprs[mosa])
        self.local_carrier_offsets[mosa] = distant_carrier_offsets \
            - self.fplan[mosa] * (1 + self.clock_noise_offsets[sc(mosa)])

        logger.debug("Computing carrier fluctuations for local beam %s "
                     "locked on distant beam %s", mosa, distant(mosa))
        carrier_fluctuations = \
            self.local_carrier_fluctuations[distant(mosa)] \
            - (self.central_freq + self.local_carrier_offsets[distant(mosa)]) \
                * self.distant_ttls[distant(mosa)] / c
        distant_carrier_fluctuations = \
            (1 - self.d_pprs[mosa]) * self.interpolate(carrier_fluctuations, -self.pprs[mosa]) \
            - (self.central_freq + self.local_carrier_offsets[mosa]) * self.gws[mosa] \
            - (self.central_freq + self.local_carrier_offsets[mosa]) * self.local_ttls[mosa] / c
        self.local_carrier_fluctuations[mosa] = distant_carrier_fluctuations \
            - self.fplan[mosa] * self.clock_noise_fluctuations[sc(mosa)] \
            + self.central_freq * self.oms_isi_carrier_noises[mosa] \
            + self.tdir_modulations_tseries[mosa]


    def simulate_locking(self):
        """Simulate local beams from the locking configuration."""
        # pylint: disable=too-many-statements
        adjacent = ForEachMOSA.adjacent_mosa
        distant = ForEachMOSA.distant_mosa

        self.local_carrier_offsets = ForEachMOSA(None)
        self.local_carrier_fluctuations = ForEachMOSA(None)
        self.local_usb_offsets = ForEachMOSA(None)
        self.local_usb_fluctuations = ForEachMOSA(None)

        # Transform the lock dictionary into a dependency dictionary
        dependencies = {}
        logger.debug("Computing laser locking dependencies")
        for mosa, lock_type in self.lock.items():
            if lock_type == 'cavity':
                dependencies[mosa] = None
            elif lock_type == 'adjacent':
                dependencies[mosa] = adjacent(mosa)
            elif lock_type == 'distant':
                dependencies[mosa] = distant(mosa)
            else:
                raise ValueError(f"invalid locking type '{self.lock[mosa]}' for laser '{mosa}'")
        logger.debug("Laser locking dependencies read: %s", dependencies)

        # Apply locking conditions in order
        logger.debug("Applying locking conditions")
        just_locked = [None]
        while dependencies:
            being_locked = []
            available_mosas = [mosa for mosa, dep in dependencies.items() if dep in just_locked]
            for mosa in available_mosas:
                if self.lock[mosa] == 'cavity':
                    logger.debug("Locking laser %s on cavity", mosa)
                    self.lock_on_cavity(mosa)
                elif self.lock[mosa] == 'adjacent':
                    logger.debug("Locking laser %s on adjacent laser %s", mosa, adjacent(mosa))
                    self.lock_on_adjacent(mosa)
                elif self.lock[mosa] == 'distant':
                    logger.debug("Locking laser %s on distant laser %s", mosa, distant(mosa))
                    self.lock_on_distant(mosa)
                else:
                    raise ValueError(f"invalid locking type '{self.lock[mosa]}' for laser '{mosa}'")
                being_locked.append(mosa)
                del dependencies[mosa]
            just_locked = being_locked
            if not just_locked:
                raise RuntimeError(f"cannot apply locking conditions to remaining lasers '{list(dependencies.keys())}'")

    def invert_scet_wrt_tps(self, scet_wrt_tps, sc):
        """Invert SCET with respect to TPS of a given spacecraft.

        We recursively solve the implicit equation dtau(tau) = dtau_hat(tau - dtau(tau)) until the
        convergence criteria (tolerance) is met, or we exceed the maximum number of iterations.

        Args:
            scet_wrt_tps: array of SCETs with respect to TPS
            sc: spacecraft index
        """
        logger.debug("Inverting SCET with respect to TPS for spacecraft %s", sc)
        logger.debug("Solving iteratively (tolerance=%s s, maxiter=%s)",
            self.clockinv_tolerance, self.clockinv_maxiter)

        # Drop samples at the edges to compute error
        edge = min(100, len(scet_wrt_tps) // 2 - 1)
        error = 0

        niter = 0
        next_inverse = scet_wrt_tps
        while not niter or error > self.clockinv_tolerance:
            if niter >= self.clockinv_maxiter:
                logger.warning("Maximum number of iterations '%s' reached for SC %s (error=%.2E)", niter, sc, error)
                break
            logger.debug("Starting iteration #%s", niter)
            inverse = next_inverse
            next_inverse = self.interpolate(scet_wrt_tps, -inverse)
            error = np.max(np.abs((inverse - next_inverse)[edge:-edge]))
            logger.debug("End of iteration %s, with an error of %.2E s", niter, error)
            niter += 1
        logger.debug("End of SCET with respect to TCB inversion after %s iterations with an error of %.2E s", niter, error)
        return inverse

    def _write_attr(self, hdf5, *names):
        """Write a single object attribute as metadata on ``hdf5``.

        This method is used in :meth:`lisainstrument.Instrument._write_metadata`
        to write Python self's attributes as HDF5 attributes.

        >>> instru = Instrument()
        >>> instru.parameter = 42
        >>> instru._write_attr('parameter')

        Args:
            hdf5 (:obj:`h5py.Group`): an HDF5 file, or a dataset
            names* (str): attribute names
        """
        for name in names:
            value = getattr(self, name)
            # Take string representation for non-native types
            if not isinstance(value, (int, float, np.ndarray)):
                value = str(value)
            hdf5.attrs[name] = value

    def _write_metadata(self, hdf5):
        """Write relevant object's attributes as metadata on ``hdf5``.

        This is for tracability and reproducibility. All parameters
        necessary to re-instantiate the instrument object and reproduce the
        exact same simulation should be written to file.

        Use the :meth:`lisainstrument.Instrument._write_attr` method.

        .. admonition:: Suclassing notes
            This class is intended to be overloaded by subclasses to write
            additional attributes.

        .. important::
            You MUST call super implementation in subclasses.

        Args:
            hdf5 (:obj:`h5py.Group`): an HDF5 file, or a dataset
        """
        self._write_attr(hdf5,
            'git_url', 'version', 'concurrent',
            'dt', 't0', 'size', 'fs', 'duration',
            'initial_telemetry_size', 'telemetry_downsampling', 'telemetry_fs',
            'telemetry_dt', 'telemetry_size', 'telemetry_t0',
            'physics_upsampling', 'physics_size', 'physics_dt', 'physics_fs',
            'aafilter_coeffs',
            'central_freq', 'lock_config', 'lock',
            'fplan_file', 'fplan',
            'laser_asds', 'modulation_asds', 'modulation_freqs',
            'tdir_modulations',
            'clock_asds','clock_offsets', 'clock_freqoffsets', 'clock_freqlindrifts',
            'clock_freqquaddrifts', 'clockinv_tolerance', 'clockinv_maxiter',
            'ranging_biases', 'ranging_asds', 'prn_ambiguity',
            'backlink_asds', 'backlink_fknees',
            'testmass_asds', 'testmass_fknees',
            'oms_isi_carrier_asds', 'oms_isi_usb_asds',
            'oms_tmi_carrier_asds', 'oms_tmi_usb_asds',
            'oms_rfi_carrier_asds', 'oms_rfi_usb_asds', 'oms_fknees',
            'ttl_coeffs_local_phis', 'ttl_coeffs_distant_phis',
            'ttl_coeffs_local_etas', 'ttl_coeffs_distant_etas',
            'sc_jitter_phi_asds', 'sc_jitter_eta_asds',
            'sc_jitter_theta_asds', 'mosa_jitter_phi_asds',
            'dws_asds', 'mosa_angles',
            'orbit_file', 'orbit_dataset', 'orbit_t0',
            'gw_file', 'gw_group',
            'glitch_file',
            'interpolation_order',
            'electro_delays_isis', 'electro_delays_tmis', 'electro_delays_rfis',
        )

    def write(self, output='measurements.h5', mode='w-', keep_all=False):
        """Run a simulation.

        Args:
            output: path to measurement file
            mode: measurement file opening mode
            keep_all: whether to write all quantities to file
        """
        # pylint: disable=too-many-statements
        with File(output, mode) as hdf5:

            if not self.simulated:
                self.simulate(keep_all)

            logger.info("Writing simulation to '%s'", output)
            logger.debug("Writing metadata and physics time dataset to '%s'", output)

            self._write_metadata(hdf5)

            if keep_all:

                logger.debug("Writing proper pseudo-ranges to '%s'", output)
                self.pprs.write(hdf5, 'pprs')
                self.d_pprs.write(hdf5, 'd_pprs')

                logger.debug("Writing TPS with respect to TCB to '%s'", output)
                self.tps_wrt_tcb.write(hdf5, 'tps_wrt_tcb')

                logger.debug("Writing gravitational-wave responses to '%s'", output)
                self.gws.write(hdf5, 'gws')

                logger.debug("Writing glitch signals to '%s'", output)
                self.glitch_tms.write(hdf5, 'glitch_tms')
                self.glitch_lasers.write(hdf5, 'glitch_lasers')

                logger.debug("Writing TDIR assistance modulations to %s", output)
                self.tdir_modulations_tseries.write(hdf5, 'tdir_modulations_tseries')

                logger.debug("Writing laser noise to '%s'", output)
                self.laser_noises.write(hdf5, 'laser_noises')

                logger.debug("Writing clock noise to '%s'", output)
                self.clock_noise_offsets.write(hdf5, 'clock_noise_offsets')
                self.clock_noise_fluctuations.write(hdf5, 'clock_noise_fluctuations')
                self.clock_noise_fluctuations_covering_telemetry.write(
                    hdf5, 'clock_noise_fluctuations_withinitial')
                self.integrated_clock_noise_fluctuations_covering_telemetry.write(
                    hdf5, 'integrated_clock_noise_fluctuations_withinitial')
                self.integrated_clock_noise_fluctuations.write(
                    hdf5, 'integrated_clock_noise_fluctuations')

                logger.debug("Writing modulation noise to '%s'", output)
                self.modulation_noises.write(hdf5, 'modulation_noises')

                logger.debug("Writing backlink noise to '%s'", output)
                self.backlink_noises.write(hdf5, 'backlink_noises')

                logger.debug("Writing test-mass acceleration noise to '%s'", output)
                self.testmass_noises.write(hdf5, 'testmass_noises')

                logger.debug("Writing ranging noise to '%s'", output)
                self.ranging_noises.write(hdf5, 'ranging_noises')

                logger.debug("Writing OMS noise to '%s'", output)
                self.oms_isi_carrier_noises.write(hdf5, 'oms_isi_carrier_noises')
                self.oms_isi_usb_noises.write(hdf5, 'oms_isi_usb_noises')
                self.oms_tmi_carrier_noises.write(hdf5, 'oms_tmi_carrier_noises')
                self.oms_tmi_usb_noises.write(hdf5, 'oms_tmi_usb_noises')
                self.oms_rfi_carrier_noises.write(hdf5, 'oms_rfi_carrier_noises')
                self.oms_rfi_usb_noises.write(hdf5, 'oms_rfi_usb_noises')

                logger.debug("Writing MOC time correlation noise to '%s'", output)
                self.moc_time_correlation_noises.write(hdf5, 'moc_time_correlation_noises')

                logger.debug("Writing spacecraft angular jitter to '%s'", output)
                self.sc_jitter_phis.write(hdf5, 'sc_jitter_phis')
                self.sc_jitter_etas.write(hdf5, 'sc_jitter_etas')
                self.sc_jitter_thetas.write(hdf5, 'sc_jitter_thetas')

                logger.debug("Writing MOSA angular jitter to '%s'", output)
                self.mosa_jitter_phis.write(hdf5, 'mosa_jitter_phis')

                logger.debug("Writing MOSA total angular jitter to '%s'", output)
                self.mosa_total_jitter_phis.write(hdf5, 'mosa_total_jitter_phis')
                self.mosa_total_jitter_etas.write(hdf5, 'mosa_total_jitter_etas')

                logger.debug("Writing local beams to '%s'", output)
                self.local_carrier_offsets.write(hdf5, 'local_carrier_offsets')
                self.local_carrier_fluctuations.write(hdf5, 'local_carrier_fluctuations')
                self.local_usb_offsets.write(hdf5, 'local_usb_offsets')
                self.local_usb_fluctuations.write(hdf5, 'local_usb_fluctuations')

                logger.debug("Writing local SCET with respect to TPS to '%s'", output)
                self.scet_wrt_tps_local.write(hdf5, 'scet_wrt_tps_local')

                logger.debug("Writing SCET with respect to TCB to '%s'", output)
                self.scet_wrt_tcb_withinitial.write(hdf5, 'scet_wrt_tcb_withinitial')

                logger.debug("Writing tilt-to-length couplings to '%s'", output)
                self.local_ttls.write(hdf5, 'local_ttls')
                self.distant_ttls.write(hdf5, 'distant_ttls')

                logger.debug("Writing propagated distant beams to '%s'", output)
                self.distant_carrier_offsets.write(hdf5, 'distant_carrier_offsets')
                self.distant_carrier_fluctuations.write(hdf5, 'distant_carrier_fluctuations')
                self.distant_usb_offsets.write(hdf5, 'distant_usb_offsets')
                self.distant_usb_fluctuations.write(hdf5, 'distant_usb_fluctuations')

                logger.debug("Writing propagated SCETs with respect to TCB to '%s'", output)
                self.scet_wrt_tps_distant.write(hdf5, 'scet_wrt_tcb_distant')

                logger.debug("Writing propagated adjacent beams to '%s'", output)
                self.adjacent_carrier_offsets.write(hdf5, 'adjacent_carrier_offsets')
                self.adjacent_carrier_fluctuations.write(hdf5, 'adjacent_carrier_fluctuations')
                self.adjacent_usb_offsets.write(hdf5, 'adjacent_usb_offsets')
                self.adjacent_usb_fluctuations.write(hdf5, 'adjacent_usb_fluctuations')

                logger.debug("Writing local beams at inter-spacecraft interferometer to '%s'", output)
                self.local_isi_carrier_offsets.write(hdf5, 'local_isi_carrier_offsets')
                self.local_isi_carrier_fluctuations.write(hdf5, 'local_isi_carrier_fluctuations')
                self.local_isi_usb_offsets.write(hdf5, 'local_isi_usb_offsets')
                self.local_isi_usb_fluctuations.write(hdf5, 'local_isi_usb_fluctuations')

                logger.debug("Writing distant beams at inter-spacecraft interferometer to '%s'", output)
                self.distant_isi_carrier_offsets.write(hdf5, 'distant_isi_carrier_offsets')
                self.distant_isi_carrier_fluctuations.write(hdf5, 'distant_isi_carrier_fluctuations')
                self.distant_isi_usb_offsets.write(hdf5, 'distant_isi_usb_offsets')
                self.distant_isi_usb_fluctuations.write(hdf5, 'distant_isi_usb_fluctuations')

                logger.debug("Writing inter-spacecraft beatnotes on TPS to '%s'", output)
                self.tps_isi_carrier_offsets.write(hdf5, 'tps_isi_carrier_offsets')
                self.tps_isi_carrier_fluctuations.write(hdf5, 'tps_isi_carrier_fluctuations')
                self.tps_isi_usb_offsets.write(hdf5, 'tps_isi_usb_offsets')
                self.tps_isi_usb_fluctuations.write(hdf5, 'tps_isi_usb_fluctuations')

                logger.debug("Writing inter-spacecraft DWS measurements on TPS to '%s'", output)
                self.tps_isi_dws_phis.write(hdf5, 'tps_isi_dws_phis')
                self.tps_isi_dws_etas.write(hdf5, 'tps_isi_dws_etas')

                logger.debug("Writing measured pseudo-ranges on TPS to '%s'", output)
                self.tps_mprs.write(hdf5, 'tps_mprs')

                logger.debug("Writing local beams at test-mass interferometer to '%s'", output)
                self.local_tmi_carrier_offsets.write(hdf5, 'local_tmi_carrier_offsets')
                self.local_tmi_carrier_fluctuations.write(hdf5, 'local_tmi_carrier_fluctuations')
                self.local_tmi_usb_offsets.write(hdf5, 'local_tmi_usb_offsets')
                self.local_tmi_usb_fluctuations.write(hdf5, 'local_tmi_usb_fluctuations')

                logger.debug("Writing adjacent beams at test-mass interferometer to '%s'", output)
                self.adjacent_tmi_carrier_offsets.write(hdf5, 'adjacent_tmi_carrier_offsets')
                self.adjacent_tmi_carrier_fluctuations.write(hdf5, 'adjacent_tmi_carrier_fluctuations')
                self.adjacent_tmi_usb_offsets.write(hdf5, 'adjacent_tmi_usb_offsets')
                self.adjacent_tmi_usb_fluctuations.write(hdf5, 'adjacent_tmi_usb_fluctuations')

                logger.debug("Writing test-mass beatnotes on TPS to '%s'", output)
                self.tps_tmi_carrier_offsets.write(hdf5, 'tps_tmi_carrier_offsets')
                self.tps_tmi_carrier_fluctuations.write(hdf5, 'tps_tmi_carrier_fluctuations')
                self.tps_tmi_usb_offsets.write(hdf5, 'tps_tmi_usb_offsets')
                self.tps_tmi_usb_fluctuations.write(hdf5, 'tps_tmi_usb_fluctuations')

                logger.debug("Writing local beams at reference interferometer to '%s'", output)
                self.local_rfi_carrier_offsets.write(hdf5, 'local_rfi_carrier_offsets')
                self.local_rfi_carrier_fluctuations.write(hdf5, 'local_rfi_carrier_fluctuations')
                self.local_rfi_usb_offsets.write(hdf5, 'local_rfi_usb_offsets')
                self.local_rfi_usb_fluctuations.write(hdf5, 'local_rfi_usb_fluctuations')

                logger.debug("Writing adjacent beams at reference interferometer to '%s'", output)
                self.adjacent_rfi_carrier_offsets.write(hdf5, 'adjacent_rfi_carrier_offsets')
                self.adjacent_rfi_carrier_fluctuations.write(hdf5, 'adjacent_rfi_carrier_fluctuations')
                self.adjacent_rfi_usb_offsets.write(hdf5, 'adjacent_rfi_usb_offsets')
                self.adjacent_rfi_usb_fluctuations.write(hdf5, 'adjacent_rfi_usb_fluctuations')

                logger.debug("Writing reference beatnotes on TPS to '%s'", output)
                self.tps_rfi_carrier_offsets.write(hdf5, 'tps_rfi_carrier_offsets')
                self.tps_rfi_carrier_fluctuations.write(hdf5, 'tps_rfi_carrier_fluctuations')
                self.tps_rfi_usb_offsets.write(hdf5, 'tps_rfi_usb_offsets')
                self.tps_rfi_usb_fluctuations.write(hdf5, 'tps_rfi_usb_fluctuations')

                logger.debug("Writing TPS with respect to SCET to '%s'", output)
                self.tps_wrt_scet.write(hdf5, 'tps_wrt_scet')

                logger.debug("Writing inter-spacecraft beatnotes sampled to SCET grid to '%s'", output)
                self.scet_isi_carrier_offsets.write(hdf5, 'scet_isi_carrier_offsets')
                self.scet_isi_carrier_fluctuations.write(hdf5, 'scet_isi_carrier_fluctuations')
                self.scet_isi_usb_offsets.write(hdf5, 'scet_isi_usb_offsets')
                self.scet_isi_usb_fluctuations.write(hdf5, 'scet_isi_usb_fluctuations')

                logger.debug("Writing inter-spacecraft DWS measurements sampled to SCET grid to '%s'", output)
                self.scet_isi_dws_phis.write(hdf5, 'scet_isi_dws_phis')
                self.scet_isi_dws_etas.write(hdf5, 'scet_isi_dws_etas')

                logger.debug("Writing measured pseudo-ranges sampled to SCET grid to '%s'", output)
                self.scet_mprs.write(hdf5, 'scet_mprs')

                logger.debug("Writing test-mass beatnotes sampled to SCET grid to '%s'", output)
                self.scet_tmi_carrier_offsets.write(hdf5, 'scet_tmi_carrier_offsets')
                self.scet_tmi_carrier_fluctuations.write(hdf5, 'scet_tmi_carrier_fluctuations')
                self.scet_tmi_usb_offsets.write(hdf5, 'scet_tmi_usb_offsets')
                self.scet_tmi_usb_fluctuations.write(hdf5, 'scet_tmi_usb_fluctuations')

                logger.debug("Writing reference beatnotes sampled to SCET grid to '%s'", output)
                self.scet_rfi_carrier_offsets.write(hdf5, 'scet_rfi_carrier_offsets')
                self.scet_rfi_carrier_fluctuations.write(hdf5, 'scet_rfi_carrier_fluctuations')
                self.scet_rfi_usb_offsets.write(hdf5, 'scet_rfi_usb_offsets')
                self.scet_rfi_usb_fluctuations.write(hdf5, 'scet_rfi_usb_fluctuations')

                logger.debug("Writing inter-spacecraft beatnotes with electronic delay to '%s'", output)
                self.electro_isi_carrier_offsets.write(hdf5, 'electro_isi_carrier_offsets')
                self.electro_isi_carrier_fluctuations.write(hdf5, 'electro_isi_carrier_fluctuations')
                self.electro_isi_usb_offsets.write(hdf5, 'electro_isi_usb_offsets')
                self.electro_isi_usb_fluctuations.write(hdf5, 'electro_isi_usb_fluctuations')

                logger.debug("Writing test-mass beatnotes with electronic delays to '%s'", output)
                self.electro_tmi_carrier_offsets.write(hdf5, 'electro_tmi_carrier_offsets')
                self.electro_tmi_carrier_fluctuations.write(hdf5, 'electro_tmi_carrier_fluctuations')
                self.electro_tmi_usb_offsets.write(hdf5, 'electro_tmi_usb_offsets')
                self.electro_tmi_usb_fluctuations.write(hdf5, 'electro_tmi_usb_fluctuations')

                logger.debug("Writing reference beatnotes with electronic delays to '%s'", output)
                self.electro_rfi_carrier_offsets.write(hdf5, 'electro_rfi_carrier_offsets')
                self.electro_rfi_carrier_fluctuations.write(hdf5, 'electro_rfi_carrier_fluctuations')
                self.electro_rfi_usb_offsets.write(hdf5, 'electro_rfi_usb_offsets')
                self.electro_rfi_usb_fluctuations.write(hdf5, 'electro_rfi_usb_fluctuations')

                logger.debug("Writing filtered inter-spacecraft beatnotes to '%s'", output)
                self.filtered_isi_carrier_offsets.write(hdf5, 'filtered_isi_carrier_offsets')
                self.filtered_isi_carrier_fluctuations.write(hdf5, 'filtered_isi_carrier_fluctuations')
                self.filtered_isi_usb_offsets.write(hdf5, 'filtered_isi_usb_offsets')
                self.filtered_isi_usb_fluctuations.write(hdf5, 'filtered_isi_usb_fluctuations')

                logger.debug("Writing filtered inter-spacecraft DWS measurements to '%s'", output)
                self.filtered_isi_dws_phis.write(hdf5, 'filtered_isi_dws_phis')
                self.filtered_isi_dws_etas.write(hdf5, 'filtered_isi_dws_etas')

                logger.debug("Writing filtered measured pseudo-ranges to '%s'", output)
                self.filtered_mprs.write(hdf5, 'filtered_mprs')
                self.mprs_unambiguous.write(hdf5, 'mprs_unambiguous')

                logger.debug("Writing filtered test-mass beatnotes to '%s'", output)
                self.filtered_tmi_carrier_offsets.write(hdf5, 'filtered_tmi_carrier_offsets')
                self.filtered_tmi_carrier_fluctuations.write(hdf5, 'filtered_tmi_carrier_fluctuations')
                self.filtered_tmi_usb_offsets.write(hdf5, 'filtered_tmi_usb_offsets')
                self.filtered_tmi_usb_fluctuations.write(hdf5, 'filtered_tmi_usb_fluctuations')

                logger.debug("Writing filtered reference beatnotes to '%s'", output)
                self.filtered_rfi_carrier_offsets.write(hdf5, 'filtered_rfi_carrier_offsets')
                self.filtered_rfi_carrier_fluctuations.write(hdf5, 'filtered_rfi_carrier_fluctuations')
                self.filtered_rfi_usb_offsets.write(hdf5, 'filtered_rfi_usb_offsets')
                self.filtered_rfi_usb_fluctuations.write(hdf5, 'filtered_rfi_usb_fluctuations')

            logger.debug("Writing downsampled inter-spacecraft beatnotes to '%s'", output)
            self.isi_carrier_offsets.write(hdf5, 'isi_carrier_offsets')
            self.isi_carrier_fluctuations.write(hdf5, 'isi_carrier_fluctuations')
            self.isi_usb_offsets.write(hdf5, 'isi_usb_offsets')
            self.isi_usb_fluctuations.write(hdf5, 'isi_usb_fluctuations')

            logger.debug("Writing downsampled inter-spacecraft DWS measurements to '%s'", output)
            self.isi_dws_phis.write(hdf5, 'isi_dws_phis')
            self.isi_dws_etas.write(hdf5, 'isi_dws_etas')

            logger.debug("Writing downsampled measured pseudo-ranges to '%s'", output)
            self.mprs.write(hdf5, 'mprs')

            logger.debug("Writing downsampled test-mass beatnotes to '%s'", output)
            self.tmi_carrier_offsets.write(hdf5, 'tmi_carrier_offsets')
            self.tmi_carrier_fluctuations.write(hdf5, 'tmi_carrier_fluctuations')
            self.tmi_usb_offsets.write(hdf5, 'tmi_usb_offsets')
            self.tmi_usb_fluctuations.write(hdf5, 'tmi_usb_fluctuations')

            logger.debug("Writing downsampled reference beatnotes to '%s'", output)
            self.rfi_carrier_offsets.write(hdf5, 'rfi_carrier_offsets')
            self.rfi_carrier_fluctuations.write(hdf5, 'rfi_carrier_fluctuations')
            self.rfi_usb_offsets.write(hdf5, 'rfi_usb_offsets')
            self.rfi_usb_fluctuations.write(hdf5, 'rfi_usb_fluctuations')

            logger.debug("Writing total beatnote frequencies to '%s'", output)
            self.isi_carriers.write(hdf5, 'isi_carriers')
            self.isi_usbs.write(hdf5, 'isi_usbs')
            self.tmi_carriers.write(hdf5, 'tmi_carriers')
            self.tmi_usbs.write(hdf5, 'tmi_usbs')
            self.rfi_carriers.write(hdf5, 'rfi_carriers')
            self.rfi_usbs.write(hdf5, 'rfi_usbs')

            logger.debug("Writing MOC time correlations to '%s'", output)
            self.moc_time_correlations.write(hdf5, 'moc_time_correlations')

            logger.info("Closing measurement file '%s'", output)

    def plot_fluctuations(self, output=None, skip=0):
        """Plot beatnote frequency fluctuations generated by the simulation.

        Args:
            output: output file, None to show the plots
            skip: number of initial samples to skip [samples]
        """
        # Run simulation if needed
        if not self.simulated:
            self.simulate()
        # Plot signals
        logger.info("Plotting beatnote frequency fluctuations")
        _, axes = plt.subplots(3, 1, figsize=(16, 18))
        plot = lambda axis, x, label: axis.plot(self.t[skip:], np.broadcast_to(x, self.size)[skip:], label=label)
        for mosa in self.MOSAS:
            plot(axes[0], self.isi_carrier_fluctuations[mosa], mosa)
            plot(axes[1], self.tmi_carrier_fluctuations[mosa], mosa)
            plot(axes[2], self.rfi_carrier_fluctuations[mosa], mosa)
        # Format plot
        axes[0].set_title("Beatnote frequency fluctuations")
        axes[2].set_xlabel("Time [s]")
        axes[0].set_ylabel("Inter-spacecraft frequency [Hz]")
        axes[1].set_ylabel("Test-mass frequency [Hz]")
        axes[2].set_ylabel("Reference frequency [Hz]")
        for axis in axes:
            axis.grid()
            axis.legend()
        # Save or show glitch
        if output is not None:
            logger.info("Saving plot to %s", output)
            plt.savefig(output, bbox_inches='tight')
        else:
            plt.show()

    def plot_offsets(self, output=None, skip=0):
        """Plot beatnote frequency offsets generated by the simulation.

        Args:
            output: output file, None to show the plots
            skip: number of initial samples to skip [samples]
        """
        # Run simulation if needed
        if not self.simulated:
            self.simulate()
        # Plot signals
        logger.info("Plotting beatnote frequency offsets")
        _, axes = plt.subplots(3, 1, figsize=(16, 18))
        plot = lambda axis, x, label: axis.plot(self.t[skip:], np.broadcast_to(x, self.size)[skip:], label=label)
        for mosa in self.MOSAS:
            plot(axes[0], self.isi_carrier_offsets[mosa], mosa)
            plot(axes[1], self.tmi_carrier_offsets[mosa], mosa)
            plot(axes[2], self.rfi_carrier_offsets[mosa], mosa)
        # Format plot
        axes[0].set_title("Beatnote frequency offsets")
        axes[2].set_xlabel("Time [s]")
        axes[0].set_ylabel("Inter-spacecraft frequency [Hz]")
        axes[1].set_ylabel("Test-mass frequency [Hz]")
        axes[2].set_ylabel("Reference frequency [Hz]")
        for axis in axes:
            axis.grid()
            axis.legend()
        # Save or show glitch
        if output is not None:
            logger.info("Saving plot to %s", output)
            plt.savefig(output, bbox_inches='tight')
        else:
            plt.show()

    def plot_totals(self, output=None, skip=0):
        """Plot beatnote total frequencies generated by the simulation.

        Args:
            output: output file, None to show the plots
            skip: number of initial samples to skip [samples]
        """
        # Run simulation if needed
        if not self.simulated:
            self.simulate()
        # Plot signals
        logger.info("Plotting beatnote total frequencies")
        _, axes = plt.subplots(3, 1, figsize=(16, 18))
        plot = lambda axis, x, label: axis.plot(self.t[skip:], np.broadcast_to(x, self.size)[skip:], label=label)
        for mosa in self.MOSAS:
            plot(axes[0], self.isi_carriers[mosa], mosa)
            plot(axes[1], self.tmi_carriers[mosa], mosa)
            plot(axes[2], self.rfi_carriers[mosa], mosa)
        # Format plot
        axes[0].set_title("Beatnote total frequencies")
        axes[2].set_xlabel("Time [s]")
        axes[0].set_ylabel("Inter-spacecraft frequency [Hz]")
        axes[1].set_ylabel("Test-mass frequency [Hz]")
        axes[2].set_ylabel("Reference frequency [Hz]")
        for axis in axes:
            axis.grid()
            axis.legend()
        # Save or show glitch
        if output is not None:
            logger.info("Saving plot to %s", output)
            plt.savefig(output, bbox_inches='tight')
        else:
            plt.show()

    def plot_mprs(self, output=None, skip=0):
        """Plot measured pseudo-ranges (MPRs) generated by the simulation.

        Args:
            output: output file, None to show the plots
            skip: number of initial samples to skip [samples]
        """
        # Run simulation if needed
        if not self.simulated:
            self.simulate()
        # Plot signals
        logger.info("Plotting measured pseudo-ranges")
        _, axes = plt.subplots(2, 1, figsize=(16, 12))
        plot = lambda axis, x, label: axis.plot(self.t[skip:], np.broadcast_to(x, self.size)[skip:], label=label)
        for mosa in self.MOSAS:
            plot(axes[0], self.mprs[mosa], mosa)
            plot(axes[1], np.gradient(self.mprs[mosa], self.dt), mosa)
        # Format plot
        axes[0].set_title("Measured pseudo-ranges")
        axes[1].set_xlabel("Time [s]")
        axes[0].set_ylabel("Pseudo-range [s]")
        axes[1].set_ylabel("Pseudo-range derivative [s/s]")
        for axis in axes:
            axis.grid()
            axis.legend()
        # Save or show glitch
        if output is not None:
            logger.info("Saving plot to %s", output)
            plt.savefig(output, bbox_inches='tight')
        else:
            plt.show()

    def plot_dws(self, output=None, skip=0):
        """Plot DWS measurements generated by the simulation.

        Args:
            output: output file, None to show the plots
            skip: number of initial samples to skip [samples]
        """
        # Run simulation if needed
        if not self.simulated:
            self.simulate()
        # Plot signals
        logger.info("Plotting DWS measurements")
        _, axes = plt.subplots(2, 1, figsize=(16, 12))
        plot = lambda axis, x, label: axis.plot(self.t[skip:], np.broadcast_to(x, self.size)[skip:], label=label)
        for mosa in self.MOSAS:
            plot(axes[0], self.isi_dws_phis[mosa], mosa)
            plot(axes[1], self.isi_dws_etas[mosa], mosa)
        # Format plot
        axes[0].set_title("DWS measurements")
        axes[1].set_xlabel("Time [s]")
        axes[0].set_ylabel("ISI Yaw (phi) [rad/s]")
        axes[1].set_ylabel("ISI Pitch (eta) [rad/s]")
        for axis in axes:
            axis.grid()
            axis.legend()
        # Save or show glitch
        if output is not None:
            logger.info("Saving plot to %s", output)
            plt.savefig(output, bbox_inches='tight')
        else:
            plt.show()

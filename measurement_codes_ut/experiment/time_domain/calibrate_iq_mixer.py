
from logging import getLogger
import os
import numpy as np
import matplotlib.pyplot as plt
from plottr.data.datadict_storage import DataDict, DDH5Writer
from sklearn.decomposition import PCA
from tqdm import tqdm
import time

from sequence_parser.instruction import *

from measurement_codes_ut.helper.plot_helper import PlotHelper
from plottr.data.datadict_storage import datadict_from_hdf5
from scipy.optimize import minimize
from qcodes_drivers.iq_corrector import IQCorrector

logger = getLogger(__name__)


class CalibrateIQMixer(object):
    experiment_name = "CalibrateIQMixer"

    def __init__(self,
                 target_port: str,
                 target_channel: tuple,
                 if_lo: int,  # MHz
                 if_hi: int,  # MHz
                 if_step: int,  # MHz
                 reference_level_rf=0,  # dBm
                 reference_level_leakage=-30,  # dBm
                 i_amp=0.5, ):
        self.dataset = {}
        self.target_port = target_port
        self.target_channel = target_channel
        assert 1000 % if_step == 0
        assert if_lo % if_step == 0
        assert if_hi % if_step == 0
        if_freqs = np.arange(if_lo, if_hi + if_step, if_step)
        self.if_freqs = if_freqs[if_freqs != 0]

        self.reference_level_rf = reference_level_rf
        self.reference_level_leakage = reference_level_leakage
        self.i_amp = i_amp

    def execute(self, tdm, check=True, update_tdm=True):
        if not hasattr(tdm, "spectrum_analyzer"):
            tdm.add_spectrum_analyzer()
        tdm.spectrum_analyzer.span(0)  # Hz
        tdm.spectrum_analyzer.npts(101)
        tdm.spectrum_analyzer.resolution_bandwidth(1e4)  # Hz
        tdm.spectrum_analyzer.video_bandwidth(1e4)  # Hz
        tdm.spectrum_analyzer.reference_level(
            self.reference_level_leakage)  # dBm

        self.awg_index = tdm.awg_ch[self.target_port][0]
        self.awg = tdm.awg[self.awg_index]
        ref_dict = {1: self.awg.ch1, 2: self.awg.ch2,
                    3: self.awg.ch3, 4: self.awg.ch4}
        self.awg_i, self.awg_q = ref_dict[self.target_channel[0]
                                          ], ref_dict[self.target_channel[1]]

        lo = tdm.lo[self.target_port]
        self.lo_freq = lo.frequency()
        tdm.spectrum_analyzer.center(lo.frequency())
        try:
            lo.output(True)
        except:
            lo.on()
        print('Running LO leakage calibration...', end='')
        self.minimize_lo_leakage(tdm)
        print('done')
        print('Running image sideband minimization...', end='')
        self.minimize_image_sideband(tdm)
        print('done')
        print('Running rf power measurement...', end='')
        self.measure_rf_power(tdm)
        print('done')

        if check:
            print('Checking IQ mixer...', end='')
            self.iq_mixer_check(tdm)
            print('done')

        try:
            lo.output(False)
        except:
            lo.off()

        iq_corrector = IQCorrector(
            self.awg_i,  # awg_q2 or awg2_q2
            self.awg_i,  # awg_i2 or awg2_i2
            tdm.save_path,
            lo_leakage_datetime=self.lo_leakage_path,
            rf_power_datetime=self.rf_power_path,
            len_kernel=41,
            fit_weight=10,
        )

        if update_tdm:
            tdm.IQ_corrector[self.target_port] = iq_corrector
            tdm.awg_ch[self.target_port][1] = self.target_channel

        return self.dataset

    def minimize_lo_leakage(self, tdm):

        data = DataDict(
            iteration=dict(),
            i_offset=dict(unit="V", axes=["iteration"]),
            q_offset=dict(unit="V", axes=["iteration"]),
            lo_leakage=dict(unit="dBm", axes=["iteration"]),
        )
        data.validate()

        x0 = 0  # initial guess for i_offset
        x1 = 0  # initial guess for q_offset
        d = 0.1  # initial step size

        name = f"iq_calibrator_lo_leakage_slot{self.awg.slot_number()}_ch{self.awg_i.channel}_ch{self.awg_q.channel}"

        with DDH5Writer(data, tdm.save_path, name=name) as writer:
            tdm.prepare_experiment(writer, __file__)
            iteration = 0

            def measure(i_offset: float, q_offset: float):
                nonlocal iteration
                self.awg_i.dc_offset(i_offset)
                self.awg_q.dc_offset(q_offset)
                dbm = tdm.spectrum_analyzer.trace_mean()
                writer.add_data(
                    iteration=iteration,
                    i_offset=i_offset,
                    q_offset=q_offset,
                    lo_leakage=dbm,
                )
                iteration += 1
                return 10 ** (dbm / 10)

            self.i_offset, self.q_offset = minimize(
                lambda iq_offsets: measure(*iq_offsets),
                [x0, x1],
                method="Nelder-Mead",
                options=dict(
                    initial_simplex=[[x0, x1], [x0 + d, x1], [x0, x1 + d]],
                    xatol=1e-4,
                ),
            ).x
            measure(self.i_offset, self.q_offset)
            files = os.listdir(tdm.save_path)
            date = files[-1] + '/'
            files = os.listdir(tdm.save_path+date)
            data_path = files[-1]

            data_path_all = tdm.save_path+date+data_path + '/'

            print(f"lo_leakage saved in {data_path_all}")
            dataset = datadict_from_hdf5(data_path_all+"data")
            self.lo_leakage_path = data_path[:17]
            self.dataset['lo_leakage'] = dataset

    def output_if(self, if_freq: int, i_amp: float, q_amp: float, theta: float):
        t = np.arange(1000) / 1000
        i = i_amp * np.cos(2 * np.pi * if_freq * t)
        q = q_amp * np.sin(2 * np.pi * if_freq * t + theta)
        self.awg.stop_all()
        self.awg.flush_waveform()
        self.awg_i.dc_offset(self.i_offset)
        self.awg_q.dc_offset(self.q_offset)
        self.awg.load_waveform(i, 0, suppress_nonzero_warning=True)
        self.awg.load_waveform(q, 1, suppress_nonzero_warning=True)
        self.awg_i.queue_waveform(0, trigger="auto", cycles=0)
        self.awg_q.queue_waveform(1, trigger="auto", cycles=0)
        self.awg.start_all()

    def minimize_image_sideband(self, tdm, awg_resolution=1e-3):
        """Minimize image sideband when
            i(t) = i_amp * cos(2π*if_freq*t),
            q(t) = q_amp * sin(2π*if_freq*t + theta)
        are applied to the iq mixer.
        """
        tdm.spectrum_analyzer.span(0)  # Hz
        tdm.spectrum_analyzer.npts(101)
        tdm.spectrum_analyzer.resolution_bandwidth(1e4)  # Hz
        tdm.spectrum_analyzer.video_bandwidth(1e4)  # Hz
        tdm.spectrum_analyzer.reference_level(
            self.reference_level_leakage)  # dBm

        data = DataDict(
            if_freq=dict(unit="MHz"),
            iteration=dict(),
            i_amp=dict(axes=["if_freq", "iteration"]),
            q_amp=dict(axes=["if_freq", "iteration"]),
            theta=dict(axes=["if_freq", "iteration"]),
            image_sideband=dict(unit="dBm", axes=["if_freq", "iteration"]),
        )
        data.validate()

        x0 = self.i_amp  # initial guess for q_amp
        x1 = 0  # initial guess for theta
        d0 = -0.1 * self.i_amp  # initial step size for q_amp
        d1 = 0.1  # initial step size for theta
        self.q_amps = np.full(len(self.if_freqs), np.nan)
        self.thetas = np.full(len(self.if_freqs), np.nan)

        name = f"iq_calibrator_image_sideband_slot{self.awg.slot_number()}_ch{self.awg_i.channel}_ch{self.awg_q.channel}"

        try:
            with DDH5Writer(data, tdm.save_path, name=name) as writer:
                tdm.prepare_experiment(writer, __file__)

                for i in tqdm(range(len(self.if_freqs))):
                    iteration = 0

                    def measure(if_freq: int, i_amp: float, q_amp: float, theta: float):
                        nonlocal iteration
                        self.output_if(if_freq, i_amp, q_amp, theta)
                        tdm.spectrum_analyzer.center(
                            self.lo_freq - if_freq * 1e6)
                        dbm = tdm.spectrum_analyzer.trace_mean()
                        writer.add_data(
                            if_freq=if_freq,
                            iteration=iteration,
                            i_amp=i_amp,
                            q_amp=q_amp,
                            theta=theta,
                            image_sideband=dbm,
                        )
                        iteration += 1
                        return 10 ** (dbm / 10)

                    x0, x1 = minimize(
                        lambda x: measure(self.if_freqs[i], self.i_amp, *x),
                        [x0, x1],
                        method="Nelder-Mead",
                        options=dict(
                            initial_simplex=[[x0, x1], [
                                x0 + d0, x1], [x0, x1 + d1]],
                            xatol=awg_resolution,
                        ),
                    ).x
                    self.q_amps[i] = x0
                    self.thetas[i] = x1
                    measure(
                        self.if_freqs[i], self.i_amp, self.q_amps[i], self.thetas[i]
                    )
                    d0 = 0.01
                    d1 = 0.01
        finally:
            self.awg.stop_all()

    def measure_rf_power(self, tdm):
        """Measure rf_power[mW] when
            i(t) = i_amp * cos(2π*if_freq*t),
            q(t) = q_amp * sin(2π*if_freq*t + theta)
        are applied to the iq mixer.
        """
        tdm.spectrum_analyzer.span(1e9)  # Hz
        tdm.spectrum_analyzer.npts(1001)
        tdm.spectrum_analyzer.resolution_bandwidth(5e6)  # Hz
        tdm.spectrum_analyzer.video_bandwidth(1e4)  # Hz
        tdm.spectrum_analyzer.reference_level(self.reference_level_rf)  # dBm
        tdm.spectrum_analyzer.center(self.lo_freq)

        data = DataDict(
            if_freq=dict(unit="MHz"),
            i_amp=dict(axes=["if_freq"]),
            q_amp=dict(axes=["if_freq"]),
            theta=dict(axes=["if_freq"]),
            lo_leakage=dict(unit="dBm", axis=["if_freq"]),
            image_sideband=dict(unit="dBm", axes=["if_freq"]),
            rf_power=dict(unit="dBm", axes=["if_freq"]),
        )
        data.validate()

        name = f"iq_calibrator_rf_power_slot{self.awg.slot_number()}_ch{self.awg_i.channel}_ch{self.awg_q.channel}"

        try:
            with DDH5Writer(data, tdm.save_path, name=name) as writer:

                tdm.prepare_experiment(writer, __file__)

                for i in tqdm(range(len(self.if_freqs))):
                    self.output_if(
                        self.if_freqs[i], self.i_amp, self.q_amps[i], self.thetas[i]
                    )
                    trace = tdm.spectrum_analyzer.trace()
                    writer.add_data(
                        if_freq=self.if_freqs[i],
                        i_amp=self.i_amp,
                        q_amp=self.q_amps[i],
                        theta=self.thetas[i],
                        lo_leakage=trace[500],
                        image_sideband=trace[500 - self.if_freqs[i]],
                        rf_power=trace[500 + self.if_freqs[i]],
                    )
        finally:
            self.awg.stop_all()
            files = os.listdir(tdm.save_path)
            date = files[-1] + '/'
            files = os.listdir(tdm.save_path+date)
            data_path = files[-1]

            data_path_all = tdm.save_path+date+data_path + '/'

            print(f"rf_power data saved in {data_path_all}")
            dataset = datadict_from_hdf5(data_path_all+"data")
            self.rf_power_path = data_path[:17]
            self.dataset['rf_power'] = dataset

    def iq_mixer_check(self, tdm):
        iq_corrector = IQCorrector(
            self.awg_i,  # awg_q2 or awg2_q2
            self.awg_i,  # awg_i2 or awg2_i2
            tdm.save_path,
            lo_leakage_datetime=self.lo_leakage_path,
            rf_power_datetime=self.rf_power_path,
            len_kernel=41,
            fit_weight=10,
        )
        lo = tdm.lo[self.target_port]
        if "Rohde" in lo.IDN()['model']:
            lo.on()
        else:
            lo.output(True)

        iq_corrector.check(
            files=[__file__],
            data_path=tdm.save_path,
            wiring=tdm.wiring_info,
            station=tdm.station,
            awg=self.awg,
            spectrum_analyzer=tdm.spectrum_analyzer,
            lo_freq=tdm.lo[self.target_port].frequency(),  # Hz
            if_step=10,  # MHz
            amps=np.linspace(0.1, 1.4, 14),
            reference_level=0,  # dBm
        )
        files = os.listdir(tdm.save_path)
        date = files[-1] + '/'
        files = os.listdir(tdm.save_path+date)
        data_path = files[-1]

        data_path_all = tdm.save_path+date+data_path + '/'

        print(f"IQ mixer check data saved in {data_path_all}")
        dataset = datadict_from_hdf5(data_path_all+"data")
        self.dataset['mixer_check'] = dataset

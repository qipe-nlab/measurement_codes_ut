import matplotlib.pyplot as plt
import numpy as np
import qcodes as qc
from numpy.typing import NDArray
from qcodes.dataset.experiment_container import Experiment
from scipy.fft import fft, fftfreq, fftshift, ifftshift
from scipy.ndimage import convolve
from scipy.optimize import least_squares

from qcodes_drivers.E4407B import E4407B
from qcodes_drivers.M3202A import M3202A, SD_AWG_CHANNEL


class IQCorrector:
    def __init__(
        self,
        awg_i: SD_AWG_CHANNEL,
        awg_q: SD_AWG_CHANNEL,
        database_path: str,
        lo_leakage_id: int,
        rf_power_id: int,
        len_kernel=41,
        fit_weight=10,
        plot=False,
    ):
        self.awg_i = awg_i
        self.awg_q = awg_q
        qc.initialise_or_create_database_at(database_path)
        offset_data = qc.load_by_run_spec(
            captured_run_id=lo_leakage_id
        ).to_pandas_dataframe()
        # print(awg_i.dc_offset.full_name)
        i_offset = offset_data[awg_i.dc_offset.full_name].values[-1]
        q_offset = offset_data[awg_q.dc_offset.full_name].values[-1]
        # if awg_i.dc_offset.full_name=="awg_f0g1_ch1_dc_offset":
        #     i_offset = offset_data["awg_ch1_dc_offset"].values[-1]
        #     q_offset = offset_data["awg_ch2_dc_offset"].values[-1]
        #     print('fogi AWG')
        # elif awg_i.dc_offset.full_name=="awg_Qdrive_ch1_dc_offset":
        #     i_offset = offset_data["awg2_ch1_dc_offset"].values[-1]
        #     q_offset = offset_data["awg2_ch2_dc_offset"].values[-1]
        #     print('Qdrive AWG')
        # else:
        #     raise AssertionError("no offset parameter")
        awg_i.dc_offset(i_offset)
        awg_q.dc_offset(q_offset)

        data = qc.load_by_run_spec(captured_run_id=rf_power_id).to_pandas_dataframe(
            "i_amp", "q_amp", "theta", "rf_power"
        )
        measured_if = data.index.values.astype(int)
        measured_i_amp = data["i_amp"].values
        measured_q_amp = data["q_amp"].values
        measured_theta = data["theta"].values
        measured_rf_power = 10 ** (data["rf_power"].values / 10)

        if_step = measured_if[1] - measured_if[0]
        if_freqs = ifftshift(np.arange(-500, 500, if_step))
        i_amps = np.interp(if_freqs, measured_if, measured_i_amp, period=1000)
        q_amps = np.interp(if_freqs, measured_if, measured_q_amp, period=1000)
        thetas = np.interp(if_freqs, measured_if, measured_theta, period=1000)
        rf_powers = np.interp(if_freqs, measured_if,
                              measured_rf_power, period=1000)

        max_amp = max(max(i_amps), max(q_amps))
        i_amps /= max_amp
        q_amps /= max_amp

        # normalize i_amp and q_amp such that rf_powers are equal
        rf_powers /= min(rf_powers)
        i_amps /= np.sqrt(rf_powers)
        q_amps /= np.sqrt(rf_powers)

        def residual(x):
            i_kernel = x.view(complex)[:len_kernel]
            q_kernel = x.view(complex)[len_kernel:]
            i_kernel_full = np.zeros(len(if_freqs), complex)
            q_kernel_full = np.zeros(len(if_freqs), complex)
            i_kernel_full[: len_kernel - len_kernel //
                          2] = i_kernel[len_kernel // 2:]
            q_kernel_full[: len_kernel - len_kernel //
                          2] = q_kernel[len_kernel // 2:]
            i_kernel_full[-(len_kernel // 2):] = i_kernel[: len_kernel // 2]
            q_kernel_full[-(len_kernel // 2):] = q_kernel[: len_kernel // 2]
            i_kernel_f = fft(i_kernel_full)
            q_kernel_f = fft(q_kernel_full)
            i_residual = (i_kernel_f - np.exp(-0.5j * thetas)
                          * i_amps) * weight
            q_residual = (q_kernel_f - np.exp(0.5j * thetas) * q_amps) * weight
            return np.concatenate([i_residual, q_residual]).view(float)

        init_i_kernel = np.zeros(len_kernel, dtype=complex)
        init_q_kernel = np.zeros(len_kernel, dtype=complex)
        init_i_kernel[len_kernel // 2] = 1
        init_q_kernel[len_kernel // 2] = 1
        weight = np.ones(len(if_freqs))
        weight[measured_if // if_step] = fit_weight
        x = least_squares(
            residual,
            np.concatenate([init_i_kernel, init_q_kernel]).view(float),
            method="lm",
        ).x
        i_kernel = x.view(complex)[:len_kernel]
        q_kernel = x.view(complex)[len_kernel:]

        if plot:
            plt.figure("i_kernel")
            plt.plot(i_kernel.real, label="real")
            plt.plot(i_kernel.imag, label="imag")
            plt.legend()
            plt.xlabel("Time (ns)")
            plt.ylabel("Amplitude")

            plt.figure("q_kernel")
            plt.plot(q_kernel.real, label="real")
            plt.plot(q_kernel.imag, label="imag")
            plt.legend()
            plt.xlabel("Time (ns)")
            plt.ylabel("Amplitude")

            i_kernel_full = np.zeros(1000, dtype=np.complex128)
            q_kernel_full = np.zeros(1000, dtype=np.complex128)
            i_kernel_full[: len_kernel - len_kernel //
                          2] = i_kernel[len_kernel // 2:]
            q_kernel_full[: len_kernel - len_kernel //
                          2] = q_kernel[len_kernel // 2:]
            i_kernel_full[-(len_kernel // 2):] = i_kernel[: len_kernel // 2]
            q_kernel_full[-(len_kernel // 2):] = q_kernel[: len_kernel // 2]
            i_kernel_f = fftshift(fft(i_kernel_full))
            q_kernel_f = fftshift(fft(q_kernel_full))
            i_kernel_f_phase = np.angle(i_kernel_f, deg=True)
            q_kernel_f_phase = np.angle(q_kernel_f, deg=True)
            freqs = fftshift(fftfreq(1000, 0.001))

            plt.figure("abs")
            plt.plot(measured_if, i_amps[measured_if // if_step], "o")
            plt.plot(measured_if, q_amps[measured_if // if_step], "o")
            plt.plot(freqs, abs(i_kernel_f), "C0", label="i_kernel")
            plt.plot(freqs, abs(q_kernel_f), "C1", label="q_kernel")
            plt.legend()
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Amplitude")

            plt.figure("phase")
            plt.plot(freqs, i_kernel_f_phase, label="i_kernel")
            plt.plot(freqs, q_kernel_f_phase, label="q_kernel")
            plt.legend()
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Phase (deg)")

            plt.figure("phase imbalance")
            plt.plot(
                measured_if, thetas[measured_if // if_step] / np.pi * 180, "o")
            plt.plot(
                freqs, (q_kernel_f_phase - i_kernel_f_phase +
                        180) % 360 - 180, "C0"
            )
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Phase (deg)")

        self.i_offset = i_offset
        self.q_offset = q_offset
        self.i_kernel = i_kernel
        self.q_kernel = q_kernel

    def correct(self, signal: NDArray[np.complex128], cyclic=False):
        if cyclic:
            i = convolve(signal, self.i_kernel, mode="wrap").real
            q = convolve(signal, self.q_kernel, mode="wrap").imag
        else:
            i = convolve(signal, self.i_kernel, mode="constant").real
            q = convolve(signal, self.q_kernel, mode="constant").imag
        # i+=self.i_offset
        # q+=self.q_offset
        return i, q

    def check(
        self,
        experiment: Experiment,
        wiring: str,
        station: qc.Station,
        awg: M3202A,
        spectrum_analyzer: E4407B,
        lo_freq: float,  # Hz
        if_step: int,  # MHz
        amps: NDArray[np.float64],
    ):
        assert spectrum_analyzer.external_frequency_reference()
        spectrum_analyzer.cont_meas(False)
        spectrum_analyzer.span(1e9)  # Hz
        spectrum_analyzer.npts(1001)
        spectrum_analyzer.resolution_bandwidth(5e6)  # Hz
        spectrum_analyzer.video_bandwidth(1e4)  # Hz
        spectrum_analyzer.reference_level(-10)  # dBm
        spectrum_analyzer.center(lo_freq)

        name = f"iq_corrector check slot{awg.slot_number()} ch{self.awg_i.channel} ch{self.awg_q.channel}"
        measurement = qc.Measurement(experiment, station, name)
        amp_param = qc.Parameter("amp", unit="V")
        if_freq_param = qc.Parameter("if_freq", unit="MHz")
        rf_power_param = qc.Parameter("rf_power", unit="dBm")
        rf_power_linearity_param = qc.Parameter(
            "rf_power_linearity", unit="mW/V^2")
        lo_leakage_param = qc.Parameter("lo_leakage", unit="dBm")
        image_sideband_param = qc.Parameter("image_sideband", unit="dBm")
        measurement.register_parameter(amp_param, paramtype="array")
        measurement.register_parameter(if_freq_param, paramtype="array")
        measurement.register_parameter(
            spectrum_analyzer.freq_axis, paramtype="array")
        measurement.register_parameter(
            spectrum_analyzer.trace,
            setpoints=(amp_param, if_freq_param, spectrum_analyzer.freq_axis),
            paramtype="array",
        )
        measurement.register_parameter(
            rf_power_param, setpoints=(
                amp_param, if_freq_param), paramtype="array"
        )
        measurement.register_parameter(
            rf_power_linearity_param,
            setpoints=(amp_param, if_freq_param),
            paramtype="array",
        )
        measurement.register_parameter(
            lo_leakage_param, setpoints=(
                amp_param, if_freq_param), paramtype="array"
        )
        measurement.register_parameter(
            image_sideband_param,
            setpoints=(amp_param, if_freq_param),
            paramtype="array",
        )

        try:
            with measurement.run() as datasaver:
                datasaver.dataset.add_metadata("wiring", wiring)
                for amp in amps:
                    for if_freq in np.arange(-500 + if_step, 500, if_step):
                        t = np.arange(1000) / 1000
                        signal = amp * np.exp(2j * np.pi * if_freq * t)
                        i, q = self.correct(signal, cyclic=True)
                        awg.stop_all()
                        awg.flush_waveform()
                        self.awg_i.dc_offset(self.i_offset)
                        self.awg_q.dc_offset(self.q_offset)
                        awg.load_waveform(i, 0, suppress_nonzero_warning=True)
                        awg.load_waveform(q, 1, suppress_nonzero_warning=True)
                        self.awg_i.queue_waveform(0, trigger="auto", cycles=0)
                        self.awg_q.queue_waveform(1, trigger="auto", cycles=0)
                        awg.start_all()
                        trace = spectrum_analyzer.trace()
                        rf_power = trace[500 + if_freq]
                        datasaver.add_result(
                            (amp_param, amp),
                            (if_freq_param, if_freq),
                            (
                                spectrum_analyzer.freq_axis,
                                spectrum_analyzer.freq_axis(),
                            ),
                            (spectrum_analyzer.trace, trace),
                            (rf_power_param, rf_power),
                            (
                                rf_power_linearity_param,
                                10 ** (rf_power / 10) / amp**2,
                            ),
                            (lo_leakage_param, trace[500]),
                            (image_sideband_param, trace[500 - if_freq]),
                        )
        finally:
            awg.stop_all()

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import qcodes as qc
from numpy.typing import NDArray
from plottr.data.datadict_storage import DataDict, DDH5Writer, search_datadict
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
        data_path: str,
        lo_leakage_datetime: str,
        rf_power_datetime: str,
        len_kernel=41,
        fit_weight=10,
        plot=False,
    ):
        self.awg_i = awg_i
        self.awg_q = awg_q

        _, datadict = search_datadict(data_path, lo_leakage_datetime)
        i_offset = datadict["i_offset"]["values"][-1]
        q_offset = datadict["q_offset"]["values"][-1]
        awg_i.dc_offset(i_offset)
        awg_q.dc_offset(q_offset)

        _, datadict = search_datadict(data_path, rf_power_datetime)
        measured_if = datadict["if_freq"]["values"]
        measured_i_amp = datadict["i_amp"]["values"]
        measured_q_amp = datadict["q_amp"]["values"]
        measured_theta = datadict["theta"]["values"]
        measured_rf_power = 10 ** (datadict["rf_power"]["values"] / 10)

        if_step = measured_if[1] - measured_if[0]
        if_freqs = ifftshift(np.arange(-500, 500, if_step))
        i_amps = np.interp(if_freqs, measured_if, measured_i_amp, period=1000)
        q_amps = np.interp(if_freqs, measured_if, measured_q_amp, period=1000)
        thetas = np.interp(if_freqs, measured_if, measured_theta, period=1000)
        rf_powers = np.interp(if_freqs, measured_if, measured_rf_power, period=1000)

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
            i_kernel_full[: len_kernel - len_kernel // 2] = i_kernel[len_kernel // 2 :]
            q_kernel_full[: len_kernel - len_kernel // 2] = q_kernel[len_kernel // 2 :]
            i_kernel_full[-(len_kernel // 2) :] = i_kernel[: len_kernel // 2]
            q_kernel_full[-(len_kernel // 2) :] = q_kernel[: len_kernel // 2]
            i_kernel_f = fft(i_kernel_full)
            q_kernel_f = fft(q_kernel_full)
            i_residual = (i_kernel_f - np.exp(-0.5j * thetas) * i_amps) * weight
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
            i_kernel_full[: len_kernel - len_kernel // 2] = i_kernel[len_kernel // 2 :]
            q_kernel_full[: len_kernel - len_kernel // 2] = q_kernel[len_kernel // 2 :]
            i_kernel_full[-(len_kernel // 2) :] = i_kernel[: len_kernel // 2]
            q_kernel_full[-(len_kernel // 2) :] = q_kernel[: len_kernel // 2]
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
            plt.plot(measured_if, thetas[measured_if // if_step] / np.pi * 180, "o")
            plt.plot(
                freqs, (q_kernel_f_phase - i_kernel_f_phase + 180) % 360 - 180, "C0"
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
        return i, q

    def check(
        self,
        files: Sequence[str],
        data_path: str,
        wiring: str,
        station: qc.Station,
        awg: M3202A,
        spectrum_analyzer: E4407B,
        lo_freq: float,  # Hz
        if_step: int,  # MHz
        amps: NDArray[np.float64],
        reference_level=0,  # dBm
    ):
        assert spectrum_analyzer.external_frequency_reference()
        spectrum_analyzer.cont_meas(False)
        spectrum_analyzer.span(1e9)  # Hz
        spectrum_analyzer.npts(1001)
        spectrum_analyzer.resolution_bandwidth(5e6)  # Hz
        spectrum_analyzer.video_bandwidth(1e4)  # Hz
        spectrum_analyzer.reference_level(reference_level)  # dBm
        spectrum_analyzer.center(lo_freq)

        data = DataDict(
            amplitude=dict(unit="V"),
            if_freq=dict(unit="MHz"),
            lo_leakage=dict(unit="dBm", axis=["amplitude", "if_freq"]),
            image_sideband=dict(unit="dBm", axes=["amplitude", "if_freq"]),
            rf_power=dict(unit="dBm", axes=["amplitude", "if_freq"]),
            rf_power_per_amplitude_squared=dict(unit="mW/V", axes=["amplitude", "if_freq"]),
        )
        data.validate()

        spectrum_data = DataDict(
            amplitude=dict(unit="V"),
            if_freq=dict(unit="MHz"),
            frequency=dict(unit="Hz"),
            power=dict(unit="dBm", axes=["amplitude", "if_freq", "frequency"]),
        )
        spectrum_data.validate()

        name = f"iq_corrector check slot{awg.slot_number()} ch{self.awg_i.channel} ch{self.awg_q.channel}"
        spectrum_name = f"iq_corrector check spectrum slot{awg.slot_number()} ch{self.awg_i.channel} ch{self.awg_q.channel}"

        try:
            with DDH5Writer(data, data_path, name=name) as writer, \
                    DDH5Writer(spectrum_data, data_path, name=spectrum_name) as spectrum_writer:
                writer.backup_file(files + [__file__])
                writer.save_text("wiring.md", wiring)
                writer.save_dict("station_snapshot.json", station.snapshot())
                spectrum_writer.backup_file(files + [__file__])
                spectrum_writer.save_text("wiring.md", wiring)
                spectrum_writer.save_dict("station_snapshot.json", station.snapshot())

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
                        writer.add_data(
                            amplitude=amp,
                            if_freq=if_freq,
                            lo_leakage=trace[500],
                            image_sideband=trace[500 - if_freq],
                            rf_power=rf_power,
                            rf_power_per_amplitude_squared=10 ** (rf_power / 10) / amp**2
                        )
                        spectrum_writer.add_data(
                            amplitude=amp,
                            if_freq=if_freq,
                            frequency=spectrum_analyzer.freq_axis(),
                            power=trace,
                        )
        finally:
            awg.stop_all()

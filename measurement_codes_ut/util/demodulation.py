import numpy as np

def demodulate(data, averaging_waveform=True, as_complex=True):
    t = np.arange(data.shape[-1]) * dig_if1a.sampling_interval() * 1e-9
    if averaging_waveform:
        if as_complex:
            return (data * np.exp(2j * np.pi * readout_if_freq * t)).mean(axis=-1)
        else:
            data_comp = (data * np.exp(2j * np.pi *
                         readout_if_freq * t)).mean(axis=-1)
            return np.stack((data_comp.real, data_comp.imag), axis=-1)
    else:
        if as_complex:
            return data * np.exp(2j * np.pi * readout_if_freq * t)
        else:
            data_comp = data * np.exp(2j * np.pi * readout_if_freq * t)
            return np.stack((data_comp.real, data_comp.imag), axis=-1)


def demodulate_weighted(data, window, as_complex=True):
    t = np.arange(data.shape[-1]) * dig_if1a.sampling_interval() * 1e-9
    data_demod = data * np.exp(2j * np.pi * readout_if_freq * t)
    data_weighted = np.dot(data_demod, window)
    if as_complex:
        return data_weighted
    else:
        return np.stack((data_weighted.real, data_weighted.imag), axis=-1)


def stop():
    hvi_trigger.output(False)
    awg1.stop_all()
    awg2.stop_all()
    dig_if1a.stop()
    lo_readout.output(False)
    lo_qubit.output(False)
    # drive_source.output(False)

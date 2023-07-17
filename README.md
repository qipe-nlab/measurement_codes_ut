# measurement_codes_ut

This is a library supporting time-domain experiments using the fridges in Univ. of Tokyo.

This packages aims to complete everything within jupyter notebook without any setup files like setup_td.py.

Users are reccomended to use CalibratonNote class to save/load calibration results instead of manually noting information on setup files.

This package also includes python files to execute automated timedomain basic measurements.

## Installation
```
pip install plottr@git+https://github.com/qipe-nlab/plottr.git@search-datadict
pip install qcodes_drivers@git+https://github.com/qipe-nlab/qcodes_drivers.git
pip install sequence_parser@git+https://github.com/qipe-nlab/sequence_parser.git
pip install measurement_codes_ut@git+https://github.com/qipe-nlab/measurement_codes_ut.git 
```

## Usage
### Import
```python
from measurement_codes_ut.measurement_tool.session import SessionManager as Session
from measurement_codes_ut.measurement_tool.datataking.time_domain import TimeDomainInstrumentManager as TDM
from measurement_codes_ut.measurement_tool import CalibrationNote
from sequence_parser import Sequence, Variable, Variables
from sequence_parser.instruction import *
```

### Create Session object to designate user name, cooling down, and sample name.
```python
session = Session(
    cooling_down_id='CDxxx', 
    experiment_username="YOUR NAME", 
    sample_name="SAMPLE NAME")
```

### Save all the information of measurement instruments in TimeDomainInstrumentManager class. 
```python
tdm = TDM(session, trigger_address="PXI0::1::BACKPLANE", save_path="your-save-directory")

wiring = "\n".join([
    "readout",
    "E8257D(lo1) - 1500mm - LO2",
    "M3202A_slot2_ch1 - 500mm - 10dB - IFin2",
    "RFout2 - 1000mm - 20dB - F-19480 - In1D",
    "Out2A - Miteq - 1500mm - RFin1",
    "IFout1 - 500mm - DCblock - M3102A_slot9_ch1",

    "drive_line",
    "N5173B - 2000mm - LO#4",
    "M3202A_slot2_ch2 - 500mm - 10dB - IFin4",
    "RFout#4 - 1500mm - 10dB - In1C"
])

tdm.set_wiring_note(wiring)

tdm.add_readout_line(
    port_name="readout",
    lo_address="TCPIP0::192.168.100.5::inst0::INSTR",
    lo_power=24,
    awg_chasis=1,
    awg_slot=2,
    awg_channel=1,
    dig_chasis=1,
    dig_slot=9,
    dig_channel=1,
    IQ_corrector=None,
    if_freq=125e6,
    sideband='lower'
)

# For qubit control line
tdm.add_qubit_line(
    port_name="qubit",
    lo_address="TCPIP0::192.168.100.9::inst0::INSTR",
    lo_power=18,
    awg_chasis=1,
    awg_slot=2,
    awg_channel=2,
    IQ_corrector=None,
    if_freq=150e6,
    sideband='lower'
)
```

### Run experiments
```python
readout_freq = 9.23e9
qubit_freq = 8.025e9
num_shot = 1000
tdm.set_acquisition_mode(averaging_waveform=True, averaging_shot=True)
tdm.set_shots(num_shot)
tdm.set_repetition_margin(100e3)

readout_port = tdm.port['readout'].port
qubit_port = tdm.port['qubit'].port
acq_port = tdm.acquire_port['readout_acquire']

tdm.port['readout'].frequency = readout_freq
tdm.port['qubit'].frequency = qubit_freq
ports = [readout_port, qubit_port, acq_port]

dur_range = np.linspace(10, 500, 40, dtype=int)
qubit_amplitude = 1.0
readout_amplitude = 0.4

qubit_amplitude = 0.5
readout_amplitude = 0.4

duration = Variable("duration", np.linspace(10, 200, 20), "ns")
variables = Variables([duration])

seq = Sequence(ports)
seq.add(FlatTop(Gaussian(amplitude=qubit_amplitude, fwhm=10, duration=20, zero_end=True), top_duration=duration), qubit_port)
seq.trigger(ports)
seq.add(ResetPhase(phase=0), readout_port, copy=False)
seq.add(Square(amplitude=readout_amplitude, duration=2000), readout_port)
seq.add(Acquire(duration=2000), acq_port)

seq.trigger(ports)

tdm.sequence = seq
tdm.variables = variables

dataset = tdm.take_data(dataset_name="test", dataset_subpath="test", as_complex=True, exp_file="test.ipynb")

```

### Plot
```python
time = dataset['duration']['values']
cplx = dataset['readout_acquire']['values']

plot = PlotHelper(title="")
plot.plot(time, cplx.real, label='I')
plot.plot(time, cplx.imag, label='Q')
plot.label("Time (ns)", "Signal")
plt.legend()
plt.show()

```

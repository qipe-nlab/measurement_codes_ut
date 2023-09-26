# measurement_codes_ut

This is a library supporting time-domain experiments using the fridges at Univ. of Tokyo.

This package aims to complete everything within jupyter notebook without any setup files like setup_td.py.

Users are recommended to use CalibratonNote class to save/load calibration results instead of manually noting information on setup files.

This package also includes python files to execute automated time-domain basic measurements.

## Installation
```
pip install measurement_codes_ut@git+https://github.com/qipe-nlab/measurement_codes_ut.git 
```
Note this will also install plottr, qcodes_drivers, and sequence-parser as required installation.

## Usage
### Import
Note you have to append these paths to the system path.
```python
import sys
sys.path.append("your-library-path/measurement_codes_ut")
sys.path.append("your-library-path/measurement_codes_ut/measurement_codes_ut")
```

```python
from measurement_codes_ut.measurement_tool import Session
from measurement_codes_ut.measurement_tool.datataking.time_domain import TimeDomainInstrumentManager as TDM
from measurement_codes_ut.measurement_tool import CalibrationNote
from sequence_parser import Sequence, Variable, Variables
from sequence_parser.instruction import *
from measurement_codes_ut.helper import PlotHelper
from measurement_codes_ut.measurement_tool.wrapper import Dataset
```

### Create a Session object to designate the user name, cooling down, and sample name.
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
    "your wiring information"
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
You can use plot-helper.
```python
time = dataset.data['duration']['values']
cplx = dataset.data['readout_acquire']['values']
data_label = str(dataset.number) + "-" + dataset.name
plot = PlotHelper(title=f"{data_label}", columns=1)

plot.plot(time, cplx.real, label='I')
plot.plot(time, cplx.imag, label='Q')
plot.label("Time (ns)", "Response")
plt.tight_layout()
plt.show()
```

### Dataset loading
Your can load your previous data as Dataset.

```python
dataset = Dataset(session)
save_path = f'your-save-directory'
dataset.load(idx, save_path)
```

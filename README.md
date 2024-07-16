# measurement_codes_ut

This is a library supporting time-domain experiments using the fridges at Hongo.

This package aims to complete everything within jupyter notebook without any setup files like setup_td.py.

This package also includes python files to execute automated time-domain basic measurements, but not perfect.

## Installation
Clone this repository to your preferred directory on the experimental PC.

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
    sample_name="SAMPLE NAME",
    save_path="YOUR PREFERRED PATH TO SAVE DATA")
```

### Save all the information of measurement instruments in TimeDomainInstrumentManager class. 
```python
tdm = TDM(session, trigger_address="PXI0::1::BACKPLANE")

wiring = "\n".join([
    "your wiring information"
])

tdm.set_wiring_note(wiring)

dataset_subpath = ""

tdm.add_readout_line(
    port_name="readout",
    lo_address="TCPIP0::192.168.100.7::inst0::INSTR",
    lo_power=17,
    awg_chasis=1,
    awg_slot=2,
    awg_channel=1,
    dig_chasis=1,
    dig_slot=9,
    dig_channel=1,
    IQ_corrector=None,
    if_freq=50e6,
    sideband='lower'
)

# For qubit control line
tdm.add_qubit_line(
    port_name="qubit",
    lo_address="TCPIP0::192.168.100.8::inst0::INSTR",
    lo_power=7,
    awg_chasis=1,
    awg_slot=2,
    awg_channel=2,
    IQ_corrector=None,
    if_freq=100e6,
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

tdm.show_sweep_plan()

dataset = tdm.take_data(dataset_name="test", dataset_subpath=dataset_subpath, as_complex=True, exp_file="test.ipynb")

```

For more advanced experiments, contact and ask your senior.

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
dataset.load(idx, dataset_subpath)
```




import numpy as np
from .base import FittingModelBase

class RamseyInterferenceModel(FittingModelBase):
    # override
    def __init__(self,x,y):
        FittingModelBase.__init__(self, datatype=float)
        self.param_names = [
            "amplitude",
            "decay_time",
            "amplitude_offset",
            "frequency",
            "phase_offset",
        ]
    
    # override
    def _model_function(self, time, amplitude, decay_time, amplitude_offset, frequency, phase_offset):
        y = amplitude*np.exp(-time/decay_time)*np.cos(frequency*time+phase_offset) + amplitude_offset
        return y

    # override
    def initial_guess(self, x, y):
        """
        il = signal.argrelmin(y, order=5)
        bo = self.proj.mean()
        a0 = abs(y[il][0]-(y[il][1]-y[il][0])/(self.param[il][1]-self.param[il][0])*self.param[il][0]-bo)

        y2 = np.log(abs((y[il]-bo)/a0))
        a0_ = np.gradient(y2).mean()/np.gradient(self.param[il]).mean()
        t0 = -1./curve_fit(lambda x,a,b:a*x+b,self.param[il],y2,[a0_,0])[0][0]

        y__ = (y_-bo)/abs(exp_decay(self.param,a0,t0,bo))
        freq,power = fft_calc(y__,self.param)
        ps = freq[power.argmax()]
        po = 0.5 - self.param[il][0]/np.gradient(self.param[il]).mean()
        """



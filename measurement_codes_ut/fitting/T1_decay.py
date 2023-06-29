
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from .base import FittingModelBase

class T1DecayModel(FittingModelBase):
    # override
    def __init__(self):
        FittingModelBase.__init__(self, datatype=float)
        self.param_names = [
            "amplitude",
            "decay_rate",
            "amplitude_offset",
        ]
    
    # override
    def _model_function(self, time, amplitude, decay_rate, amplitude_offset):
        y = amplitude*np.exp(-time*decay_rate) + amplitude_offset
        return y

    # override
    def _initial_guess(self,x,y):
        # log_abs_grad_y = log(amplitude) + log(decay_rate) + log(dx) - time * decay_rate
        log_abs_grad_y = np.log(np.abs(np.gradient(y)) + 1e-15)
        linear_func = lambda val,coef,offset : coef*val+offset
        fit_param, _ = curve_fit(linear_func, x, log_abs_grad_y)
        coef, _ = fit_param
        decay_rate    = -coef

        # abs_dif_y = -amplitude*decay_rate*np.exp(-time*decay_rate)
        amplitude     = np.mean(-1.0 * np.gradient(y)/np.gradient(x) * np.exp(x*decay_rate) / decay_rate)

        # func(x,amp,decay,amp_offset) - func(x,amp,decay,0) = amp_offset
        y_pred_without_offset = self._model_function(x, amplitude, decay_rate, 0)
        amplitude_offset    = np.mean(y-y_pred_without_offset)

        param_dict = {
            'amplitude'         : amplitude,
            'decay_rate'        : decay_rate,
            'amplitude_offset'  : amplitude_offset,
        }
        return param_dict

class T1DecayDualModel(FittingModelBase):
    # override
    def __init__(self):
        FittingModelBase.__init__(self, datatype=float)
        self.param_names = [
            "amplitude",
            "decay_rate",
            "amplitude_offset",
        ]
    
    # override
    def _model_function(self, time, amplitude, decay_rate, amplitude_offset):
        y = np.empty( (2, time.shape[0]) )
        y[0,:] = amplitude*np.exp(-time/decay_rate) + amplitude_offset
        y[1,:] = -amplitude*np.exp(-time/decay_rate) + amplitude_offset
        return y

    # override
    def _initial_guess(self,x,y):
        # log_abs_grad_y = log(amplitude) + log(decay_rate) + log(dx) - time * decay_rate
        y_upper = y[0]
        log_abs_grad_y = np.log(np.abs(np.gradient(y_upper)) + 1e-15)
        linear_func = lambda val,coef,offset : coef*val+offset
        fit_param, _ = curve_fit(linear_func, x, log_abs_grad_y)
        coef, _ = fit_param
        decay_rate    = -coef

        # abs_dif_y = -amplitude*decay_rate*np.exp(-time*decay_rate)
        amplitude     = np.mean(-1.0 * np.gradient(y_upper)/np.gradient(x) * np.exp(x*decay_rate) / decay_rate)

        # func(x,amp,decay,amp_offset) - func(x,amp,decay,0) = amp_offset
        y_pred_without_offset = self._model_function(x, amplitude, decay_rate, 0)
        amplitude_offset    = np.mean(y-y_pred_without_offset)

        param_dict = {
            'amplitude'         : amplitude,
            'decay_rate'        : decay_rate,
            'amplitude_offset'  : amplitude_offset,
        }
        return param_dict
        
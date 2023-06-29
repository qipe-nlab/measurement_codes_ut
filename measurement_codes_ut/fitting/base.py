
import numpy as np
import sys
from scipy.optimize import leastsq

class FittingModelBase(object):
    """Base class of fitting models

    This class is a base class of fitting experimental data with known models.
    Basically, these models are intended to use with the following lines.
    ```
    model = Model()
    model.fit(x,y)
    y_fit = model.predict(x)
    ```
    """

    def __init__(self, param_names, datatype = float):
        """Initiliazer of fitting model class

        Constructor of fitting model class.
        Subclass of Model must list name of fitting parameters in param_names,
         and its order must be the same as the list of arguments of self._model_function except the first argument.

        Args:
            param_names (list(str)): list of names of fitting parameters.
            datatype (type): Return type of model function. Now float or complex is supported. Defaults to float.
        """
        self.param_names = param_names
        self._param_name_to_index = {}
        for index, name in enumerate(self.param_names):
            self._param_name_to_index[name] = index
        self.param_list = [0.] * len(self.param_names)
        self.param_error_list = [np.inf] * len(self.param_names)
        # type of estimation value, float or complex
        self.datatype = datatype
        
    def _model_function(self, *args, x):
        """Core function of model

        For given fitting parameter, returns current estimation.

        Args:
            *args: list of fitting parameters.
            x (float or np.ndarray): input scalar or vector.

        Retruns:
            float, complex, or np.ndarray: output scalar or vector.
            
        """
        raise NotImplementedError("This is pure virtual method")
        
    def _initial_guess(self,x,y):
        """Return fitting parameters with initial guess

        Args:
            x (numpy.ndarray): input vector
            y (numpy.ndarray): output vector

        Retruns:
            dict: dictionary of (key,value) = (param_name, guessed_param_value)

        """
        raise NotImplementedError("This is pure virtual method")

    def _generate_cost_function(self):
        """Generate cost function with ordered argumets for <code>scipy.optimize.leastsq</code>.

        This function returns a function which calculate real-valued vector of errors according to 
        fitting parameters, input vector, and output vector, i.e., (*fitting_parameters, x, y) -> y_errors.
        This function absorbs return-type-dependencey of _model_functions.

        Returns:
            function: function calculating error vector from input values.
        """
        error_func = lambda param_list, x, y : self._model_function(x, *param_list) - y
        complex_datatype = [complex, np.complex64, np.complex128]
        if self.datatype not in complex_datatype:
            return error_func
        else:
            error_func_abs = lambda param_list, x, y : np.abs(error_func(param_list, x, y))
            return error_func_abs

    def fit(self, x, y, guess_param = None):
        """Perform vector fitting with <code>scipy.optimize.leastsq</code>

        Note that as far as input and output can be assumed as a vector, 
        <code>leastsq</code> is basically better than other ones such as scipy.minimize.

        Args:
            x (numpy.ndarray): input vector
            y (numpy.ndarray): output vector
            guess_param (dict): If not None, used as initial guess. Defaults to None.

        Returns:
            str: message given by fitting result
        """
        error_func = self._generate_cost_function()

        if guess_param is None:
            guess_param = self._initial_guess(x,y)
        for key, value in guess_param.items():
            if key not in self._param_name_to_index.keys():
                print("Key:{} is not used".format(key), file=sys.stderr)
            else:
                index = self._param_name_to_index[key]
                self.param_list[index] = value

        # perform fitting
        fitting_result = leastsq(error_func, self.param_list, args=(x,y), full_output=True)

        # store fitted data
        self.param_list = fitting_result[0]
        mean_error = np.sqrt(np.mean(error_func(self.param_list, x, y)**2))
        correction_coef = np.var(error_func(self.param_list, x, y))
        if fitting_result[1] is not None:
            covariance_matrix = fitting_result[1]*correction_coef
            self.param_error_list = np.sqrt(np.diag(covariance_matrix))        

        fitting_message = fitting_result[4]
        return fitting_message, mean_error

    def predict(self, x):
        """Obtain fitted line with input vector

        Args:
            x (numpy.ndarray): input vector

        Returns:
            np.ndarray: fitted output vector
        """
        y_pred = self._model_function(x, *self.param_list)
        return y_pred


    def _check_initial_guess(self, x, y, guess_param=None):
        """Obtain line with initial guess.

        This function is for debugging the performance of initial guess quickly.

        Args:
            x (numpy.ndarray): input vector
            y (numpy.nnarray): output vector
            guess_param (dict): If not None, used as initial guess

        Returns:
            np.ndarray: vector obtained with initial guess
        """

        param_list = np.array(self.param_list)
        param_dict = self._initial_guess(x,y)
        if guess_param is not None:
            param_dict.update(guess_param)
        for key,value in param_dict.items():
            param_list[self._param_name_to_index[key]] = value
        y_pred = self._model_function(x, *param_list)
        return y_pred

    '''
    # hand tuning is temporaly removed since it is unstable and out-dated.
    def hand_tuning(self, x, y, guess_param=None):
        """Tuning initial guess parameter by hand in jupyter notebook

        Tuning initial guess parameter by hand in jupyter notebook.
        When you call this function, jupyter notebook shows interactive window.
        Each fitting parameter can be tuned by hand, and update by pusing "Update" button.

        Args:
            x (numpy.ndarray): input vector
            y (numpy.ndarray): output vector
            guess_param (dict): If not None, used as initial guess
        """

        import matplotlib.pyplot as plt
        from ipywidgets import FloatSlider, RadioButtons, interact, widgets
        from IPython.display import display

        if guess_param is not None:
            param_dict = self._initial_guess(x,y)
        else:
            param_dict = guess_param
        guess_y = self._model_function(x,*self.param_list)

        plt.figure(figsize=(4,4))
        plt.title('hand tunine')
        plt.plot(x,y,'k--',label='data')
        line, = plt.plot(x,guess_y,'r',label='fitting')

        vals = FloatSlider(min=-0.1, max=1.0, step=0.1, value=0.2)
        keys = RadioButtons(options=self.param_names)

        @interact(vals=vals,keys=keys)
        def update(vals,keys):
            param_list = list(vals)
            line.set_ydata(self._model_function(x,*param_list))
            plt.draw()

        start_button = widgets.Button(description="update")
        display(start_button)
        def start_action():
            self.param_dict = param_dict
        start_button.on_click(start_action)
    '''

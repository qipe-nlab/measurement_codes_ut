
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class PlotHelper(object):
    """Helper class for typical data plotting

    This class helps to plot data in unified ways, such as fontsize, colors, etc.
    This class is a simple wrapper of matplotlib, and user can additionally call matplotlib functions between each function call.
    """
    def __init__(self, title, rows = 1, columns = 1, width = 8, height = 6, color_map_name = "Set1"):
        """Initializer of helper class

        Args:
            title (str): Title of whole plot.
            rows (int): Number of rows in subplot matrix.
            column (int): Number of columns in subplot matrix.
            width (flaot): Width of figure for each subplot in inch.
            height (float): Height of figure for each subplot in inch. 
        """
        plt.rcParams['xtick.top']           = True
        plt.rcParams['ytick.right']         = True
        plt.rcParams['ytick.minor.visible'] = False
        plt.rcParams['xtick.major.width']   = 0.5
        plt.rcParams['ytick.major.width']   = 0.5
        plt.rcParams['font.size']           = 16
        plt.rcParams['axes.linewidth']      = 1.0
        plt.rcParams['xtick.direction']     = 'in'
        plt.rcParams['ytick.direction']     = 'in'
        # plt.rcParams['font.family']         = 'arial'
        # plt.rcParams["mathtext.fontset"]    = 'stixsans'

        self.rows = rows
        self.columns = columns
        self.width = width
        self.height = height
        self.plot_count = rows * columns
        self.color_counter = [0] * self.plot_count
        #self.color_map = plt.get_cmap(color_map_name)
        colormap = plt.get_cmap(color_map_name)
        colors = [colormap(1), colormap(0), colormap(2), colormap(3), colormap(4), colormap(5), colormap(6), colormap(7), colormap(8)]
        custom_color_map = matplotlib.colors.ListedColormap(colors, name='from_list')
        self.color_map = plt.get_cmap(custom_color_map)
        plt.figure(figsize=(width*columns, height*rows))
        self.current_index = -1
        self.change_plot(0,0)
        if self.plot_count == 1:
            plt.title(title)
        else:
            plt.suptitle(title)

    def change_plot(self, row, column):
        """Move another subplot.

        Args:
            row (int): row-index of subplot matrix.
            column (int): column-index of subplot matrix.

        Raises:
            IndexError: specified row and column are out of index.
        """
        if not ((0 <= row and row < self.rows) and (0 <= column and column < self.columns)):
            raise IndexError("Subplot matrix shape is {}, but {} is specified.".format((self.rows, self.columns), (row,column)))
        previous_index = self.current_index
        self.current_index = row * self.columns + column
        if previous_index != self.current_index:
            plt.subplot(self.rows, self.columns, self.current_index + 1)
            plt.grid()

    def label(self, xproperty, yproperty, title = None):
        """Add x and y label to plots.

        Args:
            xproperty (tuple or str): Tuple of x-axis variable obtained as data_vault.variable. Or, string such as "Voltage [mV]".
            yproperty (tuple or str): Tuple of y-axis variable obtained as data_vault.variable. Or, string such as "Voltage [mV]".
            title (str, optional): Title of current subplot.
        """
        if title is not None:
            plt.title(title)

        def process_tuple(tuple_data):
            # In labrad, size of independent variable is 2 (label, unit), and size of dependent variable is 3 (label, legend, unit).
            # This function absorbs this difference and returns single str.
            if len(tuple_data) == 2:
                label = "{} [{}]".format(*tuple_data)
            elif len(tuple_data) == 3:
                label = "{} {} [{}]".format(*tuple_data)
            return label

        if isinstance(xproperty, tuple):
            plt.xlabel(process_tuple(xproperty))
        else:
            plt.xlabel(xproperty)

        if isinstance(yproperty, tuple):
            plt.ylabel(process_tuple(yproperty))
        else:
            plt.ylabel(yproperty)

    def _get_color(self, is_same_color):
        # Get color of current plot.
        # If is_same_color is False, increment count of color cycle.
        color = self.color_map(self.color_counter[self.current_index])
        if not is_same_color:
            self.color_counter[self.current_index] += 1
        return color

    def plot(self, x, y, label, is_same_color = False, line_for_data = True):
        """Plot simple one-dimensional data.

        Args:
            x (np.ndarray): values of x-axis
            y (np.ndarray): values of y-axis
            label: label of data plot
            is_same_color: If true, color count is not incremented and next plot is shown with the same color.
            line_for_data: If ture, draw lines connecting plots.
        """
        self.plot_fitting(x,y,label,is_same_color=is_same_color, line_for_data=line_for_data)

    def plot_fitting(self, x, y_data, label = "", y_fit=None, y_init=None, y_processed = None, is_same_color = False, line_for_data = True):
        """Plot some one-dimensional data sharing the same x-values.

        Args:
            x (np.ndarray): values of x-axis
            y_data (np.ndarray): data values of y-axis
            label: label of data plot
            y_fit (np.ndarray, optional): fitted values of y-axis
            y_init (np.ndarray, optional): values of y-axis generated by initial guess in fitting
            y_processed (np.ndarray, optional): values of y-axis after some preprocessing such as LPF.
            is_same_color: If true, color count is not incremented and next plot is shown with the same color.
            line_for_data: If ture, draw lines connecting plots.
        """
        if len(label)>0: label = label + " "
        if line_for_data:
            plt.plot(x,y_data, ".-", label = label + "data", color = self._get_color(is_same_color))
        else:
            plt.plot(x,y_data, ".", label = label + "data", color = self._get_color(is_same_color))

        if y_processed is not None:
            plt.plot(x, y_processed, label=label + "processed", color = self._get_color(is_same_color))

        if y_fit is not None:
            if(len(x)!=len(y_fit)):
                plt.plot(np.linspace(min(x),max(x),len(y_fit)), y_fit, label=label + "fit", color = self._get_color(is_same_color), linestyle = '--')
            else:
                plt.plot(x, y_fit, label=label + "fit", color = self._get_color(is_same_color), linestyle = '--')

        if y_init is not None:
            plt.plot(x, y_init, label=label + "initial guess", linestyle="--", color = self._get_color(is_same_color))

        if y_processed is not None or y_fit is not None or y_init is not None or len(label)>0:
            plt.legend()

    def plot_complex(self, data, label = "", fit=None, init=None, processed=None, is_same_color = False, line_for_data = True, adjust_datalimit = True):
        """Plot scattering plot for IQ-response.

        In this plot, the scales of x-axis and y-axis are shared.

        Args:
            x (np.ndarray): values of x-axis
            data (np.ndarray): data values of y-axis
            label: label of data plot
            fit (np.ndarray, optional): fitted values of y-axis
            y_init (np.ndarray, optional): values of y-axis generated by initial guess in fitting
            processed (np.ndarray, optional): values of y-axis after some preprocessing such as LPF.
            is_same_color: If true, color count is not incremented and next plot is shown with the same color.
            line_for_data: If ture, draw lines connecting plots.
            adjust_datalimit: If true, data-range is adjusted so that x-axis and y-axis have the same aspect. If not, figure box is adjusted instead.
        """
        if len(label)>0: label = label + " "
        if line_for_data:
            plt.plot(np.real(data), np.imag(data), ".-", label = label + "data", color = self._get_color(is_same_color))
        else:
            plt.plot(np.real(data), np.imag(data), ".", label = label + "data", color = self._get_color(is_same_color))

        if processed is not None:
            plt.plot(np.real(processed), np.imag(processed), label=label + "processed", color = self._get_color(is_same_color))

        if fit is not None:
            plt.plot(np.real(fit), np.imag(fit), label=label + "fit", color = self._get_color(is_same_color), linestyle = '--')

        if init is not None:
            plt.plot(np.real(fit), np.imag(fit), label=label + "initial guess", linestyle="--", color = self._get_color(is_same_color))

        if processed is not None or fit is not None or init is not None or len(label)>0:
            plt.legend()

        if adjust_datalimit:
            plt.gca().set_aspect('equal', adjustable="datalim")
        else:
            plt.gca().set_aspect('equal', adjustable='box')


    def plot_2d_heatmap(self, x, y, matrix, label = "", is_phase = False):
        """Plot 2d matrix

        Args:
            x (np.ndarray): 1d-array of x-axis values
            y (np.ndarray): 1d-array of y-axis values
            matrix (np.ndarray): 2d-array with shape (len(x), len(y))
            label (str): label of data plot
        """
        def make_frame(array):
            assert(len(array)>=2)
            frame = np.concatenate([
                [array[0]-(array[1]-array[0])/2] ,
                (array[1:] + array[:-1])/2,
                [array[-1]+(array[-1]-array[-2])/2] ,
            ])
            return frame
        frame_x = make_frame(x)
        frame_y = make_frame(y)
        mesh_x, mesh_y = np.meshgrid(frame_x, frame_y)
        if not is_phase:
            cmap = "viridis"
        else:
            cmap = "hsv"
            #plt.gca().set_zlim(-np.pi,np.pi)

        plt.pcolormesh(mesh_x, mesh_y, matrix.T, cmap=cmap, shading="auto")
        plt.axis([x[0], x[-1], y[0], y[-1]])
        plt.colorbar()
        #plt.tight_layout()


    def plot_2d_listplot(self, x, y, matrix, label = "", is_phase = False):
        """Plot 2d matrix

        Args:
            x (np.ndarray): 1d-array of x-axis values
            y (np.ndarray): 1d-array of y-axis values
            matrix (np.ndarray): 2d-array with shape (len(x), len(y))
            label (str): label of data plot
        """
        height = np.max(matrix) - np.min(matrix)
        for index,yval in enumerate(y):
            array = matrix[:,index]
            plt.plot(x,array+height*index,label="y={}".format(yval))
        plt.legend()

    def axvspan(self, xmin, xmax, facecolor="k", alpha=0.3):
        plt.axvspan(xmin=xmin,xmax=xmax,facecolor=facecolor,alpha=alpha)

    def xlim(self, xs, xe):
        plt.xlim(xs, xe)

    def ylim(self, ys, ye):
        plt.ylim(ys, ye)

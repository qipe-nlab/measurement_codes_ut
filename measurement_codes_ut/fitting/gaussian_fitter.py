import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器
from sklearn.metrics import confusion_matrix
from scipy import signal
from scipy.optimize import minimize

class GaussianFitter:
    def __init__(self, data0, data1, n_peak, grid):
        if n_peak != 1 and n_peak != 2:
            raise ValueError('n>2 is not supported.')
        
        
        self.pca = PCA(n_components=2)
        self.data0 = data0
        self.data1 = data1
        train_data = np.vstack([self.data0, self.data1])
        self.pca.fit(train_data)
        data0_d = self.pca.transform(self.data0)
        self.data0_d = data0_d[:,0]
        data1_d = self.pca.transform(self.data1)
        self.data1_d = data1_d[:,0]
        self.n = n_peak
        self.grid = grid
        
        self.mean1 = np.mean(self.data0_d)
        self.std1 = np.std(self.data0_d)
        self.mean2 = np.mean(self.data1_d)
        self.std2 = np.std(self.data1_d)
        
    def get_histograms(self):
        grid = self.grid
        self.grid = grid

        xmin, xmax = min(min(self.data0_d), min(self.data1_d)), max(max(self.data0_d), max(self.data1_d))

        data0_hist,bins=np.histogram(self.data0_d,bins=grid,range=(xmin,xmax))
        data1_hist,bins=np.histogram(self.data1_d,bins=grid,range=(xmin,xmax))
        
        self.xmin, self.xmax = xmin, xmax
        
        return data0_hist, data1_hist
    
    def gaussian(self, x, sig, A1, C1):
        inv_var = 1/(2*sig**2)
        y = A1*np.exp(-inv_var*(x-C1)**2)                      
        return y
    
    def d_gaussian(self, x, sig1, A1, C1, A2, C2, sig2):
        inv_var1 = 1/(2*sig1**2)
        y1 = A1*np.exp(-inv_var1*(x-C1)**2)  
        inv_var2 = 1/(2*sig2**2)
        y2 = A2*np.exp(-inv_var2*(x-C2)**2)
        return y1+y2
    
    def t_gaussian(self, x, sig1, A1, C1, sig2, A2, C2, sig3, A3, C3):
        inv_var1 = 1/(2*sig1**2)
        y1 = A1*np.exp(-inv_var1*(x-C1)**2)  
        inv_var2 = 1/(2*sig2**2)
        y2 = A2*np.exp(-inv_var2*(x-C2)**2)   
        inv_var3 = 1/(2*sig3**2)
        y3 = A3*np.exp(-inv_var3*(x-C3)**2) 
        return y1+y2+y3
                          
        
    def fitter(self):
        n = self.n
        data0_hist, data1_hist = self.get_histograms()
        self.data0_hist = data0_hist
        self.data1_hist = data1_hist
        x = np.linspace(self.xmin, self.xmax, self.grid)
        self.x = x  
        
        p_init1, p_init2 = self.get_init()
        if n == 1:
            popt_1, pcov = curve_fit(self.gaussian, x, data0_hist, maxfev=100000, p0=p_init1)
            popt_2, pcov = curve_fit(self.gaussian, x, data1_hist, maxfev=100000, p0=p_init2)
        if n == 2:
            popt_1, pcov = curve_fit(self.d_gaussian, x, data0_hist, maxfev=100000, p0=p_init1)
            popt_2, pcov = curve_fit(self.d_gaussian, x, data1_hist, maxfev=100000, p0=p_init2)
        
#         popt_1 = self.optimizer(self.residual, x, data0_hist, p0=p_init1)
#         popt_2 = self.optimizer(self.residual, x, data1_hist, p0=p_init2)
        
        high1 = popt_1[n::2]
        peak1 = popt_1[n+1::2]
        high2 = popt_2[n::2]
        peak2 = popt_2[n+1::2]
        self.threshold = (peak1[np.argmax(high1)] + peak2[np.argmax(high2)])/2
        
        self.popt_1 = popt_1
        self.popt_2 = popt_2
        
        return popt_1, popt_2
    
    def get_init(self):
        n = self.n
        data0_hist, data1_hist = self.get_histograms()
        A1_init = max(data0_hist)
        A2_init = max(data1_hist)
        C1_init = self.mean1
        C2_init = self.mean2
        sig1_init = self.std1
        sig2_init = self.std2
        if n == 1:
            p_init1 = [sig1_init, A1_init, C1_init]
            p_init2 = [sig2_init, A2_init, C2_init]
            return p_init1, p_init2
        if n == 2:
            p_init1 = [sig1_init, A1_init, C1_init, sig2_init, 0.1*A1_init, C2_init]
            p_init2 = [sig2_init, A2_init, C2_init, sig1_init, 0.1*A2_init, C1_init]
            return p_init1, p_init2
    
    def plot(self):
        n = self.n
        popt_1, popt_2 = self.fitter()
#         print(popt_1, popt_2)
        if n==1:
            data0_fit = self.gaussian(self.x, *popt_1)
            data1_fit = self.gaussian(self.x, *popt_2)
        if n==2:
            data0_fit = self.d_gaussian(self.x, *popt_1)
            data1_fit = self.d_gaussian(self.x, *popt_2)
        if n==3:
            data0_fit = self.t_gaussian(self.x, *popt_1)
            data1_fit = self.t_gaussian(self.x, *popt_2)
        
        x = self.x
        data0_hist, data1_hist = self.get_histograms()
        
        plt.figure()
        plt.plot(x, data0_fit, '--', color='black')
        plt.plot(x, data0_hist, 'o', markersize=2, color='blue', label='g')

        plt.plot(x, data1_fit, '--', color='black')
        plt.plot(x, data1_hist, 'o', markersize=2, color='red', label='e')
        
        plt.axvline(self.threshold, linestyle='--', color='black', label='threshold')

        plt.ylabel("Counts")
        plt.xlabel("Amplitude")
        plt.legend()
        plt.xlim(self.xmin, self.xmax)
        
        plt.show()
        
    def get_pred(self):
        if np.mean(self.data0_d) > self.threshold:
            data0_pred = [0 if d >= self.threshold else 1 for d in self.data0_d]
            data1_pred = [0 if d >= self.threshold else 1 for d in self.data1_d]
        else:
            data0_pred = [0 if d <= self.threshold else 1 for d in self.data0_d]
            data1_pred = [0 if d <= self.threshold else 1 for d in self.data1_d]
            
        pred = data0_pred + data1_pred
        
        label = [0]*len(self.data0_d) + [1]*len(self.data1_d)
            
        conf_mat = confusion_matrix(label, pred)
        return pred, conf_mat
    
                
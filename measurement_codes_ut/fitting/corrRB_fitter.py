import numpy as np
import copy
from scipy.optimize import least_squares, curve_fit
import matplotlib.pyplot as plt


class CorrRBFitter:
    def __init__(self, notes, L_list, rep):
        rb_pattern = []
        for name in notes.keys():
            q_str = copy.deepcopy(name)
            q = int(q_str.replace('Q', ''))
            rb_pattern.append([q])
        self.L_list = L_list
        self.rep = rep
        self.notes = notes
        self._n_partitions = len(rb_pattern)
        self._n_subsystems = 2**self._n_partitions-1
        self._subsystems = []
        self._subsystems2 = []
        for i in range(self._n_subsystems):
            # this gives the correlator in the subsystem
            # representation
            self._subsystems.append(
                ("{0:0%db}" % self._n_partitions).format(i+1))

            # expand into the full qubit representation
            # and save as an integer
            tmplist2 = []
            for ii in range(self._n_partitions):
                for kk in range(len(rb_pattern[ii])):
                    tmplist2.append(self._subsystems[-1][ii])

            self._subsystems2.append(int(''.join(tmplist2), base=2))

        self._r_coeff = None

        self._rb_pattern = rb_pattern
        self._nq = sum([len(i) for i in rb_pattern])

        self.statedict = {("{0:0%db}" % self._nq).format(i)
                           : 1 for i in range(2**self._nq)}

        self._zdict_ops = {}
        for subind, subsys in enumerate(self._subsystems):
            self._zdict_ops[subsys] = self.statedict.copy()
            for i in range(2**self._nq):
                # if the operator (expressed as an integer) and the state
                # overlap is even then the z operator is '1'
                # otherwise it's -1
                if bin(i & self._subsystems2[subind]).count('1') % 2 != 0:
                    self._zdict_ops[subsys][(
                        "{0:0%db}" % len(rb_pattern)).format(i)] = -1

    def get_zcorr(self, prob_total):
        zcorr = {}
        for _key in self._zdict_ops:
            zcorr[_key] = np.zeros((len(self.L_list), self.rep))
            for key in prob_total:
                zcorr[_key] += self._zdict_ops[_key][key] * \
                    np.array(prob_total[key])
        return zcorr

    def fit_rb_decay(self, zcorr, plot=True):
        def RB_function(x, p, A, B):
            return A * p**x + B

        def fit_rb(x, y):
            p_0 = 0.99
            A_0 = y[0] - y[-1]
            B_0 = y[-1]
            p_init = [p_0, A_0, B_0]
            popt, pcov = curve_fit(RB_function, x, y, p0=p_init, maxfev=10000)
            return popt, pcov
        
        self._alpha_dict = {}
        self._alpha_error = {}
        figs = {}
        for key, value in zcorr.items():
            mean = np.mean(value, axis=1)
            std = np.std(value, axis=1)
            fitter_rb, cov_rb = fit_rb(self.L_list, mean)
            L_fit = np.linspace(0, max(self.L_list), 101)
            y_fit_rb = RB_function(L_fit, *fitter_rb)
            label = ''
            for s in key:
                if s == '0':
                    label = label + 'I'
                else:
                    label = label + 'Z'
            print(f'<{label}> : p = {fitter_rb[0]}')
            
            if plot:
                fig = plt.figure(figsize=(3, 3))
                plt.errorbar(self.L_list,
                            mean,
                            yerr=std,
                            fmt='o',
                            color='red',
                            label=f'<{label}> : p = {fitter_rb[0]}',
                            lw=0.5,
                            capsize=2)

                plt.plot(L_fit,
                        y_fit_rb,
                        color='red',
                        ls='--',
                        lw=1)
                plt.ylabel(f'<{label}>')
                plt.xlabel('Sequence length')
                plt.ylim(-1.1, 1.1)
                plt.legend()
                # plt.show()
                figs[key] = fig

            self._alpha_dict[key] = fitter_rb[0]
            self._alpha_error[key] = np.sqrt(cov_rb[0,0])
            

        return figs

    def _precalc_r_coeff(self):
        # precompute the a matrix of coefficients for calculating the rauli's
        # from this matrix alpha_k = prod[1-rauli_matrix_coeff*e] where e is the error vector
        self._r_coeff = np.zeros(
            [len(self._subsystems), len(self._subsystems)], dtype=float)
        for (ii, subspace1) in enumerate(self._subsystems):
            for (kk, subspace2) in enumerate(self._subsystems):

                depol1 = self._depol_Rauli([subspace2], subspace1, 1.)[0]
                depol2 = self._depol_Rauli([subspace2], subspace1, 0.)[0]

                self._r_coeff[ii, kk] = depol1-depol2

    def _depol_Rauli(self, pauli_elem, depol_subsys, p):

        rauli = np.zeros(len(pauli_elem), dtype=float)
        for (state_index, state_paulis) in enumerate(pauli_elem):
            all_id = 1
            for j in range(len(state_paulis)):
                if depol_subsys[j] == '1' and state_paulis[j] == '1':
                    all_id = 0
            rauli[state_index] = (1.0-p) + p*all_id

        return rauli

    def _calc_alphas(self, eps_list):
        """
        Calculate the alpha from the fixed weight depolarizing errors
        The coefficient matrix has been precalculated
        Args:
            eps_list: list of epsilons from the error model (ordered
            using the self._subsystems list)
        Returns:
            List of alphas given eps_list
        """

        alpha_calc = np.ones(len(eps_list), dtype=float)

        for ii in range(len(eps_list)):
            for jj in range(len(eps_list)):
                alpha_calc[ii] *= (1.0+self._r_coeff[jj, ii]*eps_list[jj])
        return alpha_calc

    def _alpha_diff_func(self, eps_list):
        """
        Calculate the difference between the experimental alphas and
        the alphas calculated from the values in eps_dict
        Args:
            eps_dict: list of epsilons from the error model (ordered
            using the self._subsystems list)
        Returns:
            List of the differences between the calculated and measured
            alpha
        """

        alpha_calc_list = self._calc_alphas(eps_list)
        diff_list = [self._alpha_dict[self._subsystems[i]]-alpha_calc_list[i] for i
                     in range(len(self._subsystems))]

        return diff_list

    def fit_alphas_to_epsilon(self, init_guess=None):
        """
        Fit the set of decay coefficients to the fixed weight depolarizing
        map to get the set of epsilon
        Args:
            init_guess: starting point for the fit. If none will use 0.01
        Returns:
            cost value of the optimization
        """

        if self._r_coeff is None:
            # precalculate
            self._precalc_r_coeff()

        if init_guess is None:
            init_guess = 0.01 * \
                np.ones(len(self._alpha_dict.keys()), dtype=float)

        optm_results = least_squares(self._alpha_diff_func,
                                     init_guess, bounds=[0, 1.0])

        est_err = np.sqrt(optm_results.cost *
                          np.diag(np.linalg.inv(np.dot(
                                  np.transpose(optm_results.jac),
                                  optm_results.jac)))/len(init_guess))

        self._epsilons = {}
        for i, j in enumerate(self._subsystems):
            self._epsilons[j] = [optm_results.x[i], est_err[i]]

        return optm_results.cost

    def plot_epsilon(self, ax=None):
        """
        Plot the set of epsilons (need to run fit_epsilon first)
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            # ax = plt.gca()

        # this is the data as a dictionary
        eps_data = self._epsilons

        corr_list = list(eps_data.keys())
        eps_list = []
        eps_err_list = []
        for corr in corr_list:
            eps_list.append(eps_data[corr][0])
            eps_err_list.append(eps_data[corr][1])

        # sort based on weight
        corr_weight = [i.count('1') for i in corr_list]
        sort_args = np.argsort(corr_weight)

        corr_list = np.array(corr_list)[sort_args]
        eps_list = np.array(eps_list)[sort_args]
        eps_err_list = np.array(eps_err_list)[sort_args]

        color_list = ['black', 'red', 'blue', 'green', 'yellow']
        prev_weight = 0
        for ii, jj in enumerate(corr_list):
            point_weight = jj.count('1')
            if prev_weight != point_weight:
                legend_label = 'Weight %d' % point_weight
                prev_weight = point_weight
            else:
                legend_label = '_nolegend_'

            ax.errorbar(ii+1, eps_list[ii], yerr=eps_err_list[ii],
                        label=legend_label, linestyle='none',
                        marker='o',
                        color=color_list[point_weight-1])

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlim([0, len(corr_weight)+1])
        ax.set_ylim(bottom=0-np.max(eps_err_list))
        ax.grid()
        # add x-tick labels
        ax.set_xticks([y+1 for y in range(len(corr_weight))])
        ax.set_xticklabels(corr_list)
        ax.get_figure().autofmt_xdate()
        ax.set_ylabel('Probability of Error')
        ax.set_xlabel('Subsystem Correlator')
        ax.tick_params()

        # return fig

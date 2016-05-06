# This Python file uses the following encoding: utf-8

# Copyright (C) 2016 Dumur Étienne
# etienne.dumur@gmail.com

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# Update 0.3 (26.11.2015):
#     - Correct hamiltonian_exp method
#     - Add parameters method in plot_spectrum method
#     - Add plot_probability_density method
#     - Add eigen_states method

# Update 0.2 (22.11.2015):
#     - Add plot_spectrum method

import numpy as np
from scipy.special import eval_genlaguerre
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class Fluxonium(object):



    def __init__(self, Ej, Ec, El):
        """
        Class to calculate the energy spectrum of the fluxonium.
        Two method can be used.
        The 'perturbation' method use the harmonic basis and hanlde the
        Josephson potential as a perturbation.
        The 'exponential' one use also the harmonic basis but handle the
        Josephson potential by taking the exponential of coupling matrix.
        Usually, the 'perturbation' method is faster that's why it used
        by default.

        Before calculating any spectrum, use the method find_required_dimension
        to calculate the required dimension of the matrix you need to get
        precised energy levels.

        ...

        Attributes
        ----------
        Ej : float
            Josephson energy in GHz
        Ec : float
            Charging energy in GHz
        El : float
            Inductance energy in GHz

        Raises
        ------
        ValueError
            If the parameters are not in the good type.
        """

        if type(Ej) is not float:
            raise ValueError('Ej parameter must be float type.')
        if type(Ec) is not float:
            raise ValueError('Ec parameter must be float type.')
        if type(El) is not float:
            raise ValueError('El parameter must be float type.')

        self.Ej = Ej
        self.Ec = Ec
        self.El = El



    def hamiltonian(self, n, phi):
        """
        Return the Hamiltonian matrix with energy in GHz.

        Parameters
        ----------
        n : int
            Dimension of the matrix.
        phi : float
            Reduced Φ_ext/Φ_0.

        Returns
        ----------
        Hamiltonian : np.matrix
            Hamiltonian matrix.
        """

        h = np.zeros([n + 1, n + 1])

        counter = 0
        for row in range(n+1):

            for column in range(n+1):
                if row > column:
                    h[row, column] = self.coupling(column, row, phi)
                elif row == column:
                    h[row, column] = self.harmonic_oscilator_energy(row)\
                                    + self.coupling(column, row, phi)
                else:
                    h[row, column] = self.coupling(row, column, phi)

                counter += 1

        return np.asmatrix(np.reshape(h, (n+1, n+1)))



    def hamiltonian_exp(self, n, phi):
        """
        Return the Hamiltonian matrix with energy in GHz calculated with
        the exponential method.

        Parameters
        ----------
        n : int
            Dimension of the matrix.
        phi : float
            Reduced Φ_ext/Φ_0.

        Returns
        ----------
        Hamiltonian : np.matrix
            Hamiltonian matrix.
        """

        h   = np.zeros([n + 1, n + 1])
        c_p = np.zeros([n + 1, n + 1])
        c_m = np.zeros([n + 1, n + 1])

        counter = 0

        for row in range(n+1):

            for column in range(n+1):
                if row == column:
                    h[row, column] = self.harmonic_oscilator_energy(row)
                elif row == column - 1:
                    c_p[row, column] = np.sqrt(row+1.)
                elif row == column + 1:
                    c_m[row, column] = np.sqrt(row)

                counter += 1

        h   = np.matrix(h)
        c_p = np.matrix(c_p)
        c_m = np.matrix(c_m)

        phi_0 = (8.*self.Ec/self.El)**(1./4.)

        a =  1./2.*np.exp(-1j*phi*np.pi*2.)*expm(1j*phi_0/np.sqrt(2.)*(c_p + c_m))\
            +1./2.*np.exp(1j*phi*np.pi*2.)*expm(-1j*phi_0/np.sqrt(2.)*(c_p + c_m))

        return h - self.Ej*a



    def eigen_energies(self, n, phi, method='perturbation'):
        """
        Return the eigenenergies of the Hamiltonian matrix in GHz.
        The method subtract the fondamental energy to all other ones and delete
        the ground energy obtained.
        In other words, the method return the transitions in GHz.

        Parameters
        ----------
        n : int
            Dimension of the matrix.
        phi : float
            Reduced flux Φ_ext/Φ_0.
        method {'perturbation', 'exponential'} : string, optional
            Method used to get eigenenergies.

        Returns
        ----------
        Energies : np.ndarray
            Eigenenergies of the matrix

        Raises
        ------
        ValueError
            If the parameters are not in the good type.
        """

        # We get the matrix depending on the ask method
        if method == 'perturbation':

            h = self.hamiltonian(n, phi)
        elif method == 'exponential':

            h = self.hamiltonian_exp(n, phi)
        else:
            raise ValueError("Wrong method input parameter. Should be 'perturbation' or 'exponential")

        # We get the sorted eigenenergies
        temp = np.sort(np.linalg.eigvalsh(h))

        # We return the transition energies without the ground level
        return (temp-temp[0])[1:]



    def eigen_states(self, n, phi, method='perturbation'):
        """
        Return the eigenstates of the Hamiltonian matrix in GHz.
        The method subtract the fondamental energy to all other ones and delete
        the ground energy obtained.
        In other words, the method return the transitions in GHz.

        Parameters
        ----------
        n : int
            Dimension of the matrix.
        phi : float
            Reduced flux Φ_ext/Φ_0.
        method {'perturbation', 'exponential'} : string, optional
            Method used to get eigenestates.

        Returns
        ----------
        States : 2D np.array
            Eigenstates of the matrix.
            The eigenstates are sorted from the smallest to the highest one.
            The 2D array is return such as states[0] corresponds to the
            eigenstate of the smallest eigenvalues.

        Raises
        ------
        ValueError
            If the parameters are not in the good type.
        """

        # We get the matrix depending on the ask method
        if method == 'perturbation':

            h = self.hamiltonian(n, phi)
        elif method == 'exponential':

            h = self.hamiltonian_exp(n, phi)
        else:
            raise ValueError("Wrong method input parameter. Should be 'perturbation' or 'exponential")

        # We get the sorted eigenvalues
        w, v = np.linalg.eigh(h)
        idx = w.argsort()
        v = v[:,idx]

        # We return the transition energies without the ground level
        return np.transpose(v.real)


    def coupling(self, n, m, phi):
        """
        Return the out diagonal terms of the Hamiltonian matrix in GHz.

        Parameters
        ----------
        n : int
            Row number
        m : int
            Column number
        phi : float
            Reduced flux Φ_ext/Φ_0.

        Returns
        ----------
        Energies : np.array
            Eigenenergies of the matrix
        """

        # temp variable
        a = np.sqrt(8.*self.Ec/self.El)

        # We compute only the part of the fraction which is not simplified
        # afterwards
        b = np.sqrt(1./np.array(range(m+1)[n+1:], float).prod())

        # If m - n is odd
        if (m - n)%2:

            return -self.Ej*np.sin(phi*np.pi*2.)*np.sqrt(2.**n/2.**m)\
                    *a**((m - n)/2.)*b*(-1.)**((m - n - 1.)/2.)\
                    *np.exp(-a/4.)*eval_genlaguerre(n, m - n, a/2.)
        # If it is even
        else:

            return -self.Ej*np.cos(phi*np.pi*2.)*np.sqrt(2.**n/2.**m)\
                    *a**((m - n)/2.)*b*(-1.)**((m - n)/2.)\
                    *np.exp(-a/4.)*eval_genlaguerre(n, m - n, a/2.)



    def harmonic_oscilator_energy(self, n):
        """
        Return the energy of the harmonic oscillator in GHz as
        ν⋅(n + 0.5).                      ___________
        The frequency is defined as ν = ╲╱ 8⋅E_c⋅E_l

        Parameters
        ----------
        n : int
            Level number
        """

        return np.sqrt(8.*self.Ec*self.El)*(n + 0.5)



    def find_required_dimension(self, n, precision, method='perturbation'):
        """
        Find the matrix dimension you have to take into account to reach
        the required precision on the n

        Parameters
        ----------
        n : int
            Level number
        precision : float
            Precision on the calculated energy transition in GHz
        method : {'Perturbation'|'exponential'} : string, optional
            Method used to get eigenenergies.

        Returns
        ------
        level : int
            Number of level which shoulb be taken into account.
        """

        a = self.eigen_energies(n, 0., method=method)[-1]
        b = self.eigen_energies(n + 1, 0., method=method)[-2]

        counter = 2
        while abs(a-b) > precision:
            a = b
            b = self.eigen_energies(n + counter, 0., method=method)[-counter-1]
            counter += 1

        phi_0 = n+counter

        a = self.eigen_energies(n, 0.5, method=method)[-1]
        b = self.eigen_energies(n + 1, 0.5, method=method)[-2]

        counter = 2
        while abs(a-b) > precision:
            a = b
            b = self.eigen_energies(n + counter, 0.5, method=method)[-counter-1]
            counter += 1

        phi_5 = n+counter

        return max(phi_0, phi_5)



    def full_potential(self, x, phi):
        """
        Return the potential in GHz as:
        V = E_l⋅x²/2 - E_j⋅cos(x - 2 π Φ_ext/Φ_0)
        """

        return self.El*x**2./2. - self.Ej*np.cos(x - phi*2.*np.pi)



    def harmonic_potential(self, x):
        """
        Return the kinetic potential in GHz as:
        V = E_l⋅x²/2 - E_j⋅cos(x - 2 π Φ_ext/Φ_0)
        """

        return self.El*x**2./2.



    def josephson_potential(self, x, phi):
        """
        Return the Josephson potential in GHz as:
        V = E_j⋅cos(x - 2 π Φ_ext/Φ_0)
        """

        return -self.Ej*np.cos(x - phi*2.*np.pi)



    def plot_potential(self, x, displayed_levels, calculated_levels, phi = 0,
                                inductance_potential = False,
                                josephson_potential = False):
        """
        Plot the potential as weel as the eigen snergies associated.

        Parameters
        ----------
        x : np.array
            Position array in rad.
        displayed_levels : int
            Number of eigen energies displayed.
        calculated_levels : int
            Matrix dimension used for the calculation.
        phi : float, optional
            Reduced flux Φ_ext/Φ_0, default 0.
        inductance_potential : bool, optional
            Plot the kinetic part of the potential
        josephson_potential : bool, optional
            Plot the Josephson part of the potential

        Returns
        ----------
            Matplotlib figure
        """

        fig, ax = plt.subplots(1, 1)
        plt.subplots_adjust(bottom=0.2)

        # List containing all the lines we will update
        line = []

        # We get the minimum value of the full potential
        offset = (self.full_potential(x, phi) + self.Ej).min()

        if inductance_potential:

            # Plot the kinetic part of the potential
            ax.plot(x, self.harmonic_potential(x),
                   'b',
                   label='Inductance potential',
                   alpha=0.25)

        if josephson_potential:

            # Plot the Josephson part of the potential
            line.append(ax.plot(x, self.josephson_potential(x, phi) + self.Ej,
                                'b',
                                label='Josephson potential',
                                alpha=0.25)[0])

        # Plot the full part of the potential
        line.append(ax.plot(x, self.full_potential(x, phi) + self.Ej,
                            'b',
                            label='Full potential',
                            )[0])

        # Plot the eigenenergies wanted by the user
        y = self.eigen_energies(calculated_levels, phi)[:displayed_levels]

        # Only the last eigenergy line will have label
        label = None
        for i, j in enumerate(y):

            if i + 1 == len(y):

                label = 'Eigenenergies'

            line.append(ax.plot([x[0], x[-1]], np.array([j, j]) + offset,
                                '--',
                                color=plt.cm.hsv(float(i)/len(y)),
                                label=label)[0])

        # Display stuffs
        ax.set_xlabel('phi [rad]')
        ax.set_ylabel('Frequency [GHz]')

        ax.legend(loc='best').draggable()

        ax.set_title('E_J = '+str(round(self.Ej, 2))+' GHz, '\
                     'E_c = '+str(round(self.Ec, 2))+' GHz, '\
                     'E_l = '+str(round(self.El, 2))+' GHz')

        def update(phi):
            """
                Update the lines of the plot
            """
            # We get the minimum value of the full potential
            offset = (self.full_potential(x, phi) + self.Ej).min()

            # We update the potential
            a = 0
            if josephson_potential:
                line[0].set_ydata(self.josephson_potential(x, phi) + self.Ej)
                a += 1

            line[a].set_ydata(self.full_potential(x, phi) + self.Ej)
            a += 1

            # We update the eigenenergies
            y = self.eigen_energies(calculated_levels, phi)[:displayed_levels]

            for l, j in zip(line[a:], y):
                l.set_ydata(np.array([j, j])+offset)

            # Draw the new lines
            fig.canvas.draw_idle()

        # We make the axis for the slider
        slider_axis = plt.axes([0.125, 0.05, 0.775, 0.03])
        slider = Slider(slider_axis, 'Reduced\nflux',
                        -0.5, 0.5, valinit=phi)

        # We connect the slider to the update function
        slider.on_changed(update)

        # We show everything
        plt.show()



    def plot_spectrum(self, phi_start, phi_stop, phi_step,
                            displayed_levels, calculated_levels,
                            method='perturbation'):
        """
        Plot the energy spectrum as a function of the magnetic flux.

        Parameters
        ----------
        phi_start : float
            First value for the reduced magnetic flux.
        phi_stop : float
            Final value for the reduced magnetic flux.
        phi_step : float
            Step value for the reduced magnetic flux.
        displayed_levels : int
            Number of eigen energies displayed.
        calculated_levels : int
            Matrix dimension used for the calculation.

        Returns
        ----------
            Matplotlib figure
        """

        phi = np.arange(phi_start, phi_stop, phi_step)

        y = np.array([])
        for i in phi:
            y = np.concatenate((y, self.eigen_energies(calculated_levels, i,
                                                       method=method)))

        y = np.transpose(np.reshape(y, (len(phi), calculated_levels)))

        fig, ax = plt.subplots(1, 1)

        for i in y[:displayed_levels]:

            ax.plot(phi, i)

        # Display stuffs
        ax.set_xlabel('Reduced flux [dimensionless]')
        ax.set_ylabel('Frequency [GHz]')

        ax.set_title('E_J = '+str(round(self.Ej, 2))+' GHz, '\
                     'E_c = '+str(round(self.Ec, 2))+' GHz, '\
                     'E_L = '+str(round(self.El, 2))+' GHz')

        plt.show()



    def plot_probability_density(self, number_func_displayed, matrix_dimension,
                                       phi, method='perturbation'):
        """
        Plot the energy spectrum as a function of the magnetic flux.

        Parameters
        ----------
        number_func_displayed : int
            Number of probalbility density
               function displayed.
        matrix_dimension : int
            Matrix dimension used for the calculation.
        phi [0] : float
            Reduced flux Φ_ext/Φ_0.
        method {'perturbation', 'exponential'} : string
            Method used to get eigenenergies.

        Returns
        ----------
            Matplotlib figure
        """

        n = matrix_dimension

        # Get system eigenvectors
        eigenvectors =  self.eigen_states(n, phi, method=method)


        # create the position operator
        c_p = np.zeros([n + 1, n + 1])
        c_m = np.zeros([n + 1, n + 1])

        for row in range(n+1):

            for column in range(n+1):
                if row == column - 1:
                    c_p[row, column] = np.sqrt(row+1.)
                elif row == column + 1:
                    c_m[row, column] = np.sqrt(row)

        c_p = np.matrix(c_p)
        c_m = np.matrix(c_m)

        x = c_p + c_m

        # Calculate the eigenvector of the position operator
        x_values, x_vectors = np.linalg.eigh(x)

        # Sort its eigenvalues and eigenvectors
        idx = x_values.argsort()
        x_values  = x_values[idx]
        x_vectors = x_vectors[:,idx]
        x_vectors = np.transpose(x_vectors)

        # Calculate the probability density as |<x|psi>|^2
        proba_dens = np.array([])
        for eigenvector in eigenvectors:
            for x_vector in x_vectors:
                proba_dens = np.concatenate((proba_dens,
                             np.frombuffer(x_vector*np.transpose(eigenvector))**2.))

        proba_dens = np.reshape(proba_dens, (n+1, n+1))

        fig, ax = plt.subplots(1, 1)

        for curve in proba_dens[:number_func_displayed]:
            ax.plot(x_values, curve, '.-')

        ax.set_xlim(-10, 10)

        # Display stuffs
        ax.set_xlabel('Position [rad]')
        ax.set_ylabel('Probability density [normalised]')

        ax.set_title('E_J = '+str(round(self.Ej, 2))+' GHz, '\
                     'E_c = '+str(round(self.Ec, 2))+' GHz, '\
                     'E_L = '+str(round(self.El, 2))+' GHz')

        plt.show()

# Copyright 2023 Franco Aquistapace
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# ----- Here we'll define the functions related to our metric --------

# Import modules
import numpy as np
from tools import *


# ----- General utility functions -----------------

# Define function to get cubic domain that contains
# two configurations
def get_cubic_domain(C1, C2, step_size, margin_size):
    '''
    Params:
        C1, C2 : numpy array
            Numpy array containing atomic configurations
            C1 and C2, respectively.
        step_size : int or float
            Size in A of the side of a cubic volume, which
            is used to discretise the cubic domain.
        margin_size : int or float
            Size in A of the margins added in each direction
            to the cubic integration domain.
        
    Output:
        Returns a cubic space that contains both C1 and C2, 
        discretised as cubes of side step_size and represented
        by their center vector. This space is extended in each 
        dimension an amount 2 * margin_size. 
    '''
    # Build joint configuration, to help get cubic domain
    C_joint = np.concatenate([C1,C2])
    # Determine max and min values of the cubic domain
    x_max, y_max, z_max = np.max(C_joint, axis=0) + margin_size
    x_min, y_min, z_min = np.min(C_joint, axis=0) - margin_size
    
    # Build domain as a set of points
    n1 = int((x_max - x_min)/step_size) + 1
    n2 = int((y_max - y_min)/step_size) + 1
    n3 = int((z_max - z_min)/step_size) + 1
    dom = gen_sc_lattice(n1, n2, n3, step_size)
    # Displace domain to its correct origin
    dom = displace(dom, np.array([x_min, y_min, z_min]))

    return dom



# ----- Vector representation functions -----------

# Define multivariate Gaussian function
def multi_Gaussian(X, X_0, sigma):
    '''
    Params:
        X : numpy array
            One-dimensional array of length 3, where each
            element represents the x, y and z component of a 
            position vector, respectively. This array acts as
            the variable of the function.
        X_0 : numpy array
            One-dimensional array of length 3, where each
            element represents the x, y and z component of a 
            position vector, respectively. This array acts as
            the mean if the Gaussian density function.
        sigma : float or int
            Value for the standard deviation of the Gaussian
            density function.

    Output:
        Returns the value of f(X) at the given point X. The 
        function f(X) is a multivariate Gaussian density, 
        defined as:
            f(X) = sqrt(2*pi*sigma)^(-3) * exp(-|X-X_0|^2/(2*sigma))
    '''
    A = np.power(np.sqrt(2*np.pi*sigma),-3)
    B = -1 * np.power(np.linalg.norm(X-X_0),2) / (2 * sigma)
    return A * np.exp(B)

# Define measure representation of a configuration C from 
# different sources
def measure_representation(X, C, scale=1):
    '''
    Params:
        X : numpy array
            One-dimensional array of length 3, where each
            element represents the x, y and z component of a 
            position vector, respectively. This array acts as
            the variable of the function.
        C : numpy array
            Array containing atomic positions in 3 dimensions.
            In every case, C must be of shape (N, 3), where N 
            is the number of atoms in the configuration.
        scale : float or int (optional)
            Value used to transform each position vector X as:
                X -> scale * X
            Default value is 1.

    Output:
        Returns the value of the measure representation at the 
        given point X. The measure representation C_r is built as:
            C_r(X) = (1 / sqrt(|C|)) * sum_X_0(f(X,X_0))
        Where |C| is the amount of atoms in C, sum_X_0 is a sum over 
        all of the positions X_0 contained in C, and f(X,X_0) is 
        defined as:
            f(X, X_0) = multi_Gaussian(X, X_0, sigma=(4 * pi)^(-1))
        This is a multivariate Gaussian density function with 
        standard deviation std=(4 * pi)^(-1) and mean X_0. 
    '''
    scale = 1 / np.sqrt(C.shape[0])
    sigma = 1 / (4 * np.pi)
    X_0_list = [C[i,:] for i in range(C.shape[0])]
    C_rep = scale * np.sum(
        [multi_Gaussian(X, X_0, sigma) for X_0 in X_0_list])
    return C_rep


# Define dot product between two measures
def dot(C1, C2, step_size, margin_size):
    '''
    Params:
        C1, C2 : numpy array
            Numpy array containing atomic configurations
            C1 and C2, respectively.
        step_size : int or float
            Size in A of the side of a cubic volume, which
            is used to discretise the integration domain.
        margin_size : int or float
            Size in A of the margins added in each direction
            to the cubic integration domain.
        
    Output:
        Returns the dot product between C1 and C2, defined
        as:
            < C1, C2 > = int_IR^3(C1(X) * C2(X) * dX)
        Where the integral over IR^3 is approximated by a 
        sampling of the cubic space that contains both C1 
        and C2, discretised as cubes of side step_size. This 
        space is extended in each dimension an amount 
        2 * margin_size. 
        The cubic samples are used to evaluate the function
        C1(X) * C2(X), and these evaluations are summed to 
        approximate < C1, C2 >.
    '''
    # Build domain using helper function
    dom = get_cubic_domain(C1, C2, step_size, margin_size)
    # Define cubic samples volume in A^3
    cube_vol = step_size ** 3
    # Build multiplication of measure representations as 
    # a lambda expression
    C1_times_C2 = lambda X: measure_representation(X, C1) *\
                            measure_representation(X, C2)
    # Evaluate function applying to dom along 0th axis
    all_results = np.apply_along_axis(C1_times_C2, 1, dom)
    # Get result as the sum of all evaluations times the
    # sample volume
    result = np.sum(all_results) * cube_vol
    return result


# ----- Element-wise representation functions -----

def multi_Gaussian_exp(x, y, z, X_0, sigma):
    '''
    Params:
        x, y, z : float or int
            x, y and z variables of the function, respectively.
        X_0 : numpy array
            One-dimensional array of length 3, where each
            element represents the x, y and z component of a 
            position vector, respectively. This array acts as
            the mean if the Gaussian density function.
        sigma : float or int
            Value for the standard deviation of the Gaussian
            density function.

    Output:
        Returns the value of f(x,y,z) at the given point (x,y,z). The 
        function f(x,y,z) is an element-wise multivariate Gaussian 
        density, defined as:
            f(x,y,z) = sqrt(2*pi*sigma)^(-3) * exp(-|X-X_0|^2/(2*sigma))
        Where X is a vector (x,y,z).
    '''
    A = np.power(np.sqrt(2*np.pi*sigma),-3)
    X = np.array((x,y,z))
    B = -1 * np.power(np.linalg.norm(X-X_0),2) / (2 * sigma)
    return A * np.exp(B)


# Define measure representation of a configuration C from 
# different sources
def measure_representation_exp(x, y, z, C, scale=1):
    '''
    Params:
        x, y, z : float or int
            x, y and z variables of the function, respectively.
        C : numpy array
            Array containing atomic positions in 3 dimensions.
            In every case, C must be of shape (N, 3), where N 
            is the number of atoms in the configuration.
        scale : float or int (optional)
            Value used to transform each position vector X as:
                X -> scale * X
            Default value is 1.

    Output:
        Returns the value of the measure representation at the 
        given point X=(x,y,z). The measure representation C_r is 
        built as:
            C_r(X) = (1 / sqrt(|C|)) * sum_X_0(f(X,X_0))
        Where |C| is the amount of atoms in C, sum_X_0 is a sum over 
        all of the positions X_0 contained in C, and f(X,X_0) is 
        defined as:
            f(X, X_0) = multi_Gaussian(X, X_0, sigma=(4 * pi)^(-1))
        This is a multivariate Gaussian density function with 
        standard deviation std=(4 * pi)^(-1) and mean X_0. 
    '''
    scale = 1 / np.sqrt(C.shape[0])
    sigma = 1 / (4 * np.pi)
    X_0_list = [C[i,:] for i in range(C.shape[0])]
    C_rep = scale * np.sum(
        [multi_Gaussian_exp(x,y,z, X_0, sigma) for X_0 in X_0_list])
    return C_rep


def dot_exp(C1, C2, margin_size, tol):
    '''
    Params:
        C1, C2 : numpy array
            Numpy array containing atomic configurations
            C1 and C2, respectively.
        margin_size : int or float
            Size in A of the margins added in each direction
            to the cubic integration domain.
        tol : float
            Relative and absolute tolerance value to pass to 
            the tplquad integrator.
        
    Output:
        Returns the dot product between C1 and C2, defined
        as:
            < C1, C2 > = int_IR^3(C1(X) * C2(X) * dX)
        Where the integral over IR^3 is approximated by a 
        scipy integration of the space that contains both C1 
        and C2. This space is extended in each dimension an 
        amount 2 * margin_size. 
    '''
    from scipy import integrate
    # Build joint configuration, to help get cubic domain
    C_joint = np.concatenate([C1,C2])
    # Determine max and min values of the cubic domain
    x_max, y_max, z_max = np.max(C_joint, axis=0) + margin_size
    x_min, y_min, z_min = np.min(C_joint, axis=0) - margin_size
    # Build multiplication of measure representations as 
    # a lambda expression
    C1_times_C2 = lambda x,y,z: measure_representation_exp(x,y,z, C1) *\
                                measure_representation_exp(x,y,z, C2)
    # Integrate function with scipy's tplquad
    result = integrate.tplquad(C1_times_C2, x_min, x_max, 
                                            y_min, y_max, 
                                            z_min, z_max,
                                epsabs=tol, epsrel=tol)
    return result[0]
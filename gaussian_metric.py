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
from scipy import integrate

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
    X = np.array((x,y,z))
    C_rep = scale * np.sum(
        [multi_Gaussian(X, X_0, sigma) for X_0 in X_0_list])
    return C_rep
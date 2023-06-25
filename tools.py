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

# ----- Here we'll define some tools to help us in development and testing ------
# We'll use a functional programming approach for now, we can develop classes 
# later if we need them.

# Import modules
import numpy as np


# ------ Lattice generation functions ------------

# Function that can generate a cubic structure
def gen_sc_lattice(n1, n2, n3, a):
    '''
    Params:
        n1, n2, n3 : int
            Size of the lattice in the x, y and z
            directions, respectively, in terms of
            lattice unit vectors. 
        a : float or int
            Lattice constant in A for the lattice.
    
    Output:
        Returns a numpy array of shape (N,3), where
        N is the amount of atoms in the structure, 
        and each row is the position of an atom as
        (x, y, z). The array contains the positions
        for a simple cubic (SC) lattice, with unit 
        vectors x_hat, y_hat, z_hat, and dimensions
        (n1*a*x_hat) x (n2*a*y_hat) x (n3*a*z_hat).
    '''
    # Initialize empty line of points
    config = []
    # Append points to line
    for i in range(n1):
        config.append([a * i, 0, 0])
    config = np.array(config)
    # Initialize first line of plane
    new_conf = config.copy()
    # Append lines to plane
    for j in range(n2-1):
        new_vec = np.array([0, a * (j+1), 0])
        new_line = displace(config, new_vec)
        new_conf = np.concatenate([new_conf, new_line], 0)
    # Initialize first plane of space
    final_conf = new_conf.copy()
    # Append planes to space
    for k in range(n3-1):
        new_vec = np.array([0, 0, a * (k+1)])
        new_plane = displace(new_conf, new_vec)
        final_conf = np.concatenate([final_conf, new_plane], 0)
    return final_conf


# ------ Lattice transformation functions --------

# Define function to scale a lattice
def scale(C, s):
    '''
    Params:
        C : numpy array
            Array containing atomic positions in 3 dimensions.
            In every case, C must be of shape (N, 3), where N 
            is the number of atoms in the configuration.
        s : iterable
            Scaling factor to apply to the configuration C. Must 
            be of shape (,3), so that each element is the scaling
            factor for a given axis x, y or z, respectively.

    Output:
        Returns a new configuration C_new as a result of a scale
        transformation of C:
            C_new = C * Ixs
        Where Ixs is a 3x3 matrix where is a diagonal matrix in 
        which the i-th diagonal element equals the i-th element
        from s.
    '''
    Ixs = np.array([[s[0], 0,    0],
                    [0,    s[1], 0],
                    [0,    0,    s[2]]])
    return np.matmul(C, Ixs)

# Define a function to displace a lattice
def displace(C, X_0):
    '''
    Params:
        C : numpy array
            Array containing atomic positions in 3 dimensions.
            In every case, C must be of shape (N, 3), where N 
            is the number of atoms in the configuration.
        X_0 : numpy array
            Array containing the x, y and z values of a 
            position vector. This vector will be used to displace
            the configuration.
    
    Output:
        Returns a new configuraion C_new as a result of a 
        displacement of C:
            C_new = C + IxX_0
        Where IxX_0 is a matrix of same shape as C in which
        each row contains the vector X_0.
    '''
    return C + X_0

# Define a function to rotate a lattice
def rotate(C, alpha, beta, gamma):
    '''
    Params:
        C : numpy array
            Array containing atomic positions in 3 dimensions.
            In every case, C must be of shape (N, 3), where N 
            is the number of atoms in the configuration.
        alpha, beta, gamma : float
            x, y and z axes rotation angles, respectively. 
    
    Output:
        Returns a new configuraion C_new as a result of a 
        rotation of C:
            C_new = R x C
        Where R is a matrix of shape (3,3) defined as:
            R = R_z(gamma)*R_y(beta)*R_x(alpha)
        with R_x, R_y and R_z being rotations about the
        x, y and z axes, respectively.
    '''
    # Define each rotation
    c_alpha, s_alpha = np.cos(alpha), np.sin(alpha)
    R_x = np.array([[1, 0, 0],
                    [0, c_alpha, -s_alpha],
                    [0, s_alpha, c_alpha]])
    c_beta, s_beta = np.cos(beta), np.sin(beta)
    R_y = np.array([[c_beta, 0, s_beta],
                    [0, 1, 0],
                    [-s_beta, 0, c_beta]])
    c_gamma, s_gamma = np.cos(gamma), np.sin(gamma)
    R_z = np.array([[c_gamma, -s_gamma, 0],
                    [s_gamma, c_gamma, 0],
                    [0, 0, 1]])
    # Build general rotation
    R = np.matmul(R_z, np.matmul(R_y, R_x))
    # Calculate C_new
    C_new = np.transpose(np.matmul(R, C.transpose()))
    return C_new


# Define a function to flip a lattice
def flip(C, x, y, z):
    '''
    Params:
        C : numpy array
            Array containing atomic positions in 3 dimensions.
            In every case, C must be of shape (N, 3), where N 
            is the number of atoms in the configuration.
        x, y, z : bool
            Requested flip over each axis, respectively. True 
            means that the flip will be done for its respective
            axis.
    Output:
        Returns a new configuraion C_new as a result of a 
        flip of C:
            C_new = F x C
        Where F is a matrix of shape (3,3), defined as:
            F = [[+-1,  0,    0],
                 [0,    +-1,  0],
                 [0,    0,    +-1]]
        with the + or - value of each diagonal element assigned
        according to the bool value of the respective axis.
    '''
    # Define F
    F = np.array([[-2*int(x)+1, 0, 0],
              [0, -2*int(y)+1, 0],
              [0, 0, -2*int(z)+1]])
    # Get C_new
    C_new = np.transpose(np.matmul(F, C.transpose()))
    return C_new

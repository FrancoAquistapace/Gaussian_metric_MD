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
    config = []
    for k in range(n3):
        for j in range(n2):
            for i in range(n1):
                # Build position and add to config
                p = [a * i, 
                     a * j,
                     a * k]
                config.append(p)

    return np.array(config)
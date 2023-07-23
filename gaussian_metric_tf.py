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

# ----- Here we'll define the functions related to our metric, with TF --------

# Import modules
import tensorflow as tf
import math
from tools_tf import *

# ----- General utility functions -----------------

# Define function to get cubic domain that contains
# two configurations
def get_cubic_domain(C1, C2, step_size, margin_size):
    '''
    Params:
        C1, C2 : tf.Tensor
            Arrays containing atomic configurations
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
    C_joint = tf.concat([C1,C2], 0)
    # Determine max and min values of the cubic domain
    x_max, y_max, z_max = tf.reduce_max(C_joint, axis=0).numpy() + margin_size
    x_min, y_min, z_min = tf.reduce_min(C_joint, axis=0).numpy() - margin_size
    
    # Build domain as a set of points
    n1 = int((x_max - x_min)/step_size) + 1
    n2 = int((y_max - y_min)/step_size) + 1
    n3 = int((z_max - z_min)/step_size) + 1
    dom = gen_sc_lattice(n1, n2, n3, step_size)
    # Displace domain to its correct origin
    dom = displace(dom, tf.constant([x_min, y_min, z_min]))

    return dom


# Define function to generate a cubic domain that contains a 
# batch of configurations C
def get_cubic_domain_from_dataset(C, step_size, margin_size):
    '''
    Params:
        C : tf.Tensor
            Array containing batches of atomic configurations. 
            The tensor is expected to have shape (M,N,3), 
            with M being the number of configurations and N 
            being the number of atoms per configuration.
        step_size : int or float
            Size in A of the side of a cubic volume, which
            is used to discretise the cubic domain.
        margin_size : int or float
            Size in A of the margins added in each direction
            to the cubic integration domain.
        
    Output:
        Returns a cubic space that contains every configuration
        in C, discretised as cubes of side step_size and 
        represented by their center vector. This space is extended 
        in each dimension an amount 2 * margin_size. This function 
        is built to work with batches of data.
    '''
    # Determine max and min values of the cubic domain
    xyz_max = tf.reduce_max(tf.reduce_max(C, axis=0), 
                                        axis=0) + margin_size
    xyz_min = tf.reduce_min(tf.reduce_min(C, axis=0), 
                                        axis=0) - margin_size
    x_max, y_max, z_max = tf.split(xyz_max, num_or_size_splits=3)
    x_min, y_min, z_min = tf.split(xyz_min, num_or_size_splits=3)

    # Build domain as a set of points
    n1, n2, n3 = tf.split((xyz_max - xyz_min)/step_size,
                           num_or_size_splits=3)
    n1 = int(n1) + 1
    n2 = int(n2) + 1
    n3 = int(n3) + 1
    dom = gen_sc_lattice(n1, n2, n3, step_size)
    # Displace domain to its correct origin
    dom = displace(dom, xyz_min)

    return dom


# ----- Vector representation functions -----------

# Define multivariate Gaussian function
def multi_Gaussian(X, X_0, sigma):
    '''
    Params:
        X : tf.Tensor
            One-dimensional array of length 3, where each
            element represents the x, y and z component of a 
            position vector, respectively. This array acts as
            the variable of the function.
        X_0 : tf.Tensor
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
    A = tf.pow(tf.sqrt(2 * math.pi * sigma),-3)
    B = -1 * tf.reduce_sum(tf.pow(X-X_0,2), -1) / (2 * sigma)
    return A * tf.exp(B)

# Define measure representation of a configuration C from 
# different sources
def measure_representation(X, C, scale=1):
    '''
    Params:
        X : tf.Tensor
            One-dimensional array of length 3, where each
            element represents the x, y and z component of a 
            position vector, respectively. This array acts as
            the variable of the function.
        C : tf.Tensor
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
    scale = 1 / tf.sqrt(float(C.shape[0]))
    sigma = 1 / (4 * math.pi)
    X_0_list = [C[i,:] for i in range(C.shape[0])]
    multi_gs = [multi_Gaussian(X, X_0, sigma).numpy() for X_0 in X_0_list]
    C_rep = scale * tf.reduce_sum(tf.constant(multi_gs, dtype='float32'), 0)
    return C_rep


# Define dot product between two measures
def dot(C1, C2, step_size, margin_size):
    '''
    Params:
        C1, C2 : tf.Tensor
            Arrays containing atomic configurations
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
    C1_times_C2 = measure_representation(dom, C1)*measure_representation(dom, C2)
    # Get result as the sum of all evaluations times the
    # sample volume
    return tf.reduce_sum(C1_times_C2) * cube_vol


# Define function to calculate M metric 
def M(C1, C2, step_size, margin_size):
    '''
    Params:
        C1, C2 : tf.Tensor
            Arrays containing atomic configurations
            C1 and C2, respectively.
        step_size : int or float
            Size in A of the side of a cubic volume, which
            is used to discretise the integration domain.
        margin_size : int or float
            Size in A of the margins added in each direction
            to the cubic integration domain.
        
    Output:
        Returns the M value between C1 and C2, defined
        as:
            M = 1 - min(< C1, C2 >, 1)
        Where the dot product < C1, C2 > is calculated with the
        vector implementation.
    '''
    dot_prod = dot(C1, C2, step_size, margin_size).numpy()
    return 1 - tf.reduce_min([dot_prod, 1])



# ----- Functions compatible with graph mode of TF -----------

# Define function that calculates dot product in the measure
# representation, that is compatible with TF graph mode.
def graph_dot(C1, C2, dom, V):
    '''
    Params:
        C1, C2 : tf.Tensor
            Arrays containing batches of atomic configurations.
            The tensors are expected to have shape (...,N,3),
            with M being the number of configurations and N
            being the number of atoms per configuration.
        dom : tf.Tensor
            Array of shape (N_prime, 3) containing positions
            used as evaluation points for the operation.
        V : float
            Volume element to use when approximating the
            integral operation.

    Output:
        Returns the dot product between C1 and C2, defined
        as:
            < C1, C2 > = int_IR^3(C1(X) * C2(X) * dX)
        Where the integral over IR^3 is approximated by a
        sampling of the cubic space that contains both C1
        and C2, discretised as cubes represented by the
        positions in dom.
        The cubic samples are used to evaluate the function
        C1(X) * C2(X), and these evaluations are summed and
        multiplied by V to approximate < C1, C2 >. This
        function can be used in the graph mode of TF.
    '''
    # Calculate lambdas
    lambda_mb = tf.einsum('ki,...ni->...kn', dom, C1)
    lambda_mk = tf.einsum('ki,...ni->...kn', dom, C2)

    # Get phis
    phi_m = tf.reduce_sum(tf.square(dom),axis=-1)
    phi_b = tf.reduce_sum(tf.square(C1), axis=-1)
    phi_k = tf.reduce_sum(tf.square(C2), axis=-1)

    # Get thetas
    broad_phi_m = tf.broadcast_to(tf.expand_dims(phi_m, -1), 
                                  shape=tf.shape(lambda_mb)[-2:])
    theta_mb = 2 * lambda_mb - broad_phi_m
    broad_phi_m = tf.broadcast_to(tf.expand_dims(phi_m, -1), 
                                  shape=tf.shape(lambda_mk)[-2:])
    theta_mk = 2 * lambda_mk - broad_phi_m
    # Add remaining phis to thetas
    broad_phi = tf.broadcast_to(tf.expand_dims(phi_b, -2), 
                                  shape=tf.shape(theta_mb))
    theta_mb = broad_phi - 1 * theta_mb
    broad_phi = tf.broadcast_to(tf.expand_dims(phi_k, -2), 
                                  shape=tf.shape(theta_mk))
    theta_mk = broad_phi - 1 * theta_mk
    # Turn into exponential thetas and reduce_sum over b,k axes
    theta_mb = tf.reduce_sum(tf.exp(-2 * math.pi * theta_mb), 
                             axis=-1)
    theta_mk = tf.reduce_sum(tf.exp(-2 * math.pi * theta_mk),
                             axis=-1)
    # Get final sum
    prod_m = tf.einsum('...m,...m->...', theta_mk, theta_mb)
    # Get final result
    result = 8 * V * prod_m / tf.sqrt(float(
                              tf.shape(C1)[-2]*tf.shape(C2)[-2]))
    return result


# Define function that calculates the M metric, that is 
# compatible with TF graph mode
def graph_M(C1, C2, dom, V):
    '''
    Params:
        C1, C2 : tf.Tensor
            Arrays containing batches of atomic configurations.
            The tensors are expected to have shape (...,N,3),
            with M being the number of configurations and N
            being the number of atoms per configuration.
        dom : tf.Tensor
            Array of shape (N_prime, 3) containing positions
            used as evaluation points for the operation.
        V : float
            Volume element to use when approximating the
            integral operation.
        
    Output:
        Returns the M value between C1 and C2, defined
        as:
            M = 1 - min(< C1, C2 >, 1)
        Where the dot product < C1, C2 > is calculated with the
        vector implementation.
    '''
    # Get dot product
    dot_prod = graph_dot(C1, C2, dom, V)
    # Get clipped values
    clipped_dot = tf.clip_by_value(
                    dot_prod, 0, 1)
    # Return result
    return -1 * clipped_dot + 1


# Define Keras loss function that calculates the M metric
# given y_true and y_pred configurations
class MLoss(tf.keras.losses.Loss):
  def __init__(self, dom, V):
    super(MLoss, self).__init__()
    self.dom = dom
    self.V = V

  def call(self, y_true, y_pred): # Defines the computation
    loss = graph_M(y_true, y_pred, self.dom, self.V)
    return loss
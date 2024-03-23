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
from data_tf import *

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
def graph_dot(C1, C2, dom, V, equal_size=True,
              sigma=1 / (4 * math.pi)):
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
        equal_size : bool (optional)
            If True (default), it is assumed that C1 and C2 
            have the same shapes and sizes. If False, this
            is not assumed.
        sigma : float (optional)
            Sigma coefficient for the Gaussian functions. 
            Default value is 1 / (4 * math.pi).

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
    theta_mb = tf.reduce_sum(tf.exp((-1 / (2 * sigma)) * theta_mb), 
                             axis=-1)
    theta_mk = tf.reduce_sum(tf.exp((-1 / (2 * sigma)) * theta_mk),
                             axis=-1)
    # Check equal_size condition
    if not equal_size:
        theta_mb = tf.squeeze(theta_mb, axis=-1)
        theta_mk = tf.squeeze(theta_mk, axis=-1)
    # Get final sum
    prod_m = tf.einsum('...m,...m->...', theta_mk, theta_mb)
    # Get final result
    A = tf.pow(2 * math.pi * sigma,-3)
    result = A * V * prod_m / tf.sqrt(float(
                              tf.shape(C1)[-2]*tf.shape(C2)[-2]))
    return result


# Define function that computes the exponential component of 
# the dot product for a given batch of configurations C
def graph_exp_component(C, dom, 
                        sigma=1 / (4 * math.pi)):
    '''
    Params:
        C : tf.Tensor
            Array containing batches of atomic configurations.
            The tensors are expected to have shape (...,N,3),
            with a certain number of configurations and N
            being the number of atoms per configuration.
        dom : tf.Tensor
            Array of shape (N_prime, 3) containing positions
            used as evaluation points for the operation.
        sigma : float (optional)
            Sigma coefficient for the Gaussian functions. 
            Default value is 1 / (4 * math.pi).
    Output:
        Returns a tensor of shape (..., m), with m being the
        number of sampling points in dom. This tensor contains
        the results of the exponential component of the 
        measure representation dot product, for a batch of 
        configurations. The first dimensions of the resulting 
        tensor are given by the first dimensions of the C
        tensor.
    '''
    # Calculate lambda
    lambda_mb = tf.einsum('ki,...ni->...kn', dom, C)
    # Get phis
    phi_m = tf.reduce_sum(tf.square(dom),axis=-1)
    phi_b = tf.reduce_sum(tf.square(C), axis=-1)
    # Get theta
    broad_phi_m = tf.broadcast_to(tf.expand_dims(phi_m, -1), 
                                  shape=tf.shape(lambda_mb)[-2:])
    theta_mb = 2 * lambda_mb - broad_phi_m
    # Add remaining phi to theta
    broad_phi = tf.broadcast_to(tf.expand_dims(phi_b, -2), 
                                  shape=tf.shape(theta_mb))
    theta_mb = broad_phi - 1 * theta_mb
    # Turn into exponential theta and reduce_sum over b axis
    theta_mb = tf.reduce_sum(tf.exp((-1 / (2 * sigma)) * theta_mb), 
                             axis=-1)
    return theta_mb


# Define function that computes the dot product between C1
# and C2, from their exponential theta_mb representations
def graph_dot_from_thetas(C1, C2, theta_mb, theta_mk, V,
                          sigma=1 / (4 * math.pi)):
    '''
    Params:
        C1, C2 : tf.Tensor
            Arrays containing batches of atomic configurations.
            The tensors are expected to have shape (...,N,3),
            with a certain number of configurations and N
            being the number of atoms per configuration.
        theta_mb, theta_mk : tf.Tensor
            Tensors containing the exponential representations
            of C1 and C2 respectively. They can be obtained 
            through the graph_exp_component function.
        V : float
            Volume element to use when approximating the
            integral operation.
        sigma : float (optional)
            Sigma coefficient for the Gaussian functions. 
            Default value is 1 / (4 * math.pi).
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
        function can be used in the graph mode of TF. It 
        returns the same output as graph_dot, but allows
        for the exponential representations to be recycled. 
    '''
    # Get sum
    prod_m = tf.einsum('...m,...m->...', theta_mk, theta_mb)
    # Get final result
    A = tf.pow(2 * math.pi * sigma,-3)
    result = A * V * prod_m / tf.sqrt(float(
                              tf.shape(C1)[-2]*tf.shape(C2)[-2]))
    return result


# Define function that calculates the M metric, that is 
# compatible with TF graph mode
def graph_M(C1, C2, dom, V, equal_size=True,
            sigma=1 / (4 * math.pi)):
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
        equal_size : bool (optional)
            If True (default), it is assumed that C1 and C2 
            have the same shapes and sizes. If False, this
            is not assumed.
        sigma : float (optional)
            Sigma coefficient for the Gaussian functions. 
            Default value is 1 / (4 * math.pi).
        
    Output:
        Returns the M value between C1 and C2, defined
        as:
            M = 1 - min(< C1, C2 >, 1)
        Where the dot product < C1, C2 > is calculated with the
        graph implementation.
    '''
    # Get dot product
    dot_prod = graph_dot(C1, C2, dom, V, equal_size=equal_size, 
                         sigma=sigma)
    # Get clipped values
    clipped_dot = tf.clip_by_value(
                    dot_prod, 0, 1)
    # Return result
    return -1 * clipped_dot + 1


# Define function that calculates the M metric from
# the exponential representations of the configurations
def graph_M_from_thetas(C1, C2, theta_mb, theta_mk, V,
                          sigma=1 / (4 * math.pi)):
    '''
    Params:
        C1, C2 : tf.Tensor
            Arrays containing batches of atomic configurations.
            The tensors are expected to have shape (...,N,3),
            with a certain number of configurations and N
            being the number of atoms per configuration.
        theta_mb, theta_mk : tf.Tensor
            Tensors containing the exponential representations
            of C1 and C2 respectively. They can be obtained 
            through the graph_exp_component function.
        V : float
            Volume element to use when approximating the
            integral operation.
        sigma : float (optional)
            Sigma coefficient for the Gaussian functions. 
            Default value is 1 / (4 * math.pi).
     Output:
        Returns the M value between C1 and C2, defined
        as:
            M = 1 - min(< C1, C2 >, 1)
        Where the dot product < C1, C2 > is calculated with the
        graph implementation. This gives the same output as the
        graph_M function, but allows for the exponential 
        representations to be recycled.
    '''
    # Get dot product
    dot_prod = graph_dot_from_thetas(C1, C2, theta_mb, theta_mk, V,
                                     sigma=sigma)
    # Get clipped values
    clipped_dot = tf.clip_by_value(
                    dot_prod, 0, 1)
    # Return result
    return -1 * clipped_dot + 1


# ----- Keras Loss and Layer Sub-classes -----------

# Define Keras loss function that calculates the M metric
# given y_true and y_pred configurations
class MLoss(tf.keras.losses.Loss):
    def __init__(self, dom, V, equal_size=True, sigma=1 / (4 * math.pi)):
        super(MLoss, self).__init__()
        self.dom = dom
        self.V = V
        self.equal_size = equal_size
        self.sigma = sigma

    def call(self, y_true, y_pred): # Defines the computation
        loss = graph_M(y_true, y_pred, self.dom, self.V,
                        equal_size=self.equal_size, 
                        sigma=self.sigma)
        return loss


# Define a Keras layer that computes the M metric
class M_layer(tf.keras.layers.Layer):
    def __init__(self, C, dom, V, sigma=1 / (4 * math.pi)):
        super(M_layer, self).__init__()
        self.dom = dom
        self.V = V
        self.C = C # Reference structure
        self.sigma = sigma
    # Defines the computation from inputs to outputs
    def call(self, C1):
        result = graph_M(C1, self.C, self.dom, self.V, 
                        sigma=self.sigma)
        return result


# Define a Keras layer that computes the graph_dot 
class Dot_layer(tf.keras.layers.Layer):
    def __init__(self, C, dom, V, sigma=1 / (4 * math.pi)):
        super(Dot_layer, self).__init__()
        self.dom = dom
        self.V = V
        self.C = C # Reference structure
        self.sigma = sigma
    # Defines the computation from inputs to outputs
    def call(self, C1):
        result = graph_dot(C1, self.C, self.dom, self.V, 
                        sigma=self.sigma)
        return result


# Define a Keras layer that computes the exponential 
# expression
class Theta_layer(tf.keras.layers.Layer):
    def __init__(self, dom, sigma=1 / (4 * math.pi)):
        super(Theta_layer, self).__init__()
        self.dom = dom
        self.sigma = sigma
    # Define computation from inputs to outputs
    def call(self, C1):
        result = graph_exp_component(C1, self. dom,
                                    sigma=self.sigma)
        return result


# Define a keras layer that computes the 
# graph_dot_from_thetas function
class ThetasDot_layer(tf.keras.layers.Layer):
    def __init__(self, C, dom, V, sigma=1 / (4 * math.pi)):
        super(ThetasDot_layer, self).__init__()
        self.dom = dom
        self.V = V
        self.C = C # Reference structure
        self.theta_mk = graph_exp_component(C, dom, 
                                            sigma=sigma)
        self.sigma = sigma
    # Defines the computation from inputs to outputs
    def call(self, C1, theta_mb):
        result = graph_dot_from_thetas(C1, self.C, 
                            theta_mb, self.theta_mk, 
                            self.V, sigma=self.sigma)
        return result


# Define a keras layer that computes the M metric 
# from the exponential representations
class ThetasM_layer(tf.keras.layers.Layer):
    def __init__(self, C, dom, V, sigma=1 / (4 * math.pi)):
        super(ThetasM_layer, self).__init__()
        self.dom = dom
        self.V = V
        self.C = C # Reference structure
        self.theta_mk = graph_exp_component(C, dom, 
                                            sigma=sigma)
        self.sigma = sigma
    # Defines the computation from inputs to outputs
    def call(self, C1, theta_mb):
        result = graph_M_from_thetas(C1, self.C, 
                            theta_mb, self.theta_mk, 
                            self.V, sigma=self.sigma)
        return result



# ----- Complete computation pipelines -----------

# Define a pipeline that can compute the M metric
# for multiple classes, and assign the best class
# for each atom accordingly
def M_pipeline(input_files, config_file, N, config_N, 
                templates,
                sigma=1 / (4 * math.pi),
                step_size=0.3, margin_size=1, 
                sample_size=30000, 
                batch_size=512, prefetch=True,
                verbose=1, save_templates=False,
                output_prefix=None):
    '''
    Params:
        input_files : list 
            List containing the path of the files to be 
            analysed.
        config_file : str
            Path of the file from which the template 
            configurations will be fetched.
        N : int
            Number of neighbors to use when analysing the 
            files.
        config_N : int
            Number of neighbors to use when building the 
            template configurations.
        templates : dict
            Dictionary containing the output classes as 
            keys, with the element of each key being a 
            list of atomic identifiers from which to 
            build the template configurations of each 
            class. For example, a classification between
            surface and bulk would look something like 
            this:
                {"surface" : [1248, 94371],
                 "bulk" : [4372, 23405, 4920]}
            The word "class" is reserved and cannot be 
            used in templates. Words that may conflict
            with particle properties already present in the
            files should not be used either.
        sigma : float (optional)
            Sigma coefficient for the Gaussian functions. 
            Default value is 1 / (4 * math.pi).
        step_size : float (optional)
            Size in A of the side of a cubic volume, which
            is used to discretise the cubic domain. Default
            is 0.3.
        margin_size : int or float (optional)
            Size in A of the margins added in each direction
            to the cubic integration domain. Default is 1.
        sample_size : int (optional)
            Amount of atomic configurations from each file
            to use when defining the cubic domain. Default 
            is 30000.
        batch_size : int (optional)
            Batch size to use when transforming the 
            configurations into a Dataset. Default is 512.
        prefetch : bool (optional)
            Whether to use prefetch or not for each Dataset.
            Default is True.
        verbose : int (optional)
            Option to print process information on screen. 
            If 1, time elapsed is printed (default). If 2,
            all relevant information of the process is 
            printed. If 0, nothing is printed.
        save_templates : bool (optional)
            Whether to save dump files of the selected 
            templates or not. Each template is saved as 
            "temp_class_i.config", where "class" is replaced
            with the class name, and "i" is replaced with the
            template number of that class. Default is False.
        output_prefix : str (optional)
            Prefix to use for the output paths. If None 
            (default), then the outputs are saved to the 
            original input file paths.
    Output:
        Calculates and writes the M metric score between a set
        of atomic configurations and a set of template 
        structures, for each of the input files. Additionally, 
        a property "class" is saved for each atom, that contains
        the index of the class for which the M metric was the 
        lowest (i.e. most similarity).
    '''
    # Start timing if verbose is 1 or 2
    if verbose in [1,2]:
        import time
        time_1 = time.time()

    # Read the configuration file
    if verbose == 2:
        print('Gathering templates')
    configs_n = neighbors_from_file(config_file, config_N, 
                                    deltas=True)
    
    # Get configurations
    config_ids = ids_from_file(config_file)
    # Get classes
    classes = [key for key in templates]

    # Initialize dictionary with configurations
    config_dict = {}
    for c in classes:
        # Get configurations for the class
        configs = [get_config_from_id(configs_n, 
                    config_ids, i) for i in templates[c]]
        # If requested, save template configurations
        if save_templates:
            config_num = 1
            for conf in configs:
                config_name = 'temp_' + c + str(config_num) +\
                              '.config'
                write_config(conf, config_name)
                config_num += 1

        # Expand dims for each config
        for i in range(len(configs)):
            new_config = tf.expand_dims(configs[i], axis=0)
            configs[i] = new_config

        # Add configs to dictionary
        config_dict[c] = configs

    # Loop through all of the files
    k = 1
    d = len(input_files)
    for path in input_files:
        # Read df from file
        if verbose == 2:
            print('\nAnalysing file %d from %d.' % (k, d))
        df, header = df_from_file(path)

        # Gather neighbors
        if verbose == 2:
            print('Gathering neighbors')
        neighbors = neighbors_from_file(path, N, deltas=True)

        # Turn neighbors into dataset
        neighbors = tf.constant(neighbors, dtype='float32')
        neighbors = tf.data.Dataset.from_tensor_slices(neighbors)

        # Build dom and define V
        if verbose == 2:
            print('Building integration domain')
        V = step_size ** 3
        dom = get_cubic_domain_from_dataset(
                    next(iter(neighbors.batch(sample_size))),
                    step_size=step_size,
                    margin_size=margin_size)

        # Batch dataset
        neighbors = neighbors.batch(batch_size)
        # Use prefetch if requested
        if prefetch:
            neighbors = neighbors.prefetch(tf.data.AUTOTUNE)

        # Build the model that computes the M metric
        if verbose == 2:
            print('Building M model')
        # 1. Setup input
        input_C1 = tf.keras.layers.Input(shape=[N,3], dtype=tf.float32,
                                        name='C1_input')
        # 2. Add M layers
        M_layers = {}
        for c in classes:
            M_layers[c] = [M_layer(
                        C, dom, V
                        )(input_C1) for C in config_dict[c]]
        # 3. Add class layers
        class_layers = [
            tf.reduce_min(
                tf.stack(M_layers[c], axis=0), 
            axis=0) for c in classes]

        # 4. Defining model
        model = tf.keras.models.Model(inputs=[input_C1],
                                      outputs=class_layers,
                                      name='M_model')

        # 5. Compile model
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[])
            
        # Predict results
        predict_verbose = 0
        if verbose == 2:
            print('Computing similarity metrics')
            predict_verbose = 2
        predictions = model.predict(neighbors, 
                                    verbose=predict_verbose)

        # Clean df from any previous results
        need_replace = False
        for c in classes:
            if c in list(df.columns):
                df.drop(columns=[c], inplace=True)
                need_replace = True
        if 'class' in list(df.columns):
            df.drop(columns=['class'], inplace=True)
            need_replace = True
        # Add predictions to df
        for i in range(len(classes)):
            df[classes[i]] = predictions[i]
        # Add class predictions
        df['class'] = tf.argmin(tf.constant(predictions), axis=0)

        # Check if an output prefix was given
        if not output_prefix == None and '/' in path:
            # Split path by folder bars
            split_path = path.split('/')
            # Add prefix to final location
            split_path[-1] = output_prefix + split_path[-1]
            # Concatenate path
            path = '/'.join(split_path)
        elif not output_prefix == None and not '/' in path:
            # If there are no folder bars 
            # just add the prefix
            path = output_prefix + path

        # Write new file
        if verbose == 2:
            print('Writing results')
        write_dump_from_df(df, header, path, 
                            new_cols=[*classes,'class'],
                            replace_cols=need_replace)

        # Add to counter
        k += 1

    # If verbose is 1 or 2, finish timing and output total time
    if verbose in [1,2]:
        time_2 = time.time()
        minutes_final = (time_2 - time_1)//60
        seconds_final = (time_2 - time_1)%60
        print('\nProcess completed')
        print('Elapsed time: %d minutes and %.f seconds' % \
            (minutes_final,seconds_final)) 


# ----- Additional functionallities --------------
# Function to get Hamming distance between a set of
# configurations and a reference
def Hamming_compare(C, C1, N, normalized=False):
    '''
    Params:
        C : tf.Tensor
            Array containing a single reference configuration.
            The tensor is expected to have shape (N), with N
            being the number of atoms in the reference 
            configuration.
        C1 : tf.Tensor
            Array containing batches of particle types
            configurations. The tensor is expected to have 
            shape (...,N), with a given number of 
            configurations and N being the number of atoms 
            per configuration.
        N : int
            Normalization constant for the Hamming distance.
            Must be equal to the amount of atoms in C and in
            each configuration in C1.
        normalized : bool (optional)
            Whether to return normalized results or not, default
            is False.
    
    Output:
        Returns the normalized Hamming distance between each 
        configuration in C1 and the reference configuration C. 
        The Hamming distance is obtained by counting the 
        amount of differences found when comparing two 
        configurations element-wise. Then, this result is 
        normalized by the total amount of atoms in a given
        configuration (length of the sequence), N.
    '''
    # Perform boolean comparison
    bool_compare = (C1 - C != 0)
    # Turn into integer representation and count differences
    res = tf.reduce_sum(tf.cast(bool_compare, dtype=tf.int32), axis=-1)
    # Normalize if requested
    if normalized:
        res = res / N
    return res

# Keras layer that computes the normalized Hamming distance
# between a predefined reference configuration and the 
# input data
class Hamming_layer(tf.keras.layers.Layer):
    def __init__(self, C, N, normalized=False):
        super(Hamming_layer, self).__init__()
        self.C = C # Reference configuration
        self.N = N # Number of atoms in C
        self.normalized = normalized
        
    # Defines the computation from inputs to outputs
    def call(self, C1):
        result = Hamming_compare(self.C, C1, self.N, 
                            normalized=self.normalized)
        return result


# Complete pipeline that performs a Hamming distance
# analysis for a set of configurations
def Hamming_pipeline(input_files, config_file, N, 
                    templates, normalized=False,
                    batch_size=512, prefetch=True,
                    verbose=1, output_prefix=None):
    '''
    Params:
        input_files : list 
            List containing the path of the files to be 
            analysed.
        config_file : str
            Path of the file from which the template 
            configurations will be fetched.
        N : int
            Number of neighbors to use when analysing the 
            files and building the template configurations.
        templates : dict
            Dictionary containing the output classes as 
            keys, with the element of each key being a 
            list of atomic identifiers from which to 
            build the template configurations of each 
            class. For example, a classification between
            surface and bulk would look something like 
            this:
                {"surface" : [1248, 94371],
                 "bulk" : [4372, 23405, 4920]}
            The word "class" is reserved and cannot be 
            used in templates. Words that may conflict
            with particle properties already present in the
            files should not be used either.
        batch_size : int (optional)
            Batch size to use when transforming the 
            configurations into a Dataset. Default is 512.
        prefetch : bool (optional)
            Whether to use prefetch or not for each Dataset.
            Default is True.
        verbose : int (optional)
            Option to print process information on screen. 
            If 1, time elapsed is printed (default). If 2,
            all relevant information of the process is 
            printed. If 0, nothing is printed.
        output_prefix : str (optional)
            Prefix to use for the output paths. If None 
            (default), then the outputs are saved to the 
            original input file paths.
    Output:
        Calculates and writes the Hamming score between a set
        of atomic configurations and a set of template 
        structures, for each of the input files. Additionally, 
        a property "class" is saved for each atom, that contains
        the index of the class for which the Hamming distance was the 
        lowest (i.e. most similarity).
    '''
    # Start timing if verbose is 1 or 2
    if verbose in [1,2]:
        import time
        time_1 = time.time()

    # Read the configuration file
    if verbose == 2:
        print('Gathering templates')
    configs_n = neighbor_types_from_file(config_file, N)
    
    # Get configurations
    config_ids = ids_from_file(config_file)
    # Get classes
    classes = [key for key in templates]

    # Initialize dictionary with configurations
    config_dict = {}
    for c in classes:
        # Get configurations for the class
        configs = [get_config_from_id(configs_n, 
                    config_ids, i, dtype='int32') for i in templates[c]]

        # Expand dims for each config
        for i in range(len(configs)):
            new_config = tf.expand_dims(configs[i], axis=0)
            configs[i] = new_config

        # Add configs to dictionary
        config_dict[c] = configs

    # Loop through all of the files
    k = 1
    d = len(input_files)
    for path in input_files:
        # Read df from file
        if verbose == 2:
            print('\nAnalysing file %d from %d.' % (k, d))
        df, header = df_from_file(path)

        # Gather neighbors
        if verbose == 2:
            print('Gathering neighbors')
        neighbors = neighbor_types_from_file(path, N)

        # Turn neighbors into dataset
        neighbors = tf.constant(neighbors, dtype='int32')
        neighbors = tf.data.Dataset.from_tensor_slices(neighbors)

        # Batch dataset
        neighbors = neighbors.batch(batch_size)
        # Use prefetch if requested
        if prefetch:
            neighbors = neighbors.prefetch(tf.data.AUTOTUNE)

        # Build the model that computes the Hamming score
        if verbose == 2:
            print('Building Hamming score model')
        # 1. Setup input
        input_C1 = tf.keras.layers.Input(shape=[N], dtype=tf.int32,
                                        name='C1_input')
        # 2. Add Hamming layers
        H_layers = {}
        for c in classes:
            H_layers[c] = [Hamming_layer(
                        C, N, normalized=normalized
                        )(input_C1) for C in config_dict[c]]
        # 3. Add class layers
        class_layers = [
            tf.reduce_min(
                tf.stack(H_layers[c], axis=0), 
            axis=0) for c in classes]

        # 4. Defining model
        model = tf.keras.models.Model(inputs=[input_C1],
                                      outputs=class_layers,
                                      name='H_model')

        # 5. Compile model
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[])
        
        # Predict results
        predict_verbose = 0
        if verbose == 2:
            print('Computing similarity metrics')
            predict_verbose = 2
        predictions = model.predict(neighbors, 
                                    verbose=predict_verbose)
        
        # Clean df from any previous results
        need_replace = False
        for c in classes:
            if c in list(df.columns):
                df.drop(columns=[c], inplace=True)
                need_replace = True
        if 'H_class' in list(df.columns):
            df.drop(columns=['H_class'], inplace=True)
            need_replace = True
        # Add predictions to df
        for i in range(len(classes)):
            df[classes[i]] = predictions[i]
        # Add class predictions
        df['H_class'] = tf.argmin(tf.constant(predictions), axis=0)

        # Check if an output prefix was given
        if not output_prefix == None and '/' in path:
            # Split path by folder bars
            split_path = path.split('/')
            # Add prefix to final location
            split_path[-1] = output_prefix + split_path[-1]
            # Concatenate path
            path = '/'.join(split_path)
        elif not output_prefix == None and not '/' in path:
            # If there are no folder bars 
            # just add the prefix
            path = output_prefix + path

        # Write new file
        if verbose == 2:
            print('Writing results')
        write_dump_from_df(df, header, path, 
                            new_cols=[*classes,'H_class'],
                            replace_cols=need_replace)

        # Add to counter
        k += 1

    # If verbose is 1 or 2, finish timing and output total time
    if verbose in [1,2]:
        time_2 = time.time()
        minutes_final = (time_2 - time_1)//60
        seconds_final = (time_2 - time_1)%60
        print('\nProcess completed')
        print('Elapsed time: %d minutes and %.f seconds' % \
            (minutes_final,seconds_final)) 
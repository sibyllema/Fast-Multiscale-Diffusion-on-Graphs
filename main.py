import os
import numpy as np
from time import time
from tqdm import tqdm

import requests # Performing HTTPS requests
import zipfile # Manipulate zips

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm # Color map
plt.rcParams.update({'font.size': 12})

# Usefull functions
from scipy.special import ive # Bessel function
from scipy.special import factorial
from scipy.spatial.distance import squareform
from scipy.linalg import expm

# Sparse matrix algebra
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh # Eigenvalues computation
from scipy.sparse.csgraph import laplacian # Laplacian from sparse matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm_multiply as sparse_expm_multiply

# Logging. errors/warnings handling
from pdb import set_trace as bp
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("matplotlib").setLevel(logging.WARNING) # Don't want matplotlib to print so much stuff
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Max level of debug info. to display

################################################################################
### Utility functions ##########################################################
################################################################################

def plot_fancy_error_bar(x, y, ax=None, type="median_quartiles", label=None, **kwargs):
    """ Plot data with errorbars and semi-transparent error region.

    Arguments:
    x -- list or ndarray, shape (nx,)
        x-axis data
    y -- ndarray, shape (nx,ny)
        y-axis data. Usually represents ny attempts for each datum in x.
    ax -- matplotlib Axis
        Axis to plot the data on
    type -- string.
        Type of error. Either "median_quartiles" or "average_std".
    kwargs -- dict
        Extra options for matplotlib (such as color, label, etc).
    """
    if type=="median_quartiles":
        y_center    = np.percentile(y, q=50, axis=-1)
        y_up        = np.percentile(y, q=25, axis=-1)
        y_down      = np.percentile(y, q=75, axis=-1)
    elif type=="average_std":
        y_center    = np.average(y, axis=-1)
        y_std       = np.std(y, axis=-1)
        y_up        = y_center + y_std
        y_down      = y_center - y_std

    if ax is None:
        plot_ = plt.errorbar(x, y_center, (y_center - y_down, y_up - y_center), label=label, **kwargs)
        plt.fill_between(x, y_down, y_up, alpha=.3, **kwargs)
    else:
        plot_ = ax.errorbar(x, y_center, (y_center - y_down, y_up - y_center), label=label, **kwargs)
        ax.fill_between(x, y_down, y_up, alpha=.3, **kwargs)
    return plot_

def get_firstmm_db_dataset():
    logger.debug("### get_firstmm_db_dataset() ###")

    if os.path.exists("data/FIRSTMM_DB"):
        logger.debug("Dataset already downloaded")
        return None

    logger.debug("Downloading FIRSTMM_DB dataset.")
    r = requests.get("https://www.chrsmrrs.com/graphkerneldatasets/FIRSTMM_DB.zip")

    logger.debug("Writing FIRSTMM_DB dataset (as zip).")
    with open("data/FIRSTMM_DB.zip", "wb") as f:
        f.write(r.content)

    logger.debug("Unziping FIRSTMM_DB dataset.")
    with zipfile.ZipFile("data/FIRSTMM_DB.zip", "r") as zip_ref:
        zip_ref.extractall("data/")

def parse_dortmund_format(path, dataset_name, clean_data=True):
    """ Parser for attributed graphs from the Dortmund gaph collection (see:
        https://chrsmrrs.github.io/datasets/docs/datasets/)."""
    logger.debug("### parse_dortmund_format() ###")

    logger.debug("Reading node->graph mapping")
    with open(path + dataset_name + "_graph_indicator.txt") as f:
        graph_indicator = []
        current_graph_num = 0
        for line in f:
            graph_num = int(line) - 1
            graph_indicator.append(graph_num)

    # Auxiliary quantity: total number of graphs
    n_graphs = np.amax(graph_indicator) + 1

    # Auxiliary quantity: size of each graph (in nodes)
    graph_sizes = np.zeros( (n_graphs,), dtype=np.int )
    graph_min_node = np.ones( (n_graphs,), dtype=np.int ) * (len(graph_indicator) + 1)
    for i,k in enumerate(graph_indicator):
        graph_sizes[k] += 1
        graph_min_node[k] = min(i, graph_min_node[k])

    # Auxiliary quantity: are all the graphs the same size?
    # If yes, building the data require some tweaking (because of automatic
    # NumPy array formatting).
    same_size = (len(np.unique(graph_sizes))==1)

    logger.debug("Reading graphs structures")
    M = []
    current_graph_num = 0
    with open(path + dataset_name + "_A.txt") as f:
        rows = []
        cols = []
        for line in f:
            i,j = line.split(",")
            i,j = int(i)-1,int(j)-1
            if current_graph_num != graph_indicator[i]:
                M.append( csr_matrix(
                    (np.ones(len(rows)),(rows,cols)),
                    shape=(graph_sizes[current_graph_num],graph_sizes[current_graph_num])
                ) )
                current_graph_num += 1
                rows = []
                cols = []
            rows.append(i - graph_min_node[current_graph_num])
            cols.append(j - graph_min_node[current_graph_num])
        else:
            M.append( csr_matrix(
                (np.ones(len(rows)),(rows,cols)),
                shape=(graph_sizes[current_graph_num],graph_sizes[current_graph_num])
            ) )
    M = np.array(M)

    if clean_data:
        to_delete = []
        for i in range(n_graphs):
            # Removed graph with more than 1 connected component
            if connected_components(M[i])[0] > 1:
                logger.debug(f"Graph {i} is not connected.")
                to_delete.append(i)
        M = np.delete(M,to_delete, axis=0)

    return_dict = {"graph_structures": M}

    logger.debug("Reading graph labels")
    with open(path + dataset_name + "_graph_labels.txt") as f:
        L = []
        for line in f:
            L.append(int(line))
    L = np.array(L)
    # Normalise the labels: 0, 1, 2, ...
    _, L = np.unique(L, return_inverse=True)
    # Clean data?
    if clean_data:
        L = np.delete(L,to_delete, axis=0)
    # Update return dictionnary
    return_dict["graph_labels"] = L

    # Read all other files the same way
    for file in os.scandir(path):
        # Is the file already read?
        if not file.name.startswith(dataset_name+"_node"):
        # if (file.name.endswith("_A.txt")
        #         or file.name.endswith("_graph_labels.txt")
        #         or file.name.endswith("_graph_indicator.txt")
        #         or file.name=="README.txt"):
            continue
        logger.debug(f"Reading {file.name}.")
        with open(file.path) as f:
            X = []
            for i,line in enumerate(f):
                x = line.split(',')
                x = np.array(x, dtype=np.float)
                try:
                    X[graph_indicator[i]].append(x)
                except:
                    X.append([x])
            if same_size:
                X = np.array(X, dtype=np.float).reshape( (n_graphs, graph_sizes[0], -1) )
            else:
                X = np.array([np.array(x_list, dtype=np.float) for x_list in X], dtype=np.object)
                # X = np.array(X, dtype=object)
        # Clean data?
        if clean_data:
            X = np.delete(X,to_delete, axis=0)
        # Truncate the file name (to remove useless info).
        data_name = file.name[len(dataset_name)+1:-4]
        # Update return dictionnary
        return_dict[data_name] = X

    # Process node labels
    if "node_labels" in return_dict.keys():
        logger.debug("Processing node labels.")
        # Graph the labels
        X = return_dict["node_labels"]
        # Squeeze every array, and convert everything to int
        X = np.array([np.squeeze(l).astype(int) for l in X], dtype=np.object)
        # Put everything back in the dictionnary
        return_dict["node_labels"] = X

    # Some stats
    logger.debug(f"Found {n_graphs} graphs")
    logger.debug(f"Found {len(np.unique(L))} classes")
    logger.debug(f"Found {np.amin(graph_sizes)}-->{np.amax(graph_sizes)} graph sizes")

    return return_dict

################################################################################
### Krylov subspaces-base method from 1812.10165 (ART) #########################
################################################################################

def art_expm(A, v, t, toler=1e05, m=10, verbose=False):
    """ Computes y = exp(-t.A).v approximately.

    Uses the Arnoldi method with the RT (residual time) restarting proposed in
    M.A. Botchev, L.A. Knizhnerman, ART: adaptive residual-time restarting for
    Krylov subspace matrix exponential evaluations http://arxiv.org/abs/1812.10165

    Adapted from the MatLab implementation.

    Copyright (c) 2018 by M.A. Botchev
    Permission to copy all or part of this work is granted, provided that the
    copies are not made or distributed for resale, and that the copyright notice
    and this notice are retained.
    THIS WORK IS PROVIDED ON AN "AS IS" BASIS.  THE AUTHOR PROVIDES NO WARRANTY
    WHATSOEVER, EITHER EXPRESSED OR IMPLIED, REGARDING THE WORK, INCLUDING
    WARRANTIES WITH RESPECT TO ITS MERCHANTABILITY OR FITNESS FOR ANY
    PARTICULAR PURPOSE.

    Input:
    - A       (n x n)-matrix
    - v       n-vector
    - t>0     length of the time interval
    - toler>0 tolerance
    - m       maximal Krylov dimension
    Output:
    - y       the approximate solution
    - mvec    number of matrix-vector multiplications done to compute y (not anymore)
    """
    n = len(v)
    V = np.zeros((n,m+1))
    H = np.zeros((m+1,m))

    convergence = False
    mvec_count  = 0
    while not convergence:
        beta = np.linalg.norm(v)
        V[:,0] = np.squeeze(v/beta)

        for j in range(m):
            w = A@V[:,j]
            mvec_count = mvec_count + 1
            for i in range(j):
                H[i,j] = w.T @ V[:,i]
                w      = w - H[i,j]*V[:,i]
            H[j+1,j] = np.linalg.norm(w)
            e1       = np.zeros((j+1,1)); e1[0]  = 1
            ej       = np.zeros((j+1,1)); ej[-1] = 1
            s        =  [t*i/6 for i in range(6)]
            beta_j   = np.empty( (len(s),), dtype=np.float )
            for q in range(len(s)):
                u         = expm(-s[q] * H[0:j+1,0:j+1]) @ e1 # TODO: faster
                beta_j[q] = -H[j+1,j] * (ej.T @ u)
            resnorm = np.linalg.norm(beta_j, np.inf)
            if resnorm<=toler:
                if verbose: print(f"j = {j}, resnorm = {resnorm:.2e} - convergence!")
                convergence = True
                break
            elif j+1==m:
                if verbose: print("j = {j}, resnorm = {resnorm:.2e}")
                if verbose: print(f"-------- restart after {m} steps")
                # Find n_tsteps - number of steps to monitor the residual
                n_tsteps    = 100
                u           = e1
                resid_value = 2*toler
                while resid_value>toler:
                    expmH       = expm( -(t/n_tsteps) * H[0:j+1,0:j+1] )
                    u           = expmH @ e1
                    resid_value = -H[j+1,j] * (ej.T @ u)
                    if abs(resid_value)<=toler:
                        u = e1
                        break
                    n_tsteps = 2*n_tsteps
                # keyboard % to plot residual vs t - MB
                # Compute residual for intermediate time points until its
                # value exceeds tolerance
                for k in range(n_tsteps):
                    u_old       = u
                    u           = expmH @ u
                    resid_value = -H[j+1,j] * (ej.T @ u)
                    if abs(resid_value)>toler:
                        u_ok  = u_old
                        t_ok  = (k-1)/n_tsteps * t
                        y_ok  = V[:,0:j+1] @ (beta * u_ok)
                        if verbose: print(", time interval reduced by {round(t_ok/t*100)}%%")
                        t = t - t_ok
                        v = y_ok
                        break          # restart
                break                  # restart
            V[:,j+1] = w / H[j+1,j]

    y = V[:,0:j+1] @ (beta*u)

    # return y, mvec
    return y

################################################################################
### Theoretical bound definition ###############################################
################################################################################

def h(K,C):
    """ Equivalent function to g() from "Two polynomial methods of calculating
        functions of symmetric matrices". """
    # Swap g and h to experiment with the bound from Durskin89.
    m = K+1
    a = 2*C
    return 2 * (a/2)**m * np.exp(a**2/4) / (factorial(m) * (1- a/(2*(m+1))))

def g(K,C):
    # True value
    return 2 * np.exp((C**2.)/(K+2)-2*C) * (C**(K+1))/(factorial(K)*(K+1-C))
    # # Upper bound, maybe more stable
    # x = C*C/(K+2) - 2*C + (K+1)*np.log(C) - (K+.5)*np.log(K) + K
    # return np.sqrt(2/np.pi) * np.exp(x)

def get_bound_eps_generic(phi, x, tau, K):
    C  = tau*phi/2.
    return g(K,C)**2.

def get_bound_eta_generic(phi, x, tau, K):
    C  = tau*phi/2.
    assert(K > C-1)
    return g(K,C)**2. * np.exp(8*C)

def get_bound_eta_specific(phi, x, tau, K):
    C  = tau*phi/2.
    n  = len(x)
    a1 = np.sum(x)
    assert(a1 != 0.)
    return g(K,C)**2. * n * np.linalg.norm(x)**2. / (a1**2.)

def E(C, K):
    b = 2 / (1 + np.sqrt(5))
    d = np.exp(b) / (2 + np.sqrt(5))
    if K <= 4*C:
        return np.exp( (-b * (K+1)**2.) / (4*C)) * (1 + np.sqrt(C * np.pi / b)) + (d**(4*C)) / (1-d)
    else:
        return (d**K) / (1-d)

def get_bound_bergamaschi_generic(phi, x, tau, K):
    C  = tau*phi/2.
    return (2*E(C, K)*np.exp(4*C))**2.

def get_bound_bergamaschi_specific(phi, x, tau, K):
    C  = tau*phi/2.
    n  = len(x)
    a1 = np.sum(x)
    assert(a1 != 0.)
    return 4 * E(C,K)**2. * n * np.linalg.norm(x)**2. / (a1**2.)

def reverse_bound(f, phi, x, tau, err):
    """ Returns the minimal K such that f(L,x,tau,K) <= err. """
    # Starting value: C-1
    C   = tau*phi/2.
    K_min = max(1,int(C))

    # Step 0: is E(C-1) enough?
    if f(phi,x,tau,K_min) <= err:
        return K_min

    # Step 1: searches any K such that f(*args) <= err.
    K_max = 2 * K_min
    while f(phi,x,tau,K_max) > err:
        K_min = K_max
        K_max = 2 * K_min

    # Step 2: now we have f(...,K_max) <= err < f(...,K_min). Dichotomy!
    while K_max > 1+K_min:
        K_int = (K_max + K_min) // 2
        if f(phi,x,tau,K_int) <= err:
            K_max = K_int
        else:
            K_min = K_int
    return K_max

def reverse_eps_K(L, x, tau, err):
    """ Returns the minimum K required to achieve a given error (relative to the
        input). """
    y_ref = sparse_expm_multiply(-tau*L, x)
    K = 1
    def get_eps(a,b):
        return (np.linalg.norm(a-b)/np.linalg.norm(x))**2.
    while get_eps(y_ref, expm_multiply(L, x, tau, K=K)) > err:
        K += 1
    return K

def reverse_eta_K(L, x, tau, err):
    """ Returns the minimum K required to achieve a given error (relative to the
        input). """
    y_ref = sparse_expm_multiply(-tau*L, x)
    K = 1
    def get_eta(a,b):
        return (np.linalg.norm(a-b)/np.linalg.norm(a))**2.
    while get_eta(y_ref, expm_multiply(L, x, tau, K=K)) > err:
        K += 1
    return K

################################################################################
### Our method to compute the diffusion ########################################
################################################################################

def compute_chebychev_pol(X, L, phi, K):
    """ Compute the Tk(L).X, where Tk are the K+1 first Chebychev polynoms. """
    N, d = X.shape
    T = np.empty((K + 1, N, d), dtype=np.float)
    # Initialisation
    T[0] = X
    T[1] = (1 / phi) * L @ X - T[0]
    # Recursive computation of T[2], T[3], etc.
    for j in range(2, K + 1):
        T[j] = (2 / phi) * L @ T[j-1] - 2 * T[j-1] - T[j-2]
    return T

def compute_chebychev_coeff_all(phi, tau, K):
    """ Compute recursively the K+1 Chebychev coefficients for our functions. """
    return 2*ive(np.arange(0, K+1), -tau * phi)

def expm_multiply(L, X, tau, K=None, err=1e-32):
    """ Computes the action of exp(-t*L) on X for all t in X."""
    # Compute phi = l_max/2
    phi = eigsh(L, k=1, return_eigenvectors=False)[0] / 2
    # Check if tau is a single value, a list or a ndarray.
    if isinstance(tau, (float, int)):
        # Get minimal K to go below the desired error
        if K is None: K = reverse_bound(get_bound_eta_specific, phi, X, tau, err)
        # Compute Chebychev polynomials
        poly = compute_chebychev_pol(X, L, phi, K)
        # Compute Chebychev coefficients
        coeff = compute_chebychev_coeff_all(phi, tau, K)
        # Perform linear combination
        Y = .5 * coeff[0] * poly[0] + (poly[1:].T @ coeff[1:]).T
        return Y
    elif isinstance(tau, list):
        # Same as earlier, but iterate on a list.
        if K is None: K = reverse_bound(get_bound_eta_specific, phi, X, max(tau), err)
        poly = compute_chebychev_pol(X, L, phi, K)
        coeff_list = [compute_chebychev_coeff_all(phi, t, K) for t in tau]
        Y_list = [.5 * coeff[0] * poly[0] + (poly[1:].T @ coeff[1:]).T for coeff in coeff_list]
        return Y_list
    elif isinstance(tau, np.ndarray):
        if K is None: K = reverse_bound(get_bound_eta_specific, phi, X, np.amax(tau), err)
        poly = compute_chebychev_pol(X, L, phi, K)
        f = lambda t: compute_chebychev_coeff_all(phi, t, K)
        g = lambda coeff: .5 * coeff[0] * poly[0] + (poly[1:].T @ coeff[1:]).T
        h = lambda t: g(f(t))
        # Yes I know, it' s a for loop.
        # I can't make np.vectorize work >.<
        out = np.empty(tau.shape+X.shape, dtype=X.dtype)
        for index,t in np.ndenumerate(tau):
            out[index] = h(t)
        return out
    else:
        print(f"expm_multiply(): unsupported data type for tau ({type(tau)})")

def get_diffusion_fun(L, X, K=None):
    """ Creates a function to compute exp(-t*L) on X, for t given later."""
    # If K is not provided, fall back on default value.
    if K is None:
        K = K_base
    # Compute phi = l_max/2
    phi = eigsh(L, k=1, return_eigenvectors=False)[0] / 2
    # Compute Chebychev polynomials
    poly = compute_chebychev_pol(X, L, phi, K)
    # Define a function to be applied for multipel values of tau
    def f(tau):
        coeff = compute_chebychev_coeff_all(phi, tau, K)
        Y = .5 * coeff[0] * poly[0] + (poly[1:].T @ coeff[1:]).T
        return Y
    return f

################################################################################
### Data #######################################################################
################################################################################

def sample_er(N, p, gamma):
    """ Sample an Erdos-Reyni graph (as a laplacian) and a 1d gaussian signal on
        its nodes. """
    # Sample the adjacency matrix, in a compressed fashio (only generates the
    # top triangular part, as a 1-dimensional vector).
    A_compressed = np.random.choice(2, size=(N*(N-1)//2,), p=[1.-p,p])
    # Compute the graph's combinatorial laplacian
    L = laplacian(csr_matrix(squareform(A_compressed), dtype=np.float))
    # Sample the features
    X = np.random.randn(N,1) * gamma
    # Conclude
    return L, X

def get_er(k, N=200, p=.05, gamma=1.):
    """ Iterator. Yields k Erdos-Reyni graphs. """
    for i in range(k):
        yield sample_er(N, p, gamma)

def get_firstmm_db(k):
    """ Iterator. Yields k attributed graphs from the FIRSTMM_DB dataset. """
    data_dict = parse_dortmund_format(f"data/FIRSTMM_DB/", "FIRSTMM_DB")
    N = len(data_dict["node_attributes"])
    p = np.random.permutation(N)
    X_all = data_dict["node_attributes"][p[:k]]
    A_all = data_dict["graph_structures"][p[:k]]
    L_all = [laplacian(A) for A in A_all]
    return zip(L_all, X_all)

################################################################################
### How much time do the individual steps take #################################
################################################################################

def time_steps():
    n_graphs = 10
    n_tau    = 10
    tau_all  = 10**np.linspace(-2.,0.,num=n_tau)
    err      = 1e-5

    time_eig  = np.empty((n_tau,n_graphs), dtype=np.float)
    time_K    = np.empty((n_tau,n_graphs), dtype=np.float)
    time_poly = np.empty((n_tau,n_graphs), dtype=np.float)
    time_coef = np.empty((n_tau,n_graphs), dtype=np.float)
    time_comb = np.empty((n_tau,n_graphs), dtype=np.float)

    pbar = tqdm(total=n_graphs*n_tau)
    for i,(L,X) in enumerate(get_er(n_graphs, N=1000)):
        for j,tau in enumerate(tau_all):
            # Largest eigenvalue computation
            t_start = time()
            phi     = eigsh(L, k=1, return_eigenvectors=False)[0] / 2
            t_stop  = time()
            time_eig[j,i] = t_stop - t_start

            # Compute required K
            t_start = time()
            K       = reverse_bound(get_bound_eta_specific, phi, X, tau, err)
            t_stop  = time()
            time_K[j,i] = t_stop - t_start

            # Compute the polynomials
            t_start = time()
            poly    = compute_chebychev_pol(X, L, phi, K)
            t_stop  = time()
            time_poly[j,i] = t_stop - t_start

            # Compute the coefficients
            t_start = time()
            coeff   = compute_chebychev_coeff_all(phi, tau, K)
            t_stop  = time()
            time_coef[j,i] = t_stop - t_start

            # Combine polynomials & coefficients
            t_start = time()
            Y       = .5 * coeff[0] * poly[0] + (poly[1:].T @ coeff[1:]).T
            t_stop  = time()
            time_comb[j,i] = t_stop - t_start

            pbar.update(1)
    pbar.close()

    plot_fancy_error_bar(tau_all, time_eig,  label="First eigenvalue")
    plot_fancy_error_bar(tau_all, time_K,    label="Order K")
    plot_fancy_error_bar(tau_all, time_poly, label="Polynomials")
    plot_fancy_error_bar(tau_all, time_coef, label="Coefficients")
    plot_fancy_error_bar(tau_all, time_comb, label="Combination")

    plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.show()

################################################################################
### Theoretical bound analysis #################################################
################################################################################

def min_K_er():
    """ Display the minimum K to achieve a desired accuracy against tau. """
    logger.debug("### min_K_er() ###")
    n_graphs = 10
    n_val    = 20
    tau_all  = 10**np.linspace(-2.,2.,num=n_val)
    # tau      = 1.
    # err_all  = 10**np.linspace(-16, -3, num=n_val)
    err      = 1e-5
    bound_8_all  = np.empty( (n_graphs,n_val), dtype=np.float )
    bound_9_all  = np.empty( (n_graphs,n_val), dtype=np.float )
    bound_11_all = np.empty( (n_graphs,n_val), dtype=np.float )
    bound_12_all = np.empty( (n_graphs,n_val), dtype=np.float )
    real_K_all   = np.empty( (n_graphs,n_val), dtype=np.float )

    logger.debug("Computing minimum K")
    pbar = tqdm(total=n_graphs*n_val)
    for i,(L,X) in enumerate(get_er(n_graphs)):
        for j,tau in enumerate(tau_all):
        # for j,err in enumerate(err_all):
            phi = eigsh(L, k=1, return_eigenvectors=False)[0] / 2
            bound_8_all[i,j]  = reverse_bound(get_bound_eta_generic, phi, X, tau, err)
            bound_9_all[i,j]  = reverse_bound(get_bound_eta_specific, phi, X, tau, err)
            bound_11_all[i,j] = reverse_bound(get_bound_bergamaschi_specific, phi, X, tau, err)
            bound_12_all[i,j] = reverse_bound(get_bound_bergamaschi_generic, phi, X, tau, err)
            real_K_all[i,j]   = reverse_eta_K(L, X, tau, err)
            pbar.update(1)
    pbar.close()

    # Plot all this
    plot_fancy_error_bar(tau_all, bound_8_all.T, label=f"With bound V.8 (generic)", linestyle="dashed")
    plot_fancy_error_bar(tau_all, bound_9_all.T, label=f"With bound V.9 (specific)", linestyle="dashed")
    plot_fancy_error_bar(tau_all, bound_12_all.T, label=f"With Bergamaschi's generic", linestyle="dotted")
    plot_fancy_error_bar(tau_all, bound_11_all.T, label=f"With Bergamaschi's specific", linestyle="dotted")
    plot_fancy_error_bar(tau_all, real_K_all.T, label=f"Real required K", color="black")
    # plt.xlabel(r"Desired $\eta$")
    plt.xlabel(r"$\tau$")
    plt.ylabel("K")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()

################################################################################
### Speed // ER // Increasing set of tau #######################################
################################################################################

def speed_for_set_of_tau():
    logger.debug("### speed_for_set_of_tau() ###")
    n_graphs = 10
    n_rep    = 6
    tau_log_min = -5.
    tau_log_max = 0.

    time_sp = np.zeros((n_rep,n_graphs))
    time_ar = np.zeros((n_rep,n_graphs))
    time_cb = np.zeros((n_rep,n_graphs))

    pbar = tqdm(total=n_graphs*n_rep)
    for i,(L,X) in enumerate(get_er(n_graphs, N=10000, p=.05)):
        for j in range(n_rep):
            # Number of tau values to pick, beyind the max & min ones
            rep = j
            # All tau values.
            tau_all = 10**np.linspace(tau_log_min, tau_log_max,num=rep+2)
            # Compute scipy's method
            t_start = time()
            for tau in tau_all:
                _ = sparse_expm_multiply(-tau*L, X)
            t_stop = time()
            time_sp[j,i] += t_stop - t_start
            # Compute ART's method
            t_start = time()
            for tau in tau_all:
                _ = art_expm(L, X, tau, toler=1e-5, m=60)
            t_stop = time()
            time_ar[j,i] += t_stop - t_start
            # Compute our method
            t_start = time()
            _ = expm_multiply(L, X, tau_all, err=1e-5)
            t_stop = time()
            time_cb[j,i] += t_stop - t_start

            pbar.update(1)
    pbar.close()

    x = list(range(2, n_rep+2))
    plot_fancy_error_bar(x, time_sp, label="Scipy")
    plot_fancy_error_bar(x, time_cb, label="Chebychev")
    plot_fancy_error_bar(x, time_ar, label="ART (Krylov)")

    plt.xlabel(r"Number of $\tau$ values")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid()
    plt.show()

################################################################################
### Speed and precision with tau increasing ####################################
################################################################################

def speed_MSE_analysis_firstmm_db():
    # Experiment parameters
    n_graphs  = 10 # Number of graphs to average the performance over
    n_tau_val = 20 # Number of tau values
    n_runs    = 50 # Number of runs to average performances over
    tau_list  = 10**np.linspace(-5.,-1., num=n_tau_val)

    # How much time does each method takes
    time_sp = np.zeros((n_tau_val,n_runs,n_graphs))
    time_ar = np.zeros((n_tau_val,n_runs,n_graphs))
    time_cb = np.zeros((n_tau_val,n_runs,n_graphs))

    # Precision of the methods wrt to NumPy's (which we assum is correct up to arithmetic precision)
    err_ar = np.zeros((n_tau_val,n_runs,n_graphs))
    err_cb = np.zeros((n_tau_val,n_runs,n_graphs))

    # Loop over graphs
    pbar = tqdm(total=n_graphs*n_tau_val*n_runs)
    for i,(L,_) in enumerate(get_firstmm_db(n_graphs)):
        # Loop over runs
        for j in range(n_runs):
            # Build a standard signal/initial heat value: 1 on a node, 0 elswhere
            N,_ = L.shape
            idx = np.random.default_rng().integers(low=0,high=N)
            X = np.zeros((N,1), dtype=np.float)
            X[idx] = 1.
            # Pre-compute the Chebychev polynomials, and spread its
            # computation time over all values of tau.
            t_start = time()
            t_stop = time()
            time_cb[:,j,i] += (t_stop - t_start) / n_tau_val
            for k,tau in enumerate(tau_list):
                # Compute diffusion with scipy's method
                t_start = time()
                Y_sp = sparse_expm_multiply(-tau*L, X)
                t_stop = time()
                time_sp[k,j,i] += t_stop - t_start

                # Compute diffusion with ART's method
                t_start = time()
                Y_ar = art_expm(L, X, tau, toler=1e-3, m=20)
                t_stop = time()
                time_ar[k,j,i] += t_stop - t_start

                # Compute diffusion with our method
                t_start = time()
                f_cb = get_diffusion_fun(L, X, K=10)
                Y_cb = f_cb(tau)
                t_stop = time()
                time_cb[k,j,i] += t_stop - t_start

                # Compute and store MSE
                err_ar[k,j,i] = (np.linalg.norm(Y_sp-Y_ar)/np.linalg.norm(Y_sp))**2
                err_cb[k,j,i] = (np.linalg.norm(Y_sp-Y_cb)/np.linalg.norm(Y_sp))**2

                pbar.update(1)
    pbar.close()

    # Average times/MSE over graphs
    time_sp = np.average(time_sp, axis=-1)
    time_ar = np.average(time_ar, axis=-1)
    time_cb = np.average(time_cb, axis=-1)
    err_ar = np.average(err_ar, axis=-1)
    err_cb = np.average(err_cb, axis=-1)

    # Prepare plots
    f, ax0 = plt.subplots(nrows=1, ncols=1)
    # ax1 = plt.twinx()

    # Plot computation times wrt tau
    plt0sp = plot_fancy_error_bar(tau_list, time_sp, ax=ax0, color="red",   linestyle="solid", label="(time) Scipy")
    plt0cb = plot_fancy_error_bar(tau_list, time_cb, ax=ax0, color="blue",  linestyle="solid", label="(time) Chebychev")
    plt0ar = plot_fancy_error_bar(tau_list, time_ar, ax=ax0, color="green", linestyle="solid", label="(time) ART (Krylov)")

    # # Plot MSE wrt tau
    # plt1cb = plot_fancy_error_bar(tau_list, err_cb, ax=ax1, color="blue",  linestyle="dashed", label="(error) Chebychev")
    # plt1ar = plot_fancy_error_bar(tau_list, err_ar, ax=ax1, color="green", linestyle="dashed", label="(error) ART (Krylov)")

    # Configure plot
    plt.xlabel(r"$\tau$")
    plt.xscale("log")
    ax0.set_ylabel("Time (s)")
    # ax1.set_ylabel("MSE")
    # ax1.set_yscale("log")
    plt_all = [plt0sp, plt0cb, plt0ar]
    plt.legend(plt_all, [plt_.get_label() for plt_ in plt_all])
    plt.grid()
    plt.show()

################################################################################
### 3d diagram tau/K/error #####################################################
################################################################################

def generate_K_tau_err_figure():
    # Experiment parameters
    n_K_val   = 15 # Number of values of K
    n_tau_val = 20 # Number of tau values
    n_graphs  = 10 # Number of graphs to average the performance over
    K_list    = np.arange(1,1+n_K_val)
    tau_list  = 10**np.linspace(-5.,1., num=n_tau_val)

    # Experiment results
    err_all_1 = np.empty((n_K_val,n_tau_val,n_graphs), dtype=np.float)
    err_all_2 = np.empty((n_K_val,n_tau_val,n_graphs), dtype=np.float)
    err_ref   = np.empty((n_K_val,n_tau_val,n_graphs), dtype=np.float)

    pbar=tqdm(total=n_K_val*n_tau_val*n_graphs)
    for k,(L,X) in enumerate(get_firstmm_db(n_graphs)):
        for i,K in enumerate(K_list):
            for j,tau in enumerate(tau_list):
                # Compute and store first bound
                err_all_1[i,j,k] = get_bound_eta_generic(L, tau, K)
                # Compute and store second bound
                err_all_2[i,j,k] = get_bound_eta_specific(L, X, tau, K)
                # Compute and store eta (MSE on output)
                y_che = expm_multiply(L, X, tau, K)
                y_ref = sparse_expm_multiply(-tau*L, X)
                err_ref[i,j,k] = (np.linalg.norm(y_che-y_ref) / np.linalg.norm(y_ref))**2.
                pbar.update(1)
    pbar.close()

    # Average errors over all sampled graphs
    err_all_1 = np.average(err_all_1, axis=-1)
    err_all_2 = np.average(err_all_2, axis=-1)
    err_ref   = np.average(err_ref, axis=-1)

    # 3D plot the error landscapes
    fig = plt.figure()
    X,Y = np.meshgrid(K_list, np.log10(tau_list), indexing="ij")

    # First bound
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(X, Y, np.log10(err_all_1), cmap=cm.coolwarm)
    ax.set_xlabel("K", rotation=-25)
    ax.set_ylabel(r"$\log_{10}(\tau^t)$")
    ax.set_zlabel(r"$\log_{10}(bound 1)$", rotation=60)
    # Second bound
    ax = fig.add_subplot(132, projection='3d')
    ax.plot_surface(X, Y, np.log10(err_all_2), cmap=cm.coolwarm)
    ax.set_xlabel("K", rotation=-25)
    ax.set_ylabel(r"$\log_{10}(\tau^t)$")
    ax.set_zlabel(r"$\log_{10}(bound 2)$", rotation=60)
    # Real error
    ax = fig.add_subplot(133, projection='3d')
    ax.plot_surface(X, Y, np.log10(err_ref), cmap=cm.coolwarm)
    ax.set_xlabel("K", rotation=-25)
    ax.set_ylabel(r"$\log_{10}(\tau^t)$")
    ax.set_zlabel(r"$\log_{10}(\eta)$", rotation=60)

    plt.show()

################################################################################
### Main #######################################################################
################################################################################

if __name__=="__main__":
    # min_K_er()
    # time_steps()
    # speed_for_set_of_tau()

import numpy as np
from time import time
from tqdm import tqdm

# Data
from ogb.nodeproppred import NodePropPredDataset

# Plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

# Useful functions
from scipy.special import ive # Bessel function
from scipy.special import factorial
from scipy.spatial.distance import squareform
from scipy.sparse import load_npz as load_sparse
from scipy.stats import linregress # Affine regression

# Sparse matrix algebra
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh # Eigenvalues computation
from scipy.sparse.csgraph import laplacian # Laplacian from sparse matrix
from scipy.sparse.linalg import expm_multiply as scipy_expm_multiply

# Logging. errors/warnings handling
from pdb import set_trace as bp # Add breakpoints if you want to inspect things
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("matplotlib").setLevel(logging.WARNING) # Don't want matplotlib to print so much stuff
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Max level of debug info. to display

################################################################################
### Utility functions ##########################################################
################################################################################

def plot_fancy_error_bar(x, y, ax=None, type="median_quartiles", **kwargs):
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

    fill_color = kwargs["color"] if "color" in kwargs else None

    if ax is None:
        plot_ = plt.errorbar(x, y_center, (y_center - y_down, y_up - y_center), **kwargs)
        plt.fill_between(x, y_down, y_up, alpha=.3, color=fill_color)
    else:
        plot_ = ax.errorbar(x, y_center, (y_center - y_down, y_up - y_center), **kwargs)
        ax.fill_between(x, y_down, y_up, alpha=.3, color=fill_color)
    return plot_

################################################################################
### Theoretical bound definition ###############################################
################################################################################

def g(K,C):
    return 2 * np.exp((C**2.)/(K+2)-2*C) * (C**(K+1))/(factorial(K)*(K+1-C))

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
    if len(x.shape)==1:
        # Case 1: X has shape (n,), it is one signal.
        a1 = np.sum(x)
        assert(a1 != 0.)
        return g(K,C)**2. * n * np.linalg.norm(x)**2. / (a1**2.)
    elif len(x.shape)==2:
        # Case 2: X has shape (n,dim), it is multiple signals.
        # Take the maximum bound for every signal
        a1 = np.sum(x, axis=0)
        assert(not np.any(a1==0.))
        return g(K,C)**2. * n * np.amax(np.linalg.norm(x, axis=0)**2. / (a1**2.))

def E(K, C):
    b = 2 / (1 + np.sqrt(5))
    d = np.exp(b) / (2 + np.sqrt(5))
    if K <= 4*C:
        return np.exp( (-b * (K+1)**2.) / (4*C)) * (1 + np.sqrt(C * np.pi / b)) + (d**(4*C)) / (1-d)
    else:
        return (d**K) / (1-d)

def get_bound_bergamaschi_generic(phi, x, tau, K):
    C  = tau*phi/2.
    return (2*E(K, C)*np.exp(4*C))**2.

def get_bound_bergamaschi_specific(phi, x, tau, K):
    C  = tau*phi/2.
    n  = len(x)
    # Same branch as in get_bound_eta_specific()
    if len(x.shape)==1:
        a1 = np.sum(x)
        assert(a1 != 0.)
        return 4 * E(K,C)**2. * n * np.linalg.norm(x)**2. / (a1**2.)
    elif len(x.shape)==2:
        a1 = np.sum(x, axis=0)
        assert(not np.any(a1==0.))
        return 4 * E(K,C)**2. * n * np.amax(np.linalg.norm(x, axis=0)**2. / (a1**2.))

def reverse_bound(f, phi, x, tau, err):
    """ Returns the minimal K such that f(L,x,tau,K) <= err. """
    # Starting value: C-1
    C   = tau*phi/2.
    K_min = max(1,int(C))

    # Step 0: is C-1 enough?
    if f(phi,x,tau,K_min) <= err:
        return K_min

    # Step 1: searches a K such that f(*args) <= err, by doubling step size.
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
    y_ref = scipy_expm_multiply(-tau*L, x)
    K = 1
    def get_eps(a,b):
        return (np.linalg.norm(a-b)/np.linalg.norm(x))**2.
    while get_eps(y_ref, expm_multiply(L, x, tau, K=K)) > err:
        K += 1
    return K

def reverse_eta_K(L, x, tau, err):
    """ Returns the minimum K required to achieve a given error (relative to the
        input). """
    y_ref = scipy_expm_multiply(-tau*L, x)
    K = 1
    def get_eta(a,b):
        return (np.linalg.norm(a-b)/np.linalg.norm(a))**2.
    while get_eta(y_ref, expm_multiply(L, x, tau, K=K)) > err:
        K += 1
    return K

################################################################################
### Our method to compute the diffusion ########################################
################################################################################

def compute_chebychev_coeff_all(phi, tau, K):
    """ Compute the K+1 Chebychev coefficients for our functions. """
    return 2*ive(np.arange(0, K+1), -tau * phi)

def expm_multiply(L, X, tau, K=None, err=1e-32):
    # Get statistics
    phi = eigsh(L, k=1, return_eigenvectors=False)[0] / 2
    N, d = X.shape
    # Case 1: tau is a single value
    if isinstance(tau, (float, int)):
        # Compute minimal K
        if K is None: K = reverse_bound(get_bound_eta_specific, phi, X, np.amax(tau), err)
        # Compute coefficients (they should all fit in memory, no problem)
        coeff = compute_chebychev_coeff_all(phi, tau, K)
        # Initialize the accumulator with only the first coeff*polynomial
        T0 = X
        Y  = .5 * coeff[0] * T0
        # Add the second coeff*polynomial to the accumulator
        T1 = (1 / phi) * L @ X - T0
        Y  = Y + coeff[1] * T1
        # Recursively add the next coeff*polynomial
        for j in range(2, K + 1):
            T2 = (2 / phi) * L @ T1 - 2 * T1 - T0
            Y  = Y + coeff[j] * T2
            T0 = T1
            T1 = T2
        return Y
    # Case 2: tau is, in fact, a list of tau
    # In this case, we return the list of the diffusions as these times
    elif isinstance(tau, list):
        if K is None: K = reverse_bound(get_bound_eta_specific, phi, X, max(tau), err)
        coeff_list = [compute_chebychev_coeff_all(phi, t, K) for t in tau]
        T0 = X
        Y_list  = [.5 * coeff[0] * T0 for coeff in coeff_list]
        T1 = (1 / phi) * L @ X - T0
        Y_list  = [Y + coeff[1] * T1 for Y,coeff in zip(Y_list, coeff_list)]
        for j in range(2, K + 1):
            T2 = (2 / phi) * L @ T1 - 2 * T1 - T0
            Y_list = [Y + coeff[j] * T2 for Y,coeff in zip(Y_list, coeff_list)]
            T0 = T1
            T1 = T2
        return Y_list
    # Case 3: tau is a numpy array
    elif isinstance(tau, np.ndarray):
        # Compute the order K corresponding to the required error
        if K is None: K = reverse_bound(get_bound_eta_specific, phi, X, np.amax(tau), err)
        # Compute the coefficients for every tau
        coeff = np.empty(tau.shape+(K+1,), dtype=np.float64)
        for index,t in np.ndenumerate(tau):
            coeff[index] = compute_chebychev_coeff_all(phi, t, K)
        # Compute the output for just the first polynomial*coefficient
        T0 = X
        Y = np.empty(tau.shape+X.shape, dtype=X.dtype)
        for index,t in np.ndenumerate(tau):
            Y[index] = .5 * coeff[index][0] * T0
        # Add the second polynomial/*oefficient
        T1 = (1 / phi) * L @ X - T0
        for index,t in np.ndenumerate(tau):
            Y[index] = Y[index] + coeff[index][1] * T1
        # Recursively add the others polynomials*coefficients
        for j in range(2, K + 1):
            T2 = (2 / phi) * L @ T1 - 2 * T1 - T0
            for index,t in np.ndenumerate(tau):
                Y[index] = Y[index] + coeff[index][j] * T2
            T0 = T1
            T1 = T2
        return Y
    else:
        print(f"expm_multiply(): unsupported data type for tau ({type(tau)})")

################################################################################
### Data #######################################################################
################################################################################

def sample_er(N, p, gamma):
    """ Sample an Erdos-Reyni graph (as a laplacian) and a 1d gaussian signal on
        its nodes. """
    # Sample the adjacency matrix, in a compressed fashion (only generates the
    # top triangular part, as a 1-dimensional vector).
    A_compressed = np.random.choice(2, size=(N*(N-1)//2,), p=[1.-p,p])
    # Compute the graph's combinatorial laplacian
    L = laplacian(csr_matrix(squareform(A_compressed), dtype=np.float64))
    # Sample the signal
    X = np.random.randn(N,1) * gamma
    # Conclude
    return L, X

def get_er(k, N=200, p=.05, gamma=1.):
    """ Iterator. Yields k Erdos-Reyni graphs. """
    for i in range(k):
        yield sample_er(N, p, gamma)

def get_standford_bunny():
    L = load_sparse("data/standford_bunny_laplacian.npz")
    X = np.load("data/standford_bunny_coords.npy")
    return L,X

################################################################################
### Theoretical bound analysis #################################################
################################################################################

def minimal_K_against_tau():
    """ Display the minimum K to achieve a desired accuracy against tau. """
    logger.debug("### minimal_K_against_tau() ###")
    n_graphs = 100
    n_val    = 25
    tau_all  = 10**np.linspace(-2.,2.,num=n_val)
    err      = 1e-5
    bound_18_all  = np.empty( (n_graphs,n_val), dtype=np.float64 )
    bound_19_all  = np.empty( (n_graphs,n_val), dtype=np.float64 )
    bound_21_all = np.empty( (n_graphs,n_val), dtype=np.float64 )
    bound_23_all = np.empty( (n_graphs,n_val), dtype=np.float64 )
    real_K_all    = np.empty( (n_graphs,n_val), dtype=np.float64 )

    logger.debug("Computing minimum K")
    pbar = tqdm(total=n_graphs*n_val)
    for i,(L,X) in enumerate(get_er(n_graphs)):
        # Collect statistics
        lmax   = eigsh(L, k=1, return_eigenvectors=False)[0]
        phi    = lmax / 2
        for j,tau in enumerate(tau_all):
            bound_18_all[i,j]  = reverse_bound(get_bound_eta_generic, phi, X, tau, err)
            bound_19_all[i,j]  = reverse_bound(get_bound_eta_specific, phi, X, tau, err)
            bound_21_all[i,j] = reverse_bound(get_bound_bergamaschi_specific, phi, X, tau, err)
            bound_23_all[i,j] = reverse_bound(get_bound_bergamaschi_generic, phi, X, tau, err)
            real_K_all[i,j]    = reverse_eta_K(L, X, tau, err)
            pbar.update(1)
    pbar.close()

    logger.debug("Plotting data")
    plot_fancy_error_bar(tau_all, bound_18_all.T, label=f"Bound (18) (Our, generic)", linestyle="solid", marker="o", color="blue")
    plot_fancy_error_bar(tau_all, bound_19_all.T, label=f"Bound (19) (Our, specific)", linestyle="solid", marker="x", color="blue")
    plot_fancy_error_bar(tau_all, bound_21_all.T, label=f"Bound (21) (Ref [11], specific)", linestyle="dotted", marker="x", color="red")
    plot_fancy_error_bar(tau_all, bound_23_all.T, label=f"Bound (23) (Ref [11], generic)", linestyle="dotted", marker="o", color="red")
    plot_fancy_error_bar(tau_all, real_K_all.T,   label=f"Real required K", color="black")

    logger.debug("Configuring plot")
    plt.xlabel(r"$\tau$")
    plt.ylabel("K")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.legend()

    logger.debug("Displaying plot")
    plt.show()

################################################################################
### Speed and precision with tau increasing ####################################
################################################################################

def speed_standford_bunny():
    logger.debug("### speed_standford_bunny() ###")
    logger.debug("Defining experiment parameters")
    # Experiment parameters
    n_runs      = 100 # Number of runs to average performances over
    tau_log_min = -3.
    tau_log_max = 1.
    tau_num_all = 2*np.arange(1,10)

    logger.debug("Allocating memory for results")
    time_sp = np.empty((len(tau_num_all),n_runs,), dtype=np.float64)
    time_cb = np.empty((len(tau_num_all),n_runs,), dtype=np.float64)

    logger.debug("Loading data")
    L,_ = get_standford_bunny()
    N,_ = L.shape

    logger.debug("Case 1: tau values are linearly spaced")
    pbar = tqdm(total=n_runs*len(tau_num_all))
    for i in range(n_runs):
        # Build a standard signal/initial heat value: 1 on a node, 0 elsewhere
        idx = np.random.default_rng().integers(low=0,high=N)
        X = np.zeros((N,1), dtype=np.float64)
        X[idx] = 1.

        # Iterate over number of \tau values
        for j,tau_num in enumerate(tau_num_all):
            # Time Scipy's method
            t_start = time()
            if tau_num==1:
                _ = scipy_expm_multiply(tau_list[0]*L, X)
            else:
                _ = scipy_expm_multiply(-L, X, start=10**tau_log_min, stop=10**tau_log_max, num=tau_num, endpoint=True)
            t_stop = time()
            time_sp[j,i] = t_stop - t_start

            # Time our method
            t_start = time()
            tau_list = [10**tau_log_min/2+10**tau_log_max/2] if tau_num==1 else np.linspace(10**tau_log_min, 10**tau_log_max, num=tau_num)
            _ = expm_multiply(L, X, tau_list, err=1e-5)
            t_stop = time()
            time_cb[j,i] = t_stop - t_start

            pbar.update(1)
    pbar.close()

    logger.debug("Results:")
    res_sp = linregress(np.repeat(tau_num_all, n_runs), time_sp.flatten())
    res_cb = linregress(np.repeat(tau_num_all, n_runs), time_cb.flatten())
    logger.debug(f"sp: t = {res_sp.intercept:.2E} + n_scale*{res_sp.slope:.2E}")
    logger.debug(f"cb: t = {res_cb.intercept:.2E} + n_scale*{res_cb.slope:.2E}")

    logger.debug("Case 1: tau values are uniformly sampled at random")
    pbar = tqdm(total=n_runs*len(tau_num_all))
    for i in range(n_runs):
        # Build a standard signal/initial heat value: 1 on a node, 0 elsewhere
        idx = np.random.default_rng().integers(low=0,high=N)
        X = np.zeros((N,1), dtype=np.float64)
        X[idx] = 1.

        # Iterate over number of \tau values
        for j,tau_num in enumerate(tau_num_all):
            tau_list = np.random.default_rng().uniform(low=10.**tau_log_min, high=10.**tau_log_max, size=(tau_num,))

            # Time Scipy's method
            t_start = time()
            for tau in tau_list:
                _ = scipy_expm_multiply(-tau*L, X)
            t_stop = time()
            time_sp[j,i] = t_stop - t_start

            # Time our method
            t_start = time()
            _ = expm_multiply(L, X, tau_list, err=1e-5)
            t_stop = time()
            time_cb[j,i] = t_stop - t_start

            pbar.update(1)
    pbar.close()

    logger.debug("Results:")
    res_sp = linregress(np.repeat(tau_num_all, n_runs), time_sp.flatten())
    res_cb = linregress(np.repeat(tau_num_all, n_runs), time_cb.flatten())
    logger.debug(f"sp: t = {res_sp.intercept:.2E} + n_scale*{res_sp.slope:.2E}")
    logger.debug(f"cb: t = {res_cb.intercept:.2E} + n_scale*{res_cb.slope:.2E}")

################################################################################
### Timing diffusion at multiple scales in a big graph #########################
################################################################################

def speed_ogbn_arxiv():
    logger.debug("### speed_ogbn_arxiv() ###")
    logger.debug("Loading data from file")
    dataset_name = "ogbn-arxiv"
    dataset = NodePropPredDataset(name=dataset_name, root='data/')

    logger.debug("Building graph labels")
    L = dataset.labels.squeeze()
    lbl_all = np.unique(L)
    n_lbl = len(lbl_all)
    N = len(L)

    logger.debug("Building node attribute")
    x = dataset.graph["node_feat"]
    n = len(x)

    logger.debug("Building graph structure")
    # Build the adjacency matrix
    a = csr_matrix(
        arg1=(np.ones(len(dataset.graph["edge_index"][0])),
              (dataset.graph["edge_index"][0],
               dataset.graph["edge_index"][1])),
        shape=(n,n)
    )
    # Make the adjacency matrix symmetric
    a = a + a.T - diags(a.diagonal())
    # Build the laplacian from the adjacency matrix
    gl = laplacian(a)

    logger.debug("Defining parameters (tau interval, error)")
    # Following recommendation of https://arxiv.org/pdf/1710.10321.pdf
    gam = .95
    eta = .85
    l2  = 3.50654818e-03 # eigsh(gl, k=1, which="SM", return_eigenvectors=False)[0] # 3.50654818e-03 1.46529700e-12
    lm  = 13162.001602973913 # eigsh(gl, k=1, return_eigenvectors=False)[0]
    logger.debug(f"l2={l2}")
    logger.debug(f"lm={lm}")
    tau_min = -np.log(gam) / np.sqrt(l2*lm)
    tau_max = -np.log(eta) / np.sqrt(l2*lm)
    logger.debug(f"tau_min={tau_min}")
    logger.debug(f"tau_max={tau_max}")
    err = 10**-3

    t_sp = 0.
    t_cb4 = []
    t_cb8 = []
    for i in range(40):
        n_run = i + 1
        logger.debug(f"Iteration {n_run}")
        # Compute diffusion with Scipy's method
        tau = np.random.default_rng().uniform(low=tau_min,high=tau_max,size=(1,))
        t_start = time()
        _ = scipy_expm_multiply(-tau[0]*gl, x)
        t_stop = time()
        t_sp += t_stop - t_start
        print(f"t_sp={t_sp/n_run:.2E}")

        # Compute diffusion for our method, with 4 values of \tau
        tau = np.random.default_rng().uniform(low=tau_min,high=tau_max,size=(4,))
        t_start = time()
        _ = expm_multiply(gl, x, tau, err=err)
        t_stop = time()
        t_cb4.append(t_stop - t_start)
        print(f"t_cb4={t_cb4/n_run:.2E}")

        # Compute diffusion for our method, with 4 values of \tau
        tau = np.random.default_rng().uniform(low=tau_min,high=tau_max,size=(8,))
        t_start = time()
        _ = expm_multiply(gl, x, tau, err=err)
        t_stop = time()
        t_cb8.append(t_stop - t_start)
        print(f"t_cb8={t_cb4/n_run:.2E}")

        # Compute interpolation
        res_cb = linregress(
            np.repeat(np.array([4,8]), n_run),
            np.concatenate([t_cb4,t_cb8]))
        print(f"t_cb = {res_cb.intercept:.2E} + n_scale*{res_cb.slope:.2E}")

################################################################################
### Main #######################################################################
################################################################################

if __name__=="__main__":
    # Figure 1 of the paper
    minimal_K_against_tau()
    # First claim related to speed
    speed_standford_bunny()
    # Second claim related to speed
    speed_ogbn_arxiv()

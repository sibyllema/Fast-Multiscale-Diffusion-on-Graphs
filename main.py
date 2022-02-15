from core import np
from utils import plt
from time import time
from tqdm import tqdm

# Import core
from core import get_bound_eta_generic, get_bound_eta_specific
from core import get_bound_bergamaschi_generic, get_bound_bergamaschi_specific
from core import reverse_bound
from core import expm_multiply

# Import utils
from utils import plot_fancy_error_bar
# Data
from ogb.nodeproppred import NodePropPredDataset

# Plotting
plt.rcParams.update({'font.size': 8})

# Useful functions
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
### Error measurement ##########################################################
################################################################################

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
    L = load_sparse("data/bunny/standford_bunny_laplacian.npz")
    X = np.load("data/bunny/standford_bunny_coords.npy")
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
    plot_fancy_error_bar(tau_all, bound_18_all.T, label=f"Bound (18) (Ours, generic)", linestyle="solid", marker="o", color="blue")
    plot_fancy_error_bar(tau_all, bound_19_all.T, label=f"Bound (19) (Ours, specific)", linestyle="solid", marker="x", color="blue")
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
    err = 1e-5

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
            _ = expm_multiply(L, X, tau_list, err=err)
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
            _ = expm_multiply(L, X, tau_list, err=err)
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
### 3d plot of diffusion on standford bunny ####################################
################################################################################

from mpl_toolkits.mplot3d import Axes3D
def plot_bunny():
    # Load data
    L,pos = get_standford_bunny()
    N,_   = pos.shape
    lmax  = eigsh(L, k=1, return_eigenvectors=False)[0]

    # Re-order 3d coordinates, for plotting later
    pos[:,1],pos[:,2] = -pos[:,2],pos[:,1].copy()

    # Create a (Dirac) signal
    X = np.zeros((N,1), dtype=np.float64)
    X[0] = 1.

    # Prepare the figure and the expriment parameters
    fig = plt.figure()
    tau_all = np.array([.05,.1,.2,.5,1.,2.,5.,10.])
    # tau_all = np.linspace(.001, 10, num=8)
    err = 1e-2

    # Diffuse the signal with our method
    for i,tau in enumerate(tau_all):
        # Prepare plot
        ax = fig.add_subplot(240+i+1, projection='3d')
        # # Get diffusion with scipy
        # t_start = time()
        Y_cb  = expm_multiply(L, X, tau, err=err)
        # t_stop = time()
        # t_cb = t_stop - t_start
        # Get diffusion with our method
        # t_start = time()
        Y_sp = scipy_expm_multiply(-tau*L, X)
        # t_stop = time()
        # t_sp = t_stop - t_start
        # Get order K
        K = reverse_bound(get_bound_eta_specific, lmax/2, X, tau, err)
        # Get error eps_K
        eps_K = (np.linalg.norm(Y_sp - Y_cb)/np.linalg.norm(X))**2.
        # Plot
        ax.scatter(xs=pos[:,0], ys=pos[:,1], zs=pos[:,2],c=Y_cb)
        ax.set_title(f"τ'={tau}, K={K}\nε={eps_K:.2E}")
        # ax.set_title(f"s={100*(t_cb-t_sp)/t_sp:1.0f}%")
        # Configure plot
        ax.axis("off")
        ax.set_xlim(np.amin(pos[:,0])*.9, np.amax(pos[:,0])*.9)
        ax.set_ylim(np.amin(pos[:,1])*.9, np.amax(pos[:,1])*.9)
        ax.set_zlim(np.amin(pos[:,2])*.9, np.amax(pos[:,2])*.9)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Save abd display plot
    plt.savefig("fig/bunny.pdf",bbox_inches="tight")
    plt.show()

################################################################################
### Main #######################################################################
################################################################################

if __name__=="__main__":
    # Figure 1 of the paper
    minimal_K_against_tau()
    # # First claim related to speed
    speed_standford_bunny()
    # # Second claim related to speed
    speed_ogbn_arxiv()
    # # Bunny plot
    plot_bunny()

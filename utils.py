import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from sklearn.metrics import adjusted_mutual_info_score, accuracy_score

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Max level of debug info. to display
from pdb import set_trace as bp

def auto_zip_pairs(l):
    """ Iterate on all pairs that can be formed from a list, except duplicates
        (x,x) and symmetries (x,y forbids y,x). """
    n = len(l)
    for i in range(n):
        for j in range(i+1,n):
            yield l[i],l[j]

def parse_dortmund_format(path, dataset_name, clean_data=True):
    """ Parser for attributed graphs from the Dortmund gaph collection (see:
        https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)."""
    logging.debug("### parse_dortmund_format() ###")

    logging.debug("Reading node->graph mapping")
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

    logging.debug("Reading graphs structures")
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
                logging.debug(f"Graph {i} is not connected.")
                to_delete.append(i)
        M = np.delete(M,to_delete, axis=0)

    return_dict = {"graph_structures": M}

    logging.debug("Reading graph labels")
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

    # if get_features:
    #     logging.debug("Reading feature vectors")
    #     with open(path + dataset_name + "_node_attributes.txt") as f:
    #         X = []
    #         for i,line in enumerate(f):
    #
    # if get_node_labels:
    #     logging.debug("Reading node labels")
    #     with open(path + dataset_name + "_node_labels.txt") as f:
    #         A = []
    #         for i,line in enumerate(f):
    #             l = int(line)
    #             try:
    #                 A[graph_indicator[i]].append(l)
    #             except:
    #                 A.append([l])
    #     A = [np.array(l_list) for l_list in A]
    #     A = np.array(A, dtype=object).squeeze()

        #     # Remove graphs whose structure is not a square matrix (why does it happen though?)
        #     elif not (a==b and a==graph_sizes[i]):
        #         logging.debug(f"Graph {i} has an erroneous structure matrix size")
        #         to_delete.append(i)
        #     # Remove graph whose number of features does not match the number of nodes
        #     elif get_features:
        #         if not len(X[i])==graph_sizes[i]:
        #             logging.debug(f"Graph {i} does not have the right number of node features")
        #             to_delete.append(i)
        #     # Remove graphs whose number of node labels does not match
        #     elif get_node_labels:
        #         if not len(A[i])==graph_sizes[i]:
        #             logging.debug(f"Graph {i} does not have the right number of node labels")
        #             to_delete.append(i)
        # M = np.delete(M,to_delete, axis=0)
        # if get_graph_labels: L = np.delete(L,to_delete, axis=0)
        # if get_features: X = np.delete(X,to_delete, axis=0)
        # if get_node_labels: A = np.delete(A,to_delete, axis=0)
        # n_graphs = len(M)

    # Some stats
    logging.debug(f"Found {n_graphs} graphs")
    logging.debug(f"Found {len(np.unique(L))} classes")
    logging.debug(f"Found {np.amin(graph_sizes)}-->{np.amax(graph_sizes)} graph sizes")
    #
    # # Building return dictionnary
    # return_dict = {"graph_structures": M}
    # if get_graph_labels:
    # if get_features: return_dict["node_features"] = X
    # if get_node_labels: return_dict["node_labels"] = A

    return return_dict

def assignment_from_transport_map(gamma, Ls, method='avg'):
    '''
    From a transport map and a set of source labels, label the target.

    Parameters
    ----------
    gamma: 2d-array. Transport map.
    Ls: 1d-array. Source labels.
    method: string, optional. Method used to compute the target labels. Either
    'avg' or 'max'.

    Returns the target labels as a 1d-array.
    '''
    Ns,Nt = gamma.shape
    Ut = np.ones( (Ns,) ) @ gamma
    if method=='avg':
        # 1/ For every target point, compute the mass coming from each label.
        # 2/ Organize this into an Nt*n_labels matrix.
        # 3/ Use majority vote to decide each target point's label.
        L_weights = np.array([np.sum(gamma[Ls==l], axis=0) / Ut for l in set(Ls)])
        return np.argmax(L_weights, axis=0)
    elif method=='max':
        # 1/ For every target point, compute the incoming source point of maximum mass.
        # 2/ The label of a target point is the same as the computed corresponding source point.
        L_source_max = np.argmax(gamma, axis=0)
        return Ls[L_source_max]
    elif method=="test":
        # This shouuuuuld be equivalent to the "avg" method/
        # Runs more tests?
        ps = np.zeros( (Ns, np.amax(Ls)+1) )
        ps[np.arange(Ns),Ls] = 1
        p = (gamma.T @ ps) * np.broadcast_to(Ut, (np.amax(Ls)+1,Nt)).T
        return np.argmax(p, axis=1)

def score_acc_forward_backward(Ls, gamma):
    Lt_pseudo = assignment_from_transport_map(gamma, Ls)
    Ls_pseudo = assignment_from_transport_map(gamma.T, Lt_pseudo)
    # score = accuracy_score(Ls, Ls_pseudo)
    score = adjusted_mutual_info_score(Ls, Ls_pseudo, average_method="arithmetic")
    return score

def ami_score_partial(Ls, gamma, Lt, Lt_mask):
    Lt_pseudo = assignment_from_transport_map(gamma, Ls)
    score = adjusted_mutual_info_score(Lt_pseudo[Lt_mask], Lt[Lt_mask], average_method="arithmetic")
    return score

def plot_fancy_error_bar(x, y, type="median_quartiles", label=None, **kwargs):
    """ Plot data with errorbars and semi-transparent error region.

    Arguments:
    x -- list or ndarray, shape (nx,)
        x-axis data
    y -- ndarray, shape (nx,ny)
        y-axis data. Usually represents ny attempts for each datum in x.
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
        y_center    = np.average(x, axis=-1)
        y_std       = np.std(x, axis=-1)
        y_up        = y_center + y_std
        y_down      = y_center - y_std

    plt.errorbar(x, y_center, (y_center - y_down, y_up - y_center), label=label, **kwargs)
    plt.fill_between(x, y_down, y_up, alpha=.3, **kwargs)

# def pairs_2(Xa, Xb):
#     na,d = Xa.shape
#     nb,_ = Xb.shape
#     Xa_expanded = np.repeat(Xa,nb,axis=0).reshape(na,nb,d)
#     Xb_expanded = np.moveaxis(np.repeat(Xb,na,axis=0).reshape(nb,na,d),0,1)
#     return Xa_expanded,Xb_expanded
# def pairwise_2(Xa, Xb):
#     """ Pairwise L2 distance, that should pass automatic differentiation."""
#     na,d = Xa.shape
#     nb,_ = Xb.shape
#     Xa_expanded = np.repeat(Xa,nb,axis=0).reshape(na,nb,d)
#     Xb_expanded = np.moveaxis(np.repeat(Xb,na,axis=0).reshape(nb,na,d),0,1)
#     return np.linalg.norm(Xa_expanded-Xb_expanded, axis=-1)

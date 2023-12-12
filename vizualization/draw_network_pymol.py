import seaborn as sns
import pickle as pkl
from sklearn.cluster import Birch
from networkx.algorithms.community import girvan_newman, modularity
import networkx as nx
from scipy.sparse import load_npz, csr_matrix
import numpy as np
import warnings
from pymol import cmd, stored, selector
from pymol.cgo import *
import pandas as pd
from scipy.sparse.dok import dok_matrix
import matplotlib as mpl
import pickle5 as pkl5
import matplotlib.pyplot as plt
import gudhi as gd
from copy import deepcopy
import matplotlib.colors as mcolors
mpl.use('Qt5Agg')


# Liste de couleurs personnalisée
noBlueRed = [
    (1.0, 0.48627450980392156, 0.0),
    (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
    (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
    (0.6235294117647059, 0.2823529411764706, 0.0),
    (0.9450980392156862, 0.2980392156862745, 0.7568627450980392),
    (0.6392156862745098, 0.6392156862745098, 0.6392156862745098),
    (1.0, 0.7686274509803922, 0.0),
    (0.0, 0.8431372549019608, 1.0)
]



def minus_log(mat):
    """Function performing a -np.log to compute distances from input matrix.
    Sanitizes the input but raises warnings. If the input matrix has negative
    values it takes the absolute value and the if there are some values 
    in [1; +inf[ it rescales in the [0-1[ range.

    Parameters
    ----------
    x: array_like
    Input array for distance computation

    Output
    ------
    y: array_like of same size than x
    Distance matrix
    """
    if isinstance(mat, pd.DataFrame):
        mat.fillna(0, inplace=True)
        mat = mat.values

    if (0 <= mat).all() and (mat < 1).all():
        return -np.log(mat)
    else:
        warnings.warn("Values not in [0-1[ range, Rescaling to this range")
        if np.all(0 <= mat):
            # We impose x != 1 by adding a small eps to the max
            # This is not a problem to compute log but we don't want a
            # 0 distance
            mat = mat / (np.max(mat) + 1e-5)
        else:
            warnings.warn("Array has negative values, forcing positivity")
            mat = np.abs(mat)
            mat = mat / (np.max(mat) + 1e-5)
        return -np.log(mat)

def inverse(mat, fill_value=None):
    return 1/np.abs(mat)

str_to_func = {'minus_log': minus_log, 'inverse': inverse}

def load_df(df):
    try:
        return pd.read_pickle(df)
    except ValueError:
        return pkl5.load(open(df, 'rb'))


def isfloat(value):
    """Function to check that a value is a float like (convertible to float)

    Parameters
    ----------
    value: anything
    Value to check

    Output
    ----------
    bool, True if is convertible to float else False"""
    if isinstance(value, list):
        value = value[0]
    try:
        float(value)
        return True
    except ValueError:
        return False


def getnum(string):
    """Function to get the number in a string

    Parameters
    ----------
    string: str
    Input string to extract a number from

    Output
    ----------
    num: int or np.nan
    Number extracted or np.nan if no number is found in the string"""
    try:
        return int(string)
    except ValueError:
        new_str = ''
        for value in string:
            if isfloat(value):
                new_str += value
        try:
            return int(new_str)
        except ValueError:
            return np.nan


def create_topmat(sele, top, map_indexes, map_residues):
    """Function to create a topology matrix from a topology and mappings
    """
    if sele is None:
        sele = "polymer and not hydrogen"
    sele = "{} and {}".format(sele, top)
    stored.indexlist = []
    cmd.iterate(sele, 'stored.indexlist.append(index)')
    topmat = dok_matrix((len(map_indexes),
                        len(pd.unique(list(map_residues.values())))))
    for id in stored.indexlist:
        topmat[map_indexes[id], map_residues[id]] = 1
    return topmat


def connect_edges(nbunch, ebunch, connect_nodes):
    edges_to_add = []
    # Add neighbors in AANs
    if connect_nodes == 1:
        for u in nbunch:
            for v in nbunch:
                if abs(u-v) == 1 and (u, v) not in ebunch:
                    edges_to_add.append((u, v, 1e-10))

    # Add within residue in 2GN
    if connect_nodes in [2, 4]:
        for i, u in enumerate(nbunch):
            for v in nbunch[i+1:]:
                if abs(u//2-v//2) == 0 and (u, v) not in ebunch:
                    edges_to_add.append((u, v, 1e-10))

    # Add within residue in 3GN
    if connect_nodes in [3, 5]:
        for i, u in enumerate(nbunch):
            for v in nbunch[i+1:]:
                if abs(u//3-v//3) == 0 and (u, v) not in ebunch:
                    edges_to_add.append((u, v, 1e-10))

    # Add backbone-backbone in 2GN
    if connect_nodes == 4:
        for i, u in enumerate(nbunch):
            for v in nbunch[i+1:]:
                if (abs(u//2-v//2) == 1 and u % 2*v % 2 == 0
                        and (u, v) not in ebunch):
                    edges_to_add.append((u, v, 1e-10))

    # Add backbone-backbone in 3GN
    if connect_nodes == 5:
        for i, u in enumerate(nbunch):
            for v in nbunch[i+1:]:
                if (abs(u//3-v//3) == 1 and u % 3*v % 3 == 0
                        and (u, v) not in ebunch):
                    edges_to_add.append((u, v, 1e-10))

    # Add all within in 2GN
    if connect_nodes == 6:
        for i, u in enumerate(nbunch):
            for v in nbunch[i+1:]:
                if abs(u//2-v//2) <= 1 and (u, v) not in ebunch:
                    edges_to_add.append((u, v, 1e-10))

    # Add all within in 3GN
    if connect_nodes == 7:
        for i, u in enumerate(nbunch):
            for v in nbunch[i+1:]:
                if abs(u//3-v//3) <= 1 and (u, v) not in ebunch:
                    edges_to_add.append((u, v, 1e-10))

    return edges_to_add


def get_cca(df, weight='weight', source='node1', target='node2', cut_diam=3,
            smaller_max=False, connect_nodes=None, plot_cca=None):
    """Internal function that performs connected component analysis on a
    given network.
    Parameters
    ----------
    df: pandas.DataFrame
    Dataframe representing the considered network

    weight: str, default='weight'
    Column name for the weight of the considered network

    source: str, default='node1'
    Column name for source node

    target: str, default='node2'
    Column name for target node

    cut_diam: int, default=3
    Threshold on diameter for the displayed connected components

    smaller_max: bool, default=False
    Experimental parameter that allows to select a value smaller than the
    maximum for the number of connected components. Allows to clear more
    a graph if needed. This parameter is not really useful. Might be
    deprecated

    color_compo: bool, default=False
    Toggles coloring by connected components

    connect_nodes: bool, default=None
    Toggles some node connection during Connected Component Analysis

    plot_cca: str or None, default=None
    If different than None, is the output path for plotting the number of
    connected components in function of the threshold

    Output
    ---------
    df: pandas.DataFrame
    Dataframe representing the output network from connected component
    analysis.

    Reference
    ---------
    to be published
    """

    # old_colors = []
    # for i, c_str in enumerate(['color', 'color2']):
    #     if c_str in df.columns:
    #         old_colors.append(dict(zip(df['node{}'.format(i+1)], df[c_str])))
    #     else:
    #         old_colors.append({})

    net = nx.from_pandas_edgelist(df.dropna(), source=source, target=target,
                                  edge_attr=True)
    net.remove_nodes_from(list(nx.isolates(net)))
    edge_list = sorted(net.edges(data=True),
                       key=lambda t: abs(t[2].get(weight, 1)), reverse=True)
    connected_components = [[nx.number_connected_components(net), 0]]
    # prev_cc = np.array([connected_components[0][0]])
    while len(edge_list) != 0:
        u, v, dic = edge_list.pop()
        net.remove_edge(u, v)
        net.remove_nodes_from(list(nx.isolates(net)))
        cc = [nx.number_connected_components(net), abs(dic.get(weight, 1))]
        if connect_nodes is not None:
            # Add neighbors in AANs
            if connect_nodes == 1:
                if abs(u-v) == 1:
                    net.add_weighted_edges_from([(u, v, 1e-10)], weight=weight)
            # Add within residue in 2GN
            if connect_nodes in [2, 4]:
                if abs(u//2-v//2) == 0:
                    net.add_weighted_edges_from([(u, v, 1e-10)], weight=weight)
            # Add within residue in 3GN
            if connect_nodes in [3, 5]:
                if abs(u//3-v//3) == 0:
                    net.add_weighted_edges_from([(u, v, 1e-10)], weight=weight)

            cc = [nx.number_connected_components(net), abs(dic.get(weight, 1))]
        connected_components.append(cc)
    connected_components = np.array(connected_components)

    if plot_cca is not None:
        plt.plot(connected_components[:, 1],
                 connected_components[:, 0])
        plt.savefig(plot_cca)
        plt.close()
    m = np.argmax(connected_components[::-1, 0])
    if smaller_max:
        smaller_max = connected_components[-m, 0] - smaller_max
        point = connected_components[:, 0] == smaller_max
        threshold = connected_components[np.where(point)[0][-1]][1]
    else:
        threshold = connected_components[-m, 1]
    print(threshold)

    df = df.loc[df[weight].abs() > threshold]
    net = nx.from_pandas_edgelist(df.dropna(), source=source, target=target,
                                  edge_attr=True)
    if connect_nodes is not None:
        nbunch = list(net.nodes())
        ebunch = list(net.edges())
        edges_to_add = connect_edges(nbunch, ebunch, connect_nodes)

        net.add_weighted_edges_from(edges_to_add, weight=weight)

    components_list = [net.subgraph(c).copy()
                       for c in nx.connected_components(net)]
    if cut_diam > 0:
        robust = [list(c.nodes())
                  for c in components_list
                  if nx.diameter(c) >= float(cut_diam)]
        net = net.subgraph([x for robust in list(robust) for x in robust])
    components_list = [net.subgraph(c).copy()
                       for c in nx.connected_components(net)]
    df = nx.to_pandas_edgelist(net, source='node1', target='node2')
    # for elt, c_str, i in zip(old_colors, ['color', 'color2'], [1, 2]):
    #     if len(elt) != 0:
    #         df[c_str] = df['node{}'.format(i)].map(elt)
    print(len(df))
    return df


def get_girvan_newman(df, weight='weight', source='node1', target='node2',
                      color_compo=True, dist_func='minus_log',
                      plot_betweenness=False, impose_palette=None):
    """Internal function that performs community analysis using a girvan
    newman community decomposition optimizing the modularity measure.

    Parameters
    ----------
    df: pandas.DataFrame
    Dataframe representing the considered network

    weight: str, default='weight'
    Column name for the weight of the considered network

    source: str, default='node1'
    Column name for source node

    target: str, default='node2'
    Column name for target node

    color_compo: bool, default=False
    Toggles coloring by connected components

    dist_func: str or None, default='minus_log'
    if None, consider that the input network already represents distances
    else computes the distances using a predefined function

    plot_betweenness: bool, default=False
    if True the weight in the output network is replaced by the edge
    betweenness

    Output
    ---------
    df: pandas.DataFrame
    Dataframe representing the output network from connected component
    analysis.

    References
    ---------
    Girvan-Newman algorithm:

    Girvan, Michelle, and Mark EJ Newman.
    "Community structure in social and logical networks."
    Proc. Natl. Acad. Sci. 99.12 (2002): 7821-7826.

    Modularity:

    Newman, Mark EJ.
    "Modularity and community structure in networks."
    Proc. Natl. Acad. Sci. 103.23 (2006): 8577-8582.

    Application to MD simulations:

    Rivalta, Ivan, et al.
    "Allosteric pathways in imidazole glycerol phosphate synthase."
    Proc. Natl. Acad. Sci. 109.22 (2012): E1428-E1436.
    """
    if dist_func is not None:
        old_weights = {(u, v): w for u, v, w in zip(df[source],
                                                    df[target],
                                                    df[weight])}
        dist_func = str_to_func[dist_func]
        df['_{}'.format(weight)] = dist_func(df[weight].values)
        net = nx.from_pandas_edgelist(df.dropna(),
                                      source=source,
                                      target=target,
                                      edge_attr='_{}'.format(weight))
    else:
        net = nx.from_pandas_edgelist(df.dropna(),
                                      source=source,
                                      target=target,
                                      edge_attr=weight)
    net.remove_nodes_from(list(nx.isolates(net)))
    comp = girvan_newman(net)
    max_modularity = 0
    out_communities = None
    w = list(list(net.edges(data=True))[0][2].keys())[0]
    for communities in comp:
        mod = modularity(net, communities, weight=w)
        if mod >= max_modularity:
            out_communities = communities
            max_modularity = mod
    communities_list = [nx.subgraph(net, c).copy() for c in out_communities]
    if color_compo:
        n_colors = len(communities_list)
        palette = get_best_palette(n_colors, impose_palette)
        i2color = dict(enumerate(palette))
        node2compo = {}
        df = nx.to_pandas_edgelist(net, source=source, target=target)
        for i, c in enumerate(communities_list):
            for a in c:
                node2compo[a] = i

        df['community1'] = df[source].map(node2compo)
        df['community2'] = df[target].map(node2compo)

        df['color'] = df['community1'].map(i2color)
        df['color2'] = df['community2'].map(i2color)

        df['community1'] = df['community1'].map(lambda i: 'C{}'.format(i + 1))
        df['community2'] = df['community2'].map(lambda i: 'C{}'.format(i + 1))

    if dist_func is not None:
        def get_weights(x):
            if (x[source], x[target]) in old_weights:
                return old_weights[(x[source], x[target])]
            elif (x[target], x[source]) in old_weights:
                return old_weights[(x[target], x[source])]
            else:
                warnings.warn("Weight not found")
                return 0

        df[weight] = df.apply(get_weights, axis=1)
        if plot_betweenness:
            b = nx.edge_betweenness_centrality(net,
                                               weight='_{}'.format(weight))
            df[weight] = b.values()

    return df


def draw_Network(path, reset_view=True, hide_nodes=True, **kwargs):
    """Draws network """
    G = pkl.load(open(path, 'rb'))
    view = cmd.get_view()
    draw(G.df, **kwargs)
    if reset_view:
        cmd.set_view(view)
    if hide_nodes:
        cmd.disable('*nodes')


def draw_from_df(path, reset_view=True, hide_nodes=True, **kwargs):
    view = cmd.get_view()
    df = load_df(path)
    draw(df, **kwargs)
    if reset_view:
        cmd.set_view(view)
    if hide_nodes:
        cmd.disable('*nodes')


def get_best_palette(n_colors, impose_palette=None):
    if impose_palette:
        if impose_palette == "noBlueRed":
            if n_colors <= 8:
                return noBlueRed[:n_colors]
            else:
                print(noBlueRed*(n_colors // 8) + noBlueRed[:(n_colors % 8)+1], len(noBlueRed*(n_colors // 8) + noBlueRed[:(n_colors % 8)+1]))
                return noBlueRed*(n_colors // 8) + noBlueRed[:(n_colors % 8)+1]
        if isinstance(impose_palette[0], str):
            return [mcolors.to_rgb(cname) for cname in impose_palette]
        if n_colors > len(impose_palette):
            warnings.warn('Not enough colors in custom palette. Do at your\
                           own risk.')
            return sns.color_palette(impose_palette, n_colors=n_colors)
        else:
            return impose_palette[:n_colors]
    if n_colors < 8:
        palette = sns.color_palette('bright', n_colors=n_colors)

    elif 8 <= n_colors <= 9:
        palette = sns.color_palette('bright', n_colors=n_colors+1)
        palette.__delitem__(7)

    else:
        palette = sns.color_palette('husl', n_colors=n_colors)

    return palette


def _color_by(df, color_by, color_by_list, impose_palette):
    attributes = pd.unique(df[color_by])
    n_colors = len(attributes)
    if color_by_list:
        palette = color_by_list
        print(''.join('{} colored in {}; '.format(u, v)
                      for u, v in zip(attributes, palette)))
    else:
        palette = get_best_palette(n_colors, impose_palette)
    print(palette)
    attr2color = dict(zip(attributes, palette))
    df.loc[:, 'color'] = df[color_by].map(attr2color)
    df.loc[:, 'color2'] = df[color_by].map(attr2color)
    return df


def draw_from_atommat(path, perturbation=None, sele=None, sele1=None,
                      sele2=None, top=None, top_perturbation=None,
                      norm_expected=False, average_with=None, **kwargs):

    def _get_resmat(filepath, topo):
        def load_mat(path):
            if path.split('.')[-1] == 'npz':
                return load_npz(path)
            else:
                return csr_matrix(np.load(path))
        mat = load_mat(filepath)
        stored.top_indexes, stored.top_residues = [], []
        cmd.iterate(topo, 'stored.top_indexes.append(index)')
        cmd.iterate(topo, 'stored.top_residues.append(resi)')
        resid_list = []
        k = 0
        for elt1, elt2 in zip(stored.top_residues[:-1],
                              stored.top_residues[1:]):
            resid_list.append(k)
            if elt1 != elt2:
                k += 1
        resid_list.append(k)
        map_indexes = dict(zip(stored.top_indexes,
                               range(len(stored.top_indexes))))
        map_residues = dict(zip(stored.top_indexes, resid_list))
        top1 = create_topmat(sele1, topo, map_indexes, map_residues)
        top2 = create_topmat(sele2, topo, map_indexes, map_residues)
        resmat = (mat @ top1).transpose() @ top2
        resmat.setdiag(0)
        resmat.eliminate_zeros()
        if norm_expected:
            expected = (top1.sum(axis=1).transpose() @ top1).transpose() \
                @ (top2.sum(axis=1).transpose() @ top2)
            resmat /= expected
            resmat[np.isnan(resmat)] = 0
            resmat = csr_matrix(resmat)
        return resmat

    sele1 = "not hydrogen" if sele1 is None else sele1
    sele2 = "not hydrogen" if sele2 is None else sele2

    if top is None:
        top = '{}* and polymer'.format(path.split('.')[0].split('_')[0])

    if sele1 is None and sele2 is None:
        sele1, sele2 = sele, sele

    resmat = _get_resmat(path, top)
    if perturbation is not None:
        if top_perturbation is None:
            fmt = perturbation.split('.')[0].split('_')[0]
            top_perturbation = '{}* and polymer'.format(fmt)
        resmat2 = _get_resmat(perturbation, top_perturbation)
        resmat = resmat2 - resmat

    elif average_with is not None:
        if isinstance(average_with, str):
            average_with = [average_with]
        n_dynamics = len(average_with) + 1
        top_list = ['{}* and polymer'.format(filepath.split('.')[0]
                    .split('_')[0]) for filepath in average_with]
        resmat_list = [_get_resmat(_1, _2)
                       for _1, _2 in zip(average_with, top_list)]
        average = np.zeros((resmat.shape))
        for _ in [resmat] + resmat_list:
            average[_.nonzero()] += _.data
        average /= n_dynamics
        resmat = csr_matrix(average)

    stored.resnames, stored.resids, stored.chains = [], [], []
    _top = "({} or {}) and {}".format(sele1, sele2, top)
    cmd.iterate(_top, 'stored.resnames.append(resn)')
    cmd.iterate(_top, 'stored.resids.append(resi)')
    cmd.iterate(_top, 'stored.chains.append(chain)')
    nodes = pd.unique([resn + resi + ':' + chain
                       for resn, resi, chain
                       in zip(stored.resnames,
                              stored.resids,
                              stored.chains)])

    id2node = dict(enumerate(nodes))
    df = nx.to_pandas_edgelist(nx.from_scipy_sparse_matrix(resmat))
    df.columns = ['node1', 'node2', 'weight']
    df['node1'] = df['node1'].map(id2node)
    df['node2'] = df['node2'].map(id2node)

    draw(df, selection=top, **kwargs)


def cluster_birch(df, weight='weight', source='node1', target='node2',
                  n_clusters=None, save_plot=False):
    edgelist = np.array(df[weight].abs().values).reshape(-1, 1)
    brc = Birch(n_clusters=n_clusters)
    labels = brc.fit_predict(edgelist)
    max_lab = np.max(labels)
    c = []
    for i in range(max_lab+1):
        c.append(np.sum(labels == i))
    reorder = dict(enumerate(np.argsort(c)))
    labels = np.array([reorder[i] for i in labels])
    df['cluster'] = labels

    edgelist = edgelist.reshape(-1)

    if save_plot:
        fig, ax = plt.subplots()
        ordered_labels = labels[np.argsort(edgelist)]
        _, idx = np.unique(ordered_labels, return_index=True)
        i2color =  dict(zip(ordered_labels[np.sort(idx)], sns.color_palette("bright", len(np.unique(ordered_labels)))[::-1]))
        ax.scatter(np.sort(edgelist), np.linspace(len(edgelist), 0, len(edgelist), endpoint=False), 
            marker='+',
            c=np.array([i2color[label] for label in ordered_labels]))

        ax.set_xlabel('Weight')
        ax.set_ylabel('Count')
        axins = ax.inset_axes([0.3, 0.3, 0.6, 0.6])
        x0, xt = ax.get_xlim()
        y0, yt = ax.get_ylim()
        x0, xt = plt.gca().get_xlim()
        y0, yt = plt.gca().get_ylim()
        perc= 100
        w_t = np.sort(edgelist)[-perc]

        axins.scatter(np.sort(edgelist), np.linspace(len(edgelist), 0, len(edgelist), endpoint=False), 
            marker='+',
            c=np.array([i2color[label] for label in ordered_labels]))

        ticks = np.sort(edgelist)[np.where(np.diff(ordered_labels))]
        axins.set_xticks(ticks)
        axins.set_xticklabels([np.round(elt, 2) for elt in ticks], rotation=45)
        axins.set_xlim(4.5, xt)
        axins.set_ylim(0, perc)
        plt.savefig(save_plot, transparent=True)
    df['color'] = df['cluster'].map(i2color)
    return df

def get_persistent_homology(df, weight, alpha, dist_func='minus_log'):
    # print(df.loc[(df['node1'] == 3) & (df['node2'] == 4)])
    net = nx.from_pandas_edgelist(df.dropna(), source='node1', target='node2',
                                  edge_attr=True)
    adjmat = nx.to_numpy_array(net, weight=weight)

    if dist_func is not None:
        dist_func = str_to_func[dist_func]
        dmat = dist_func(adjmat)
    else:
        dmat = adjmat
    
    maxi = np.max(dmat[~np.isinf(dmat)])

    distances = [[]]
    for i, row in enumerate(dmat):
        if i > 1:
            distances.append(row[:i-1].tolist())
    rips = gd.RipsComplex(distance_matrix=distances, max_edge_length=maxi+1)
    st = rips.create_simplex_tree(max_dimension=0)
    edges_to_keep = [(vertices[0], vertices[1]) for vertices, dist
                      in st.get_filtration()
                      if len(vertices) == 2 and dist <= alpha]
    net = net.edge_subgraph(edges_to_keep)
    df = nx.to_pandas_edgelist(net, source='node1', target='node2')
    # print(df.loc[(df['node1'] == 3) & (df['node2'] == 4)])
    return df
    

def draw(df, selection='polymer', group_by=None, color_by=None,
         color_by_list=None, color_sign=False, base_color=(0.75, 0.75, 0.75),
         r=1, edge_norm=None, weight='weight', w1=None, w2=None,
         keep_previous=False, auto_patch=True, label='', threshold=None,
         labeling=None, keep_interfaces=False, save_df=False, cmap_out=None,
         topk=None, to_print=[], cca=False, smaller_max=False, center='n. CA',
         reset_view=True, samewidth=False, induced=None, group_compo=False,
         color_compo=False, girvan_newman=False, dist_func='minus_log',
         plot_betweenness=False, remove_intracomm=False, standard_diff=True,
         cut_diam=3, connect_nodes=None, plot_cca=None,
         impose_palette=None, fix_near_misses=False, group_of_relevance=None,
         save_plot_birch=False, persistent_homology=False):
    """
    draws network on a selection from a pandas DataFrame
    DataFrame should be structured this way:
           \node1 label | node2 label | weight | color | other attributes
    index1
    index2..
    """

    if reset_view:
        view = cmd.get_view()

    if (weight not in df.columns and
       not (w1 in df.columns and w2 in df.columns)):
        raise NameError('Invalid weight.\
                        Weights are {}'.format(', '.join(df.columns[2:])))

    def _auto_patch(nodes, nodes_df):
        print(len(nodes), len(nodes_df))
        if len(nodes) == len(nodes_df):
            print('Auto_patching working (length of lists)')
            if isinstance(nodes_df[0], (int, np.integer)):
                return np.sort(nodes_df)
            else:
                return nodes_df
        else:
            def _cutint(_):
                try:
                    return int(_)
                except BaseException:
                    return str(getnum(_)) + ':' + _.split(':')[-1]
            nodes_intonly = pd.Series(nodes).map(_cutint)
            int2nodes = dict(zip(nodes_intonly, nodes))
            new_nodes = pd.Series(nodes_df).map(_cutint).map(int2nodes)
            if all(np.array(new_nodes) != None):
                print('Auto patching working (indices and chain)')
                return nodes_intonly.index
            else:
                print('Auto patching not working')
                return nodes_df[:len(nodes)]

    def _draw_df(df, label='', base_color=base_color, samewidth=False):
        nodelist = pd.unique(df[['node1', 'node2']].values.ravel('K'))
        objs = []
        for index, row in df.iterrows():
            if samewidth and row[weight] != 0:
                radius = 0.5
            else:
                radius = row[weight] / edge_norm
            if isinstance(row['color'], str):
                color = mpl.colors.to_rgb(row['color'])
            else:
                color = row['color']

            if 'color2' in row:
                if isinstance(row['color2'], str):
                    color2 = mpl.colors.to_rgb(row['color2'])
                else:
                    color2 = row['color2']
            else:
                color2 = color
            objs += [CYLINDER,
                     *node2CA[row['node1']],
                     *node2CA[row['node2']],
                     radius,
                     *color,
                     *color2]

        cmd.load_cgo(objs, '{}edges'.format(label))
        if isinstance(base_color, str):
            base_color = mpl.colors.to_rgb(base_color)
        obj_nodes = [COLOR, *base_color]
        for u in nodelist:
            x, y, z = node2CA[u]
            obj_nodes += [SPHERE, x, y, z, r]
        cmd.load_cgo(obj_nodes, '{}nodes'.format(label))

    if not keep_previous:
        cmd.delete("*edges *nodes")
        cmd.label(selection=selection, expression="")

    # Get correspondance between 3D positions and labels
    if center == "first":
        first_atoms = []
        # Utiliser la fonction iterate pour parcourir tous les atomes
        cmd.iterate(f"({selection})", "first_atoms.append((resi, index))", space=locals())

        # Créer une liste des indices des premiers atomes
        first_atoms_indices = []
        previous_resi = None

        for resi, index in first_atoms:
            if resi != previous_resi:
                first_atoms_indices.append(index)
                previous_resi = resi

        # Créer une sélection à partir des indices des premiers atomes
        selection = "index " + "+".join([str(index) for index in first_atoms_indices])
        raise TypeError(len(first_atoms_indices))
    else:
        selection += " and {}".format(center)
    stored.posCA, stored.resnames, stored.resids, stored.chains = [], [], [],\
                                                                  []

    cmd.iterate_state(1,
                      selector.process(selection),
                      "stored.posCA.append([x,y,z])")
    cmd.iterate(selection, 'stored.resnames.append(resn)')
    cmd.iterate(selection, 'stored.resids.append(resi)')
    cmd.iterate(selection, 'stored.chains.append(chain)')
    nodes = [
        resn +
        resi +
        ':' +
        chain for resn,
        resi,
        chain in zip(
            stored.resnames,
            stored.resids,
            stored.chains)]

    nodes_df = pd.unique(df[['node1', 'node2']].values.ravel('K'))

    if w1 is not None and w2 is not None:
        weight = '{}-{}'.format(w2, w1)
        df[weight] = df[w2] - df[w1]
    df = df.loc[(df[weight] != 0)]  # Is the exception thrown here?
    if not all(node in nodes for node in nodes_df):
        if auto_patch:
            nodes = _auto_patch(nodes, nodes_df)
        else:
            notin = [node for node in nodes_df if node not in nodes]
            loc = (df['node1'].isin(notin)) | (df['node2'].isin(notin))
            df = df.loc[~loc]
    node2CA = dict(zip(nodes, stored.posCA))
    # print(node2CA)

    if group_of_relevance is not None:
        if group_of_relevance > 1:
            n_clusters = group_of_relevance
        else:
            n_clusters = None
        df = cluster_birch(df, weight, save_plot=save_plot_birch, n_clusters=n_clusters)
        group_by = 'cluster'
        df['n_in_cluster'] = df.groupby('cluster')['cluster'].transform(len)
        # print(df.sort_values('n_in_cluster').groupby('cluster', sort=False).size().cumsum())

    # Color by attribute
    if color_by is not None:
        df = _color_by(df, color_by, color_by_list, impose_palette)

    # Color by sign of weight
    elif color_sign:
        if isinstance(color_sign, list):
            color1, color2 = color_sign
        elif color_sign == -1:
            color1, color2 = (0, 0, 1), (1, 0, 0)
        else:
            color1, color2 = (1, 0, 0), (0, 0, 1)

        print('Positive values in {} and negative values in {}'.
              format(color1, color2))

        def weight2color(X): 
            return color1 if X >= 0 else color2

        df.loc[:, 'color'] = df.loc[:, weight].map(weight2color)
        df.loc[:, 'color2'] = df.loc[:, 'color']

    else:
        if 'color' not in df.columns:
            df['color'] = [base_color] * len(df['node1'])


    # Apply threshold/topk/cca on weight
    if isinstance(threshold, (int, float, complex)):
        df = df.loc[df[weight].abs() >= threshold]  # thrown here?
        # print((df[weight].abs() >= threshold).sum())
    elif isinstance(threshold, str):
        if threshold in df.columns:
            df = df.loc[df[weight].abs() >= df[threshold]]  # here?
        else:
            w2, w1 = threshold.split('-')
            df[threshold] = df[w2] - df[w1]
            df = df.loc[df[weight].abs() >= df[threshold].abs()]  # here?
    if topk:
        df = df.loc[df[weight].abs().sort_values(ascending=False).
                    head(n=topk).index]  # here?
        # print(df[weight].abs().min())
    if cca:
        group_compo = True
        if not color_sign:
            color_compo=True
        df = get_cca(df,
                     weight,
                     smaller_max=smaller_max,
                     cut_diam=cut_diam,
                     connect_nodes=connect_nodes,
                     plot_cca=plot_cca)
    

    if persistent_homology:
        df = get_persistent_homology(df,
                                     weight,
                                     alpha=persistent_homology,
                                     dist_func=dist_func)
    

    if girvan_newman:
        if (w1 is None and w2 is None) or standard_diff:
            df = get_girvan_newman(df=df,
                                   weight=weight,
                                   color_compo=color_compo,
                                   dist_func=dist_func,
                                   plot_betweenness=plot_betweenness)
        else:
            _df = get_girvan_newman(df=df.copy(),
                                    weight=w1,
                                    color_compo=color_compo,
                                    dist_func=dist_func,
                                    plot_betweenness=plot_betweenness)

            _df2 = get_girvan_newman(df=df.copy(),
                                     weight=w2,
                                     color_compo=color_compo,
                                     dist_func=dist_func,
                                     plot_betweenness=plot_betweenness)

            weight = '{}-{}'.format(w2, w1)
            _df[weight] = _df2[w2] - _df[w1]

            def f(x, y):
                if x == y:
                    return x
                else:
                    return '{}->{}'.format(x, y)
            vecF = np.vectorize(f)
            loc = (_df['community1'] != _df2['community1'])

            _df['community1'] = pd.DataFrame(vecF(_df['community1'],
                                                  _df2['community1']))
            _df['community2'] = pd.DataFrame(vecF(_df['community2'],
                                                  _df2['community2']))
            df = _df

        group_by = 'community1'
        if remove_intracomm:
            df = df.loc[df['community1'] != df['community2']]
    # Automatic normalization factor
    if edge_norm is None:
        edge_norm = np.max(np.abs(df[weight])) / float(r)
    else:
        edge_norm = float(edge_norm)

    if keep_interfaces:
        if isinstance(keep_interfaces, list):
            print('Keeping only a list of interfaces is not yet implemented')
        else:
            def getchain(X): return str(X[-1])
            df = df.loc[df['node1'].map(getchain) != df['node2'].
                        map(getchain)]

    df = df.loc[df['color'].notna()]
    df = df.loc[df[weight].notna()]

    if induced is not None:
        if isinstance(induced, (str, int)):
            induced = [induced]

        G = nx.from_pandas_edgelist(df,
                                    target='node1',
                                    source='node2',
                                    edge_attr=True)
        subgraph_list = []
        for node in induced:
            if node in G.nodes():
                sg = G.subgraph(nx.node_connected_component(G, node))
                subgraph_list.append(sg)
            else:
                print("{} not in graph nodelist\n", list(G.nodes()))
        if len(subgraph_list) > 0:
            print(subgraph_list)
            G = nx.compose_all(subgraph_list)
            df = nx.to_pandas_edgelist(G, target='node1', source='node2')
            print(', '.join(list(map(str, G.nodes()))))

        else:
            print('Graph empty')


    if group_compo:
        net = nx.from_pandas_edgelist(df,
                                      source="node1",
                                      target="node2",
                                      edge_attr=True)

        if 1 <= fix_near_misses <= 3:
            nbunch = np.sort(list(net.nodes()))
            ebunch = list(net.edges())
            n = fix_near_misses
            # This is the most tricky part, we add very thin edges between
            # covalently bound groups of the protein
            # In a traditional AAN this condition is abs(u-v) == 1 but in
            # CGN we have to divide by the number of groups per residue
            # and include the zero case (same residue)
            edges_to_add = [(u, v, 1e-10)
                            for u, v in zip(nbunch[:-1], nbunch[1:])
                            if abs(u//n-v//n) <= 1 and (u, v) not in ebunch]
            net.add_weighted_edges_from(edges_to_add, weight=weight)

        if connect_nodes is not None:
            nbunch = list(net.nodes())
            ebunch = list(net.edges())
            edges_to_add = connect_edges(nbunch, ebunch, connect_nodes)

            net.add_weighted_edges_from(edges_to_add, weight=weight)

        compo = {
            i: list(c) for i,
            c in enumerate(
                sorted(
                    nx.connected_components(net),
                    key=len,
                    reverse=True))}

        components = np.zeros(len(df))

        for i, l in compo.items():
            ix = np.where(df['node1'].isin(l))[0]
            components[ix] = i + 1

        df['component'] = ['C{}'.format(int(i)) for i in components]
        group_by = 'component'

        if color_compo:
            df = _color_by(df, 'component', color_by_list, impose_palette)
    

    # Draws groups or all or in function of sign of weight
    if group_by is not None:
        grouped = df.groupby(by=group_by)
        for key, loc in grouped.groups.items():
            _draw_df(df.loc[loc],
                     label=key,
                     samewidth=samewidth)
    else:
        if color_sign:
            _draw_df(df.loc[df[weight] >= 0],
                     label='pos_{}'.format(label if label != '' else weight),
                     samewidth=samewidth)
            _draw_df(df.loc[df[weight] < 0],
                     label='neg_{}'.format(label if label != '' else weight),
                     samewidth=samewidth)
        else:
            _draw_df(df, label=label, samewidth=samewidth)

    sel = pd.unique(df[['node1', 'node2']].values.ravel('K'))
    selnodes = ['first (resi {} and chain {})'.
                format(getnum(elt), str(elt).split(':')[-1]) for elt in sel]
    selnodes = ' or '.join(selnodes)

    # Labelling
    if labeling == 1:
        cmd.label(selection=selnodes, expression="oneletter+resi")
    if labeling == 3:
        cmd.label(selection=selnodes, expression="resn+resi")

    if save_df:
        pd.to_pickle(df, save_df)

    if cmap_out is not None:
        net = nx.from_pandas_edgelist(df,
                                      source="node1",
                                      target="node2",
                                      edge_attr=True)
        cmap = nx.to_numpy_array(net, weight=weight)
        np.save(cmap_out, cmap)
    net = nx.from_pandas_edgelist(df,
                                  source="node1",
                                  target="node2",
                                  edge_attr=True)

    if 'ncompos' in to_print:
        print('{} components'.
              format(nx.number_connected_components(net)))
    if 'nedges' in to_print:
        print('{} edges'.
              format(len(net.edges())))

    if 'nnodes' in to_print:
        print('{} nodes'.
              format(len(net.nodes())))

    if reset_view:
        cmd.set_view(view)


def show_mut(sele1, sele2, representation="licorice", color=None,
             label="?mutations"):
    cmd.align(sele2, sele1)
    cmd.hide("everything", label)
    cmd.delete(label)
    sele1_CA = sele1 + ' and name CA'
    sele2_CA = sele2 + ' and name CA'
    stored.res1, stored.res2, stored.resid, stored.chains = [], [], [], []
    cmd.iterate(sele1_CA, 'stored.res1.append(resn)')
    cmd.iterate(sele2_CA, 'stored.res2.append(resn)')
    cmd.iterate(sele2_CA, 'stored.resid.append(resi)')
    cmd.iterate(sele2_CA, 'stored.chains.append(chain)')
    res1, res2, resid, chains = map(np.array,
                                    [stored.res1,
                                     stored.res2,
                                     stored.resid,
                                     stored.chains])
    mutation_indexes = np.where(res1 != res2)
    mutations_resi = resid[mutation_indexes]
    mutations_chains = chains[mutation_indexes]
    selection = ["{} and resi {} and chain {}".format(sele2, resi, chain)
                 for resi, chain in zip(mutations_resi, mutations_chains)]
    cmd.select(label, " or ".join(selection))
    cmd.show_as(representation=representation, selection=label)
    if color:
        cmd.color(color=color, selection=label)
    else:
        cmd.util.cbaw(label)


def draw_df_nodes(df, key="node", weight='weight', colors=['red', 'blue'],
                  base_selection='name N+H', r=1, labeling=False,
                  keep_previous=False, show_unassigned=False):

    df = load_df(df)

    def v2color(X):
        return colors[0] if X >= 0 else colors[1]
    if r == 1:
        r = np.max(df[weight].abs())
    if not keep_previous:
        cmd.hide("spheres", '*')
        cmd.label(selection='all', expression="")
    if show_unassigned:
        cmd.select('temp', base_selection)
        cmd.show("sphere", 'temp')
        cmd.set("sphere_scale", value=0.5, selection="temp")
        cmd.set("sphere_transparency", value=0.5, selection="temp")
        cmd.color('black', selection="temp")
    all_nodes = []
    for row in df.iterrows():
        row = row[1]
        node = row[key]
        selection = "{} and resi {} and chain {}".format(base_selection,
                                                         node[3:-2],
                                                         node[-1])
        cmd.select('temp', selection)
        cmd.show("sphere", "temp")
        cmd.set("sphere_scale", value=row[weight] / r, selection='temp')
        cmd.set("sphere_transparency", value=0, selection="temp")
        cmd.color(v2color(row[weight]), selection="temp")
        all_nodes.append(selection)
    if labeling == 1:
        cmd.label(selection=' or '.join(all_nodes),
                  expression="oneletter+resi")
    if labeling == 3:
        cmd.label(selection=' or '.join(all_nodes),
                  expression="resn+resi")


def continuous_color(df, key="node", weight="weight", w1=None, w2=None,
                     base_selection='name CA', palette='blue_white_red',
                     selection="polymer"):

    df = load_df(df)
    try:
        def to_selection(X):
            return "{} and resi {} and chain {}".\
                format(base_selection, X[3:-2], X[-1])
        nodes = df[key].map(to_selection).values
    except TypeError:
        selection += " and {}".format(base_selection)
        nodes = selection
    if w1 is None and w2 is None:
        scores = df[weight].values
    else:
        scores = df[w2].values - df[w1].values
    _color(scores, nodes, palette=palette)


def draw_shortest_paths(arr_path, k=50, **kwargs):
    paths = pkl.load(open(arr_path, 'rb'))[:k]

    node1, node2, weights = list(), list(), list()
    for i, path in enumerate(paths):
        node1 += path[:-1]
        node2 += path[1:]
        weights += [k - i] * (len(path) - 1)
    df = pd.DataFrame({'node1': node1,
                       'node2': node2,
                       'weight': weights})

    draw(df, **kwargs)


def _color(scores, selection, palette="blue_white_red"):
    stored.scores = iter(scores)
    if isinstance(selection, list):
        selection = ' and name CA or '.join(selection)
    cmd.alter("name CA and (not {})".format(selection), "b=0")
    cmd.alter(selection, "b=next(stored.scores)")
    cmd.spectrum("b", palette=palette, selection="name CA", byres=1)


def continuous_color_from_df(df, source="node1", target="node2",
                             weight='weight', base_selection='name CA',
                             palette='blue_white_red'):
    df = load_df(df)

    def to_selection(X):
        return "{} and resi {} and chain {}".\
            format(base_selection, X[3:-2], X[-1])
    unique_nodes = pd.unique(df[[source, target]].values.ravel('K'))
    selection = list(map(to_selection, unique_nodes))
    scores = []
    for node in unique_nodes:
        loc = ((df[source] == node) | (df[target] == node)) \
            & (df[target] != df[source])
        scores.append(np.sum(df.loc[loc][weight].values))
    _color(scores, selection, palette=palette)


cmd.extend("draw_from_df", draw_from_df)
cmd.extend("draw_from_atommat", draw_from_atommat)
cmd.extend("show_mut", show_mut)
cmd.extend("draw_df_nodes", draw_df_nodes)
cmd.extend("continuous_color", continuous_color)
cmd.extend("continuous_color_from_df", continuous_color)
cmd.extend("draw_shortest_paths", draw_shortest_paths)

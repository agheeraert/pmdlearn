import networkx as nx
import pandas as pd
import warnings
import numpy as np
import pickle as pkl


def minus_log(arr):
    return -np.log(arr)


def remove_zero_edges(G, weight):
    ebunch = [(a, b) for a, b, attrs in G.edges(data=True)
              if attrs[weight] == 0]
    G.remove_edges_from(ebunch)
    return G


def apply_threshold(G, weight, t):
    ebunch = [(a, b) for a, b, attrs in G.edges(data=True)
              if abs(attrs[weight]) >= t]
    G.remove_edges_from(ebunch)
    return G


def prune_nodes(G):
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


class Networks():
    """
    General purpose class to handle one or several networks built using
    Features, MultiFeatures, MDFeaturizer and perform some Network Analysis

    Parameters
    ----------
    df: pandas.DataFrame
    Pandas edgelist with the first column being "node1" and the second "node2"
    representing the node indices in each edge and then other columns are the
    weights in each network represented. 0 are added if the edge is not
    present in a network but this edge isn't considered in network analysis

    Attributes
    ----------
    n_networks: int
    Total number of networks represented by the class

    names: str or list of str
    Names associated to each network
    """

    def __init__(self, df):
        self.n_networks = len(df.columns) - 2
        self.names = df.columns[2:]
        columns = df.columns[:2].tolist()
        if columns != ['node1', 'node2']:
            warnings.warn("Renaming the first columns for compatibility with\
                           other software.")

            df = df.rename(columns={columns[0]: 'node1', columns[1]: 'node2'})
        self.df = df

    def __add__(self, other):
        df = self.df.merge(other.df, on=['node1', 'node2'], how='outer')
        return Networks(df)

    def distances(self, f=minus_log, suffix='_dist', fillna=True):
        """In many cases in proteins a weight between edges increase with the
        contact or correlation. To compute shortest pathways, algorithms needs
        a "short" distance, thus this method builds distances (generally
        using decreasing functions) and returns the distance network from
        all networks in the class

        Attributes
        ---------
        f: function, default = -np.log
        Vectorized function used to compute distances. Generally a decreasing
        function such as -log or the inverse function

        suffix: str, default = '_dist'
        suffix to append to the DataFrame column names
        """
        new_values = f(self.df.values[:, 2:])
        dic = {'node1': self.df['node1'], 'node2': self.df['node2']}
        dic.update({'{}{}'.format(n, suffix): new_values[:, i]
                    for i, n in enumerate(self.names)})
        df = pd.DataFrame(dic)
        if fillna:
            df.fillna(0, inplace=True)
        return Networks(df)

    def betweenness(self, suffix='_b', **kwargs):
        """Computes the betweenness centrality of all edges in the graph
        and returns a new Networks instance with the corresponding to the
        betweenness centrality graphs

        Attribute
        ---------
        suffix: str, default = '_b'
        suffix to append to the DataFrame column names.
        """
        df = None
        for name in self.df.columns[2:]:
            G = nx.from_pandas_edgelist(self.df,
                                        source='node1',
                                        target='node2',
                                        edge_attr=name)
            G = remove_zero_edges(G, name)
            G = prune_nodes(G)
            b = nx.edge_betweenness_centrality(G, weight=name, **kwargs)
            edges = np.array([[u, v] for u, v in b.keys()])
            res1, res2 = edges.T

            _df = pd.DataFrame({'node1': res1,
                                'node2': res2,
                                '{}{}'.format(name, suffix): b.values()})
            if df is None:
                df = _df
            else:
                df = df.merge(_df,
                              on=['node1', 'node2'],
                              how='outer')
        return Networks(df)

    def eigenvector_centrality(self, suffix='_e', **kwargs):
        """Computes the eigenvector centrality of all nodes in the graph
        and returns a pandas.DataFrame instance with corresponding values
        for visualization.

        Attribute
        ---------
        suffix: str, default = '_e'
        suffix to append to the DataFrame column names.
        """
        df = None
        for name in self.df.columns[2:]:
            G = nx.from_pandas_edgelist(self.df,
                                        source='node1',
                                        target='node2',
                                        edge_attr=name)
            G = remove_zero_edges(G, name)
            G = prune_nodes(G)
            ec = nx.eigenvector_centrality(G, weight=name, **kwargs)
            _df = pd.DataFrame({'node': ec.keys(),
                                '{}{}'.format(name, suffix): ec.values()})
            if df is None:
                df = _df
            else:
                df = df.merge(_df, on='node', how='outer')
        return df

    def to_pickle(self, output):
        pkl.dump(self, open(output, 'wb'))

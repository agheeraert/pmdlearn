import numpy as np
import warnings
import pandas as pd
import pickle as pkl
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KDTree
from scipy.special import digamma
from scipy.stats import entropy
from scipy.sparse import coo_matrix
from tqdm import tqdm
from .model import Model

class Features():
    """General purpose class handling all features

    Parameters
    ----------
    values: np.ndarray of size (n_samples, n_features)
    Values taken by the features in each sample

    indices: np.ndarray of size (n_features, n_nodes)
    Indices or labels representing the features (by default the amino acid 
    index). The trivial case of n_nodes = 1 is for intrinsic properties but
    n_nodes = 2 for shared properties that can be represented as graphs and
    n_nodes > 2 for hypergraphs.

    name: string
    Name of the descripted features. Is used to assess how two Features class
    should be concatenated 

    label: string or list of strings
    Label of the descripted system
    
    replica: int or list of ints
    Number of the system replica

    Attributes
    ----------
    n_features: int
    Number of features

    n_samples: int
    Number of samples

    samples_per_label: list of int or None
    If label and replica are lists, this is a list representing how many 
    samples correspond to each label and replica.


    References
    ----------

    Examples
    --------
    """
    def __init__(self, values, indices, name, label, replica,
                 samples_per_label=None):
        self.values = values
        self.n_samples, self.n_features = self.values.shape[:2]
        self.indices = indices
        self.name = name
        self.label = label
        self.replica = replica
        if samples_per_label is None:
            self.samples_per_label = [self.n_samples]
        else:
            self.samples_per_label = samples_per_label

    def __add__(self, other):
        if not self.name == other.name:
            raise TypeError('unsupported operand type(s) for +: Features {} \
                            and Features {}'.format(self.name, other.name))
        else:
            name = self.name
        l1, l2, r1, r2, spl1, spl2 = map(lambda X: X if isinstance(X, list)
                                         else [X],
                                         [self.label,
                                          other.label,
                                          self.replica,
                                          other.replica,
                                          self.samples_per_label,
                                          other.samples_per_label])
        labels = l1 + l2
        replica = r1 + r2
        spl = spl1 + spl2

        if hasattr(self, "values_3d"):
            v1, v2 = self.values_3d, other.values_3d
        else:
            v1, v2 = self.values, other.values

        # Easy case when number of features not fluctuating
        if np.array_equal(self.indices, other.indices):
                indices = self.indices
                values = np.concatenate([v1, v2], axis=0)
        else:
            all_indices = np.concatenate([self.indices, other.indices], axis=0)
            indices, inv = np.unique(all_indices, return_inverse=True,
                                     axis=0)
            values = np.zeros((v1.shape[0] + v2.shape[0],
                              indices.shape[0]))
            values[:self.n_samples, inv[:self.n_features]] = v1
            values[self.n_samples:, inv[self.n_features:]] = v2
        return self.__class__(values, indices, name, labels, replica,
                              samples_per_label=spl)


    def ca(self, kind=PCA, by='replica', **kwargs):
        ca_obj = Model(kind, **kwargs)
        ca_obj.add_labels_features(self._get_labels(by=by), self.indices)
        if kind == TruncatedSVD:
            X_new = ca_obj.ca.fit_transform(coo_matrix(self.values))
        else:
            X_new = ca_obj.ca.fit_transform(self.values)
        ca_obj.scatter_dataframe(X_new)
        return ca_obj
    

    def _get_labels(self, by):
        if type(self.label) != list:
            warnings.warn("This trajectory has only one label.")
            Y = [self.label]*self.samples_per_label[0]
        else:
            if by == 'label':
                labels = self.label
            elif by == 'replica':
                labels = ["{}{}".format(l, r)
                          for l, r in zip(self.label, self.replica)]
            elif by == 'all':
                labels = ['all']*len(self.label)
            Y = []
            for elt, nf in zip(labels, self.samples_per_label):
                Y.extend([elt]*nf)
        return Y

    def average(self, by=['replica', 'label'], weights=None):
        if self.__class__ == MultiFeatures:
            warnings.WarningMessage('Average is not implemented for\
                                     MultiFeatures class. Might work\
                                     unexpectedly')

        if len(self.indices.shape) < 2 or self.indices.shape[1] == 1:
            df = pd.DataFrame({'indices': self.indices.flatten()})
        else:
            df = pd.DataFrame({'node{}'.format(i+1): self.indices[:, i]
                               for i in range(self.indices.shape[1])})

        if type(by) != list:
            by = list(by)
        for _by in by:
            if type(_by) == str:
                labels = self._get_labels(_by)
            elif type(_by) == np.ndarray:
                labels = _by
            else:
                raise TypeError("by argument should be of type string or \
                                np.ndarray")
            for label in pd.unique(labels):
                ix = np.where(np.array(labels, dtype=object) == label)
                if weights is not None:
                    w = weights[ix]
                else:
                    w = None
                _values = np.average(self.values[ix], axis=0, weights=w)
                df[label] = _values
        return df

    def save(self, output):
        pkl.dump(self, open(output, 'wb'))

    def mapping(self, dic):
        """Relabels indexes based on a dictionnary or residue list
        Parameters:
        dic: dictionary or list,
        Contains new labels elements. If dictionary provides older labels.
        If list is mapped to consecutive integers"""
        if isinstance(dic, list):
            dic = dict(enumerate(dic))
        U, inv = np.unique(self.indices, return_inverse=True)
        # Translates atomic contact information in residue contact info
        self.indices = np.array([dic[x] 
                                for x in U])[inv].reshape(self.indices.shape)



class MultiFeatures(Features):
    """General purpose class handling multiple features per node or set of
    nodes. A typical example is the (x, y, z) coordinates for an atom. This 
    class is mainly used to perform correlation analysis.

    Parameters
    ----------
    values: np.ndarray of size (n_samples, n_features)
    Values taken by the features in each sample

    indices: np.ndarray of size (n_features, n_nodes)
    Indices or labels representing the features (by default the amino acid 
    index). The trivial case of n_nodes = 1 is for intrinsic properties but
    n_nodes = 2 for shared properties that can be represented as graphs and
    n_nodes > 2 for hypergraphs.

    name: string
    Name of the descripted features. Is used to assess how two Features class
    should be concatenated 

    label: string or list of strings
    Label of the descripted system
    
    replica: int or list of ints
    Number of the system replica

    descriptor_labels: list of len (n_descriptors)
    Labels for each descriptors
    

    Attributes
    ----------
    values_3d: np.ndarray of size (n_samples, n_features, n_descriptors)
    3D representation of the values best suited for correlation operations.

    n_features: int
    Number of features

    n_samples: int
    Number of samples

    n_descriptors: int
    Number of descriptor per node or set of nodes

    samples_per_label: list of int or None
    If label and replica are lists, this is a list representing how many 
    samples correspond to each label and replica.

    References
    ----------

    Examples
    --------
    """

    def __init__(self, *args, descriptor_labels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_descriptors = self.values.shape[2]
        self.values_3d = self.values
        self.values = self.values.reshape(self.n_samples, -1)
        self.descriptor_labels = descriptor_labels

    def _get_mutual_information(self, estimator='gaussian',
                                by=['label', 'replica']):
        """Computes mutual information between all nodes or set of nodes in 
        for each simulation."""
        mutual_information = {}

        # Separates the computation for each simulation
        if type(by) != list:
            by = list(by)
        for _by in by:
            if type(_by) == str:
                labels = self._get_labels(_by)
            elif type(_by) == np.ndarray:
                labels = _by

            for label in pd.unique(labels):
                ix = np.where(np.array(labels, dtype=object) == label)[0]
                v = self.values_3d[ix]
                # Dispatches the correlation computation between different 
                # estimators
                if estimator == 'gaussian':
                    mutual_information[label] = self._gaussian_estimator(v)
                elif estimator == 'histogram':
                    mutual_information[label] = self._histogram_estimator(v)
                elif estimator[1:4] == 'knn':
                    # Knn estimator has 2 other arguments, the number of nn
                    # and the precise estimator used (1 or 2)
                    k = int(estimator[0])
                    if len(estimator) <= 4:
                        e = 1
                    else:
                        e = int(estimator[4])
                    mutual_information[label] = self._knn_estimator(v, k, e)

        return mutual_information

    def _histogram_estimator(self, values, bins=None):
        """Basic histogram estimator to evaluate mutual information. Usually
        has terrible performance and accuracy."""
        if bins is None:
            bins = tuple([int(np.sqrt(values.shape[0]/5))])*self.n_descriptors
        
        probabilities = np.array([np.histogramdd(values[:, i, :], 
                                  bins=bins)[0]
                                  for i in range(self.n_features)])

        entropies = np.array([entropy(row.flatten()) 
                             for row in probabilities])
        
        j_entropies = np.zeros((self.n_features, self.n_features))
        for i in tqdm(list(range(self.n_features))):
            for j in range(i, self.n_features):
                probs = np.histogramdd(np.concatenate([values[:, i, :],
                                                      values[:, j, :]], 
                                                      axis=-1))[0].flatten()
                j_entropies[[i, j], [j, i]] = entropy(probs)
        mutual_information = np.add.outer(entropies, entropies)
        mutual_information -= j_entropies
        return mutual_information
    
    def _gaussian_estimator(self, values):
        """Gaussian estimator to evaluate mutual information. Finds orthogonal
        correlation but remains a linear estimator. Rapid performance but 
        incomplete."""
        Sx = np.stack([np.cov(elt)
                        for elt in values.transpose(1, 2, 0)],
                        axis=0)
        det = np.log(np.linalg.det(Sx))
        HxHy = np.add.outer(det, det)
        Hxy = np.zeros_like(HxHy)
        for i in tqdm(list(range(0, self.n_features))):
            for j in range(i+1, self.n_features):
                covs = np.cov(np.concatenate([values[:, i, :],
                                              values[:, j, :]],
                                              axis=-1).T)
                _det = np.linalg.det(covs)
                Hxy[[i, j], [j, i]] = _det
        np.fill_diagonal(Hxy, 1)
        Hxy = np.log(Hxy)
        return 1/2*(HxHy - Hxy)

    def _knn_estimator(self, values, k=5, estimate=1, correction=True):
        """Knn estimator to evaluate mutual information. Is intrinsically 
        non-linear but has long computation time."""

        # The precise knn estimator changes one offset in the digamma function
        # and adds a constant in some cases.
        if estimate == 1:
            offset = 1
            const = 0
        if estimate == 2:
            offset = 0
            const = -1/k 
        
        #Initializing mutual information
        mutual_information = np.ones((self.n_features, self.n_features))
        mutual_information *= digamma(k) + digamma(values.shape[0]) + const
        for i in tqdm(list(range(0, self.n_features))):
            for j in range(i, self.n_features):
                x = values[:, i, :]
                y = values[:, j, :]
                points = np.concatenate([x, y], axis=1)
                # Building KDTree with max norm metric
                tree = KDTree(points, metric='chebyshev')
                # Adding an offset to not count self as nearest neighbor
                e = tree.query(points, k=k+1)[0][:, k]
                digamma_nxny = (digamma_n(x, e, offset=offset) +
                                digamma_n(y, e, offset=offset))
                digamma_nxny = digamma_nxny[np.isfinite(digamma_nxny)]
                mutual_information[[i, j], [j, i]] -= np.average(digamma_nxny)
                # print(mutual_information[i, j])
        if correction:
            mutual_information -= np.min(mutual_information)
        return mutual_information


    def generalized_correlation_coefficient(self, cmatrix=None, 
                                            mutual_information=None, 
                                            **kwargs):
        """Computes generalized correlation coefficient"""
        if mutual_information is None:
            mutual_information = self._get_mutual_information(**kwargs)
        df = None
        for label, MI in mutual_information.items():
            rMI = np.sqrt(1-np.exp(-2*MI/(self.n_descriptors)))
            if cmatrix is not None:
                cmatrix = cmatrix.astype(rMI.dtype)
                rMI = np.multiply(rMI, cmatrix)
            indices = rMI.nonzero()
            _df = pd.DataFrame({'node1': indices[0], 'node2': indices[1]})
            _df[label] = rMI[indices]
            if df is None:
                df = _df
            else:
                df = df.merge(_df, on=['node1', 'node2'], how='outer')
        return df


def digamma_n(x, e, offset=1):
    """Computes the average digamma of the number of points between x at a 
    distance e. Used in the k-nn estimator for correlation."""

    tree = KDTree(x, metric='chebyshev')
    # We need to cheat using e-1e-10 so that we don't count kth neighbor 
    # and finding in which dimension(s) the kth neighbor reaches e is too
    # expensive
    num_points = tree.query_radius(x, e-1e-10, count_only=True)

    # Removing a count for self only if it is not null
    num_points = num_points[num_points.nonzero()] - 1

    # There's an extra count in num_points so we take it into account
    num_points += offset
    return digamma(num_points)

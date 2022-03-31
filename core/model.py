from sklearn.decomposition import PCA, SparsePCA
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, ward
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import matplotlib.lines as mlines


class Model():
    """

    """

    def __init__(self, kind=PCA, **kwargs):
        self.ca = kind(**kwargs)
        self.kind = kind

    def add_labels_features(self, labels, indices):
        self.labels = labels
        self.indices = indices

    def scatter_dataframe(self, X_new):
        self.X_new = X_new
        self.n_components = X_new.shape[1]
        df = pd.DataFrame({'C{}'.format(i + 1): component
                           for i, component in enumerate(X_new.T)})
        df['labels'] = self.labels
        self.dataframe = df

    def scatterplot(self, ax=None, x='C1', y='C2', **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        try:
            sns.kdeplot(data=self.dataframe, x=x, y=y, hue=self.labels, ax=ax,
                        common_norm=False, **kwargs)
        except np.linalg.LinAlgError:
            sns.scatterplot(data=self.dataframe, x=x, y=y, hue=self.labels,
                            ax=ax, marker='+', **kwargs)
        try:
            sns.move_legend(ax, loc='lower left',
                            bbox_to_anchor=(0, 1), ncol=4)
        except ValueError:
            pass
        ax.grid('--')

    def get_influences(self):
        n_features = self.indices.shape[0]
        if hasattr(self.ca, 'components_'):
            components = self.ca.components_
        elif hasattr(self.ca, 'eigenvectors_'):
            components = self.ca.eigenvectors_
        else:
            raise TypeError('Class has no component nor \
                            eigenvectors attribute')
        if components.shape[1] == n_features:
            c = components
        else:
            c = components.reshape(self.n_components, n_features, -1)
            c = np.sum(c**2, axis=-1)

        if len(self.indices.shape) < 2 or self.indices.shape[1] == 1:
            dic_labels = {'indices': self.indices.flatten()}
        else:
            dic_labels = {'node{}'.format(i + 1): self.indices[:, i]
                          for i in range(self.indices.shape[1])}
        dic_compo = {'C{}'.format(i + 1): elt for i, elt in enumerate(c)}
        dic_labels.update(dic_compo)
        df = pd.DataFrame(dic_labels)
        return df

    def get_optimal_ward_clusters(self, fig=None):
        links = ward(self.X_new)
        Z_tail = linkage(self.X_new, 'ward')[-10:, 2]
        acceleration = np.diff(Z_tail, 2)
        self.n_optimal_ward = acceleration[::-1].argmax() + 2
        if fig is not None:
            self._plot_optimal(fig, links, Z_tail)
            return fig

    def _plot_optimal(self, fig, links, Z_tail):
        axes = fig.get_axes()
        idxs = np.arange(1, len(Z_tail) + 1)
        lns1 = axes[1].plot(idxs, Z_tail[::-1], label='height', marker='+')
        acceleration = np.diff(Z_tail, 2)
        thresh = Z_tail[self.n_optimal_ward]
        dendrogram(links, no_labels=True, count_sort='descendent', ax=axes[0],
                   color_threshold=thresh + 1)

        ax2 = axes[1].twinx()
        lns2 = ax2.plot(idxs[:-2] + 1, acceleration[::-1],
                        label='acceleration', marker='+', color='r')

        lns = lns1 + lns2
        labs = [ln.get_label() for ln in lns]
        axes[1].legend(lns, labs, loc='best')
        axes[1].set_xlabel('Number of clusters')
        axes[0].set_xlabel('Clusters')
        axes[0].set_ylabel('Height')
        axes[1].set_ylabel('Height')
        ax2.set_ylabel('Height acceleration')

    def plot_ward_clustering(self, ax=None, n_clusters=None, c1=0, c2=1,
                             return_labels=False):

        if ax is None:
            ax = plt.gca()

        if n_clusters is None:
            if not hasattr(self, "n_optimal_ward"):
                self.get_optimal_ward_clusters()
            n_clusters = self.n_optimal_ward

        cluster = AgglomerativeClustering(n_clusters=n_clusters)
        lab = cluster.fit_predict(self.X_new)
        m = []
        i = 1
        self.percentage_cluster = []
        cmap = plt.get_cmap("tab10")
        for i, l in enumerate(pd.unique(lab)):
            ix = np.where(lab == l)[0]
            perc_dict = {_: len(np.intersect1d(ix, np.where(self.labels == _)))
                         for _ in pd.unique(self.labels)}
            ax.scatter(x=self.X_new[ix, c1], y=self.X_new[ix, c2], marker='+',
                       linewidth=1, color=cmap(i))
            m.append(mlines.Line2D([], [], marker='+',
                     markersize=10, label='Cluster {}'.format(i + 1),
                     color=cmap(i)))
            self.percentage_cluster.append(perc_dict)
            i += 1
        # ax.set_aspect('equal')
        ax.set_xlabel('C1')
        ax.set_ylabel('C2')
        ax.grid('--')
        ax.legend(handles=m, loc='best', numpoints=1, ncol=2)
        if return_labels:
            return lab

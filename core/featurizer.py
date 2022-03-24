from enum import unique
import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.dihedrals import Ramachandran
from MDAnalysis.lib.distances import capped_distance, self_capped_distance
import numpy as np
from .features import Features, MultiFeatures
from tqdm import tqdm
from scipy.sparse import coo_matrix
import pandas as pd
import string
import multiprocessing as mp


class MDFeaturizer():
    """General purpose featurizer for MD simulations

    Parameters
    ----------
    traj: str
    Path of the trajectory to analyse

    topo: str
    Path of the trajectory to analyse

    begin: int or None, default = None
    Frame where to begin the analysis.

    end: int or None, default = None
    Frame where to end the analysis.

    stride: int or None, default = None
    Striding of frames.

    relabel_dict: str or dict or None, default = None
    If dict: dictionary to relabel network,
    If str: Path to open the relabeling dictionnary,
    If None: No relabeling (in practice nodes are indexed)

    label: str or None
    Label of the trajectory

    Attributes
    ----------
    n_frames
    n_residues

    """
    def __init__(self, topo, traj=None, begin=None, end=None, stride=None,
                 label=None, replica=None, align=True):
        self.traj = traj
        self.topo = topo
        self.slice = slice(begin, end, stride)
        self.label = label
        self.replica = replica
        if align:
            self.universe = self._align()
        else:
            if self.traj is None:
                self.universe = mda.Universe(self.topo)
            else:
                self.universe = mda.Universe(self.topo, self.traj)

        self.n_frames = len(self.universe.trajectory[self.slice])
        self.n_residues = len(self.universe.residues)

    def _align(self):
        """Aligns trajectories before featurization. Please note this step is
        long and useless when featurizing internal features."""
        if type(self.traj) is not None:
            mobile = mda.Universe(self.topo, self.traj)
            ref = mda.Universe(self.topo, self.traj)
        else:
            mobile = mda.Universe(topology=self.topo)
            ref = mda.Universe(topology=self.topo)
        aligner = align.AlignTraj(mobile, ref,
                                  select="backbone").run()
        return mobile


    def atomic_displacements(self, selection="name CA"):
        """Featurize atomic positions.

        Parameters:
        selection: str
        Selection of atoms to featurize atomic positions"""
        x = [self.universe.select_atoms(selection).positions
             for ts in self.universe.trajectory[self.slice]]
        values = np.stack(x, axis=0)
        avg = np.average(values, axis=0)
        values -= avg
        indices = self.universe.select_atoms(selection).residues.resindices
        return MultiFeatures(values, indices, 'atomic_positions', self.label,
                          self.replica, descriptor_labels=['x', 'y', 'z'])

    def backbone_dihedrals(self, selection="protein"):
        """Featurize phi and psi backbone dihedral angles

        Parameters:
        selection: str
        Selection of atoms to featurize backbone dihedral angles"""

        self.features_name = 'phi_psi'
        r = Ramachandran(self.universe.select_atoms(selection)).run()
        cos = np.cos(np.radians(r.results.angles))
        sin = np.sin(np.radians(r.results.angles))
        values = np.concatenate([cos, sin], axis=2)
        descriptor_labels = ['cos_phi', 'cos_psi', 'sin_phi', 'sin_psi']
        indices = np.stack([r.ag1.residues.resindices,
                            r.ag4.residues.resindices], axis=-1)
        return MultiFeatures(values, indices, "backbone dihedrals", self.label,
                          self.replica, descriptor_labels=descriptor_labels)

    def contacts(self, selection="not name H*", selection2=None, cutoff=5.,
                 prevalence=None, expected=False, entangled=False, 
                 parallel=False):
        """Featurize contacts

        Parameters:
        selection: str, default="not name H*"
        Selection of atoms on which to compute self contacts if selection2 is
        None or contacts with selection 2. Uses MDAnalysis selection commands.
        Default is removing hydrogens

        selection2: str or None, default=None
        If not None, second selection to compute contacts. Uses MDAnalysis 
        selection commands

        cutoff: float, default=5
        Cutoff used to compute interatomic contacts.

        prevalence: None or float, default=None
        if float returns an binary contact matrix of contacts with the given
        prevalence.

        expected: bool, default=False
        Divide the contact number by the counts of atom in each residue.

        entangled: bool, default=False
        Computes an entangled contact network model (i.e. backbone and 
        sidechain) are separated. Experimental.

        parallel: bool or int, default=False
        If 0 or 1, runs a sequential script. If int > 1, runs a parallel 
        script

        """

        self.cutoff = cutoff
        if selection2 is not None:  # Handling asymetric selection case
            self.features_name += '_asym'
            selection += '_{}'.format(selection2)

        # Create dictionnary with atom to residue correspondance
        if not entangled: 
            self.atom2res = {atom.index: atom.residue.resindex
                            for atom in self.universe.atoms}
        else:
            self.atom2res = {atom.index: 2 * atom.residue.resindex
                             for atom in 
                             self.universe.select_atoms('backbone')}
            self.atom2res.update({atom.index: 2* atom.residue.resindex + 1
                                  for atom in 
                                  self.universe.select_atoms('not backbone')})

        # Gets first selection and its indexing in the protein
        self.s1 = self.universe.select_atoms(selection)
        self.ix1 = np.array(self.s1.atoms.indices)

        # If a second selection: retrieves and indexes it
        if selection2 is not None:
            self.s2 = self.universe.select_atoms(selection2)
            self.ix2 = np.array(self.s2.atoms.indices)

        res1, res2, data, times = list(), list(), list(), list()
        
        # Iterates over frames to compute contacts
        if int(parallel) <= 1:
            for _t, ts in enumerate(tqdm(self.universe.trajectory[self.slice])):
                if selection2 is None:
                    pairs = self._contacts_sym()
                else:
                    pairs = self._contacts_asym()
                U, inv = np.unique(pairs, return_inverse=True)
                # Translates atomic contact info in residue contact info
                pairs_res = np.array(
                    [self.atom2res[x] for x in U])[inv].reshape(pairs.shape)
                unique_pairs, counts = np.unique(pairs_res, axis=1,
                                                return_counts=True)
                # Stores information in four different lists
                res1.append(unique_pairs[0])
                res2.append(unique_pairs[1])
                data.append(counts)  # Number of contact
                times.append(np.array(
                    [_t]*unique_pairs.shape[1]))  # Timestep
        else:
            # Checking cpu number to avoid creating too many processes
            n_cpu = min(mp.cpu_count(), parallel)
            with mp.Manager() as manager:
                shared_data = manager.list()
                processes = []
                for _t, ts in enumerate(tqdm(self.universe.trajectory[self.slice])):
                    if selection2 is None:
                        p = mp.Process(target=self._contacts_sym_parallel, 
                                       args=(shared_data, self.s1.positions,_t,))                    
                    else:
                        p = mp.Process(target=self._contacts_asym_parallel, 
                                       args=(shared_data, 
                                             self.s1.positions, 
                                             self.s2.positions, 
                                             _t,))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
                res1 = [t[0] for t in shared_data]
                res2 = [t[1] for t in shared_data]
                data = [t[2] for t in shared_data]
                times = [t[3] for t in shared_data]


        res1, res2, data, times = [np.concatenate(arr) for arr in 
                                   [res1, res2, data, times]]

        contacts = np.stack([res1, res2], axis=-1)

        # Removing intraresidual contacts
        if not entangled:
            to_k = np.where(res1 != res2)[0]
            contacts, times, data = contacts[to_k], times[to_k], data[to_k]

        else: # Removing other intraresidual contacts
            to_k = np.where(res1 // 2 != res2 // 2)[0]
            contacts, times, data = contacts[to_k], times[to_k], data[to_k]

        unique_contacts, inv = np.unique(contacts, axis=0,
                                         return_inverse=True)

        # Saving unique_contacts to re-build contacts name
        values = np.array(coo_matrix((data, (times, inv)),
                          shape=(self.n_frames,
                                 len(unique_contacts))).todense())

        # Creating subfunctions for expected normalization

        def get_div(sel):
            _, counts = np.unique([atom.resindex for atom in sel.atoms],
                                    return_counts=True)
            div = np.zeros((np.max(_)+1))
            div[_] = counts
            return div
        
        def get_div_entangled(sel):
            _b, counts_b = np.unique([atom.resindex for atom in 
                                      sel.select_atoms('backbone').atoms],
                                      return_counts=True)
            _s, counts_s = np.unique([atom.resindex for atom in 
                                      sel.select_atoms('not backbone').atoms],
                                      return_counts=True)
            div = np.zeros((2*np.max(np.concatenate([_b, _s]))+2))
            div[2*_b] = counts_b
            div[2*_s+1] = counts_s
            return div

        # If expected normalization we create the appropriate dividers
        if expected:
            if not entangled:
                div = get_div(self.s1)
            else:
                div = get_div_entangled(self.s1)
            if hasattr(self, "s2"):
                if not entangled:
                    div2 = get_div(self.s2)
                else:
                    div2 = get_div_entangled(self.s2)
                divider = np.outer(div, div2)
            else:
                divider = np.outer(div, div)
            row, col = unique_contacts.T
            expected_values = np.divide(values, divider[row, col])
            feat = Features(expected_values, unique_contacts,
                            'expected_contacts', self.label, self.replica)
        # General case

        else:
            feat = Features(values, unique_contacts, 'contacts', self.label,
                            self.replica)

        # Returning binary contact matrix with given prevalence if asked

        if prevalence is not None:
            binary_contacts = np.zeros_like(values)
            binary_contacts[values.nonzero()] = 1
            ix = np.where(np.sum(binary_contacts, axis=0) >
                          prevalence*self.n_frames)[0]
            cmatrix = np.zeros((self.n_residues, self.n_residues),
                               dtype=np.bool_)
            row, col = unique_contacts[ix].T
            cmatrix[row, col] = 1
            cmatrix[col, row] = 1
            self.cmatrix = cmatrix
            return feat, cmatrix
        
        # General case
        
        else:
            return feat

    def _contacts_sym(self):
        """Uses built-in MDAnalysis function to get interacting pairs in
        symmetric selection"""
        pairs = self_capped_distance(self.s1.positions, self.cutoff,
                                     return_distances=False)
        return np.sort(self.ix1[pairs], axis=1).T

    def _contacts_asym(self):
        """Uses built-in MDAnalysis function to get interacting pairs in
        asymmetric selection"""
        pairs = capped_distance(self.s1.positions, self.s2.positions,
                                self.cutoff, return_distances=False)
        pairs = np.stack([self.ix1[pairs[:, 0]], self.ix2[pairs[:, 1]]])
        # Removing duplicates
        pairs = np.sort(pairs, axis=1)
        pairs = np.unique(pairs, axis=1)
        return pairs

    def _contacts_sym_parallel(self, shared_data, pos, t):
        """Uses built-in MDAnalysis function to get interacting pairs in
        symmetric selection with parallel algrithm"""
        pairs = self_capped_distance(pos, self.cutoff, 
                                    return_distances=False)
        self._parallel_append(shared_data, np.sort(self.ix1[pairs], axis=1).T, t)

    def _contacts_asym_parallel(self, shared_data, pos1, pos2, t):
        pairs = capped_distance(pos1, pos2,
                                self.cutoff, return_distances=False)
        pairs = np.stack([self.ix1[pairs[:, 0]], self.ix2[pairs[:, 1]]])
        # Removing duplicates
        pairs = np.sort(pairs, axis=1)
        pairs = np.unique(pairs, axis=1)

        self._parallel_append(shared_data, pairs, t)

    def _parallel_append(self, shared_data, pairs, t):
        U, inv = np.unique(pairs, return_inverse=True)
        # Translates atomic contact info in residue contact info
        pairs_res = np.array(
            [self.atom2res[x] for x in U])[inv].reshape(pairs.shape)
        unique_pairs, counts = np.unique(pairs_res, axis=1,
                                        return_counts=True)
        shared_data.append((unique_pairs[0], 
                     unique_pairs[1], 
                     counts, 
                     np.array([t]*unique_pairs.shape[1])))


    def get_reslist(self, selection="protein", chain_labels=None):
        """Creates list of residues of the trajectory
        Parameters: 

        selection: str, default="protein"
        Selection to consider as "residues". Uses MDAnalysis selection commands.
        Default is taking the full protein
        
        chain_labels: tuple of str or None
        Custom label for chains. Appended at the end. Default is A,B,C,...
        """
        reslist = list(self.universe.select_atoms(selection).residues)
        resnames = np.array([res.resname for res in reslist])
        resids = np.array([res.resid for res in reslist])
        segids = np.array([res.segid for res in reslist])
        unique_chains = pd.unique(np.array([res.segid for res in reslist]))

        if chain_labels is None:
            chain_labels = list(string.ascii_uppercase)[:len(unique_chains)]
        seg2lab = dict(zip(unique_chains, chain_labels))
        # Crushing older segid
        segids = np.array([seg2lab[res] for res in segids])
        return ['{}{}:{}'.format(resn, resi, seg) 
                for resn, resi, seg in zip(resnames, resids, segids)]

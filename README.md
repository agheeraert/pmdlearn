# pmdlearn
Protein Molecular Dynamics LEARNing 

## Installation
You can install this package using `pip`
```bash 
pip install git+https://github.com/agheeraert/pmdlearn
```
### Requirements
Pip should be able to manage all required dependencies. If necessary, a conda environment file `env.yml` is provided. To create a new environment simply run: 
``` bash
conda create --name <env> --file env.yml 
```
### Getting started
Load the module
```python
import pmdlearn as pm
```
Load a trajectory/topology in the Featurizer. Dynamics are read using MDAnalysis[1]. To load trajectories we simply extend the base `MDAnalysis.Universe` class. For more information about how to load trajectories with MDAnalysis : https://userguide.mdanalysis.org/stable/reading_and_writing.html.
Additional arguments are provided to specify labels for each trajectory (which will be used later during analyses), the `begin` frame, the `stride` and the `end` frame. By default the trajectory is aligned. This is only useful for features which depend on translations and rotations (such as C$\alpha$ coordinates). To disable it simply set `align=False` which can save some computation time.
```python
# Loading the 10000 last frames with a stride of 5 in the reference trajectory  
mdf_reference = pm.MDFeaturizer('topo.prmtop', 'traj_reference.dcd', label="reference", align=False, begin=-10000, stride=5)
# Loading the 10000 last frames with a stride of 5 in the perturbed trajectory  
mdf_perturbed = pm.MDFeaturizer('topo.prmtop', 'traj_perturbed.dcd', label="perturbed", align=False, begin=-10000, stride=5)
# This algorithm also support loading a single structure file
mdf_xray = pm.MDFeaturizer(1GPW.pdb', label="1GPW")
```
To extract the contact features. By default the selection used is all heavy atoms: `"not name H*"`. To see more selection commands please refer to: https://docs.mdanalysis.org/stable/documentation_pages/selections.html.
```python
contacts_ref = mdf_reference.contacts()
contacts_pert = mdf_perturbed.contacts()
```
To concatenate features from different trajectories (which is necessary to run perturbation analysis on multiple trajectories), simply use the $+$ operator. As of now it is absolutely necessary that the considered selections in the different trajectories are perfectly sequentially aligned with no gaps or insertions. Unexpected behavior can arise if not checked properly. 
```python
contacts = contacts_ref + contacts_pert
```
Build a Dynamical Perturbation Contact Network DataFrame and save it for vizualization in PyMOL[2]
```python
dpcn = contacts.average()
dpcn.to_pickle('dpcn.dfp')
```
To get the principal components explaining the variance of contacts and vizualize biplots of trajectories in these principal components.
```python
# Run the signal decomposition. Any sklearn.decomposition class can be use to extract components of the overall signal.
cPCA = contacts.ca(kind=PCA, n_components=10)
# Get the weight vector in the decomposition (eigenvector or feature importance)
influences = cPCA.get_influences()
# Save the eigenvectors for vizualization
influences.to_pickle('cpcn.dfp')
# Saves the biplot of trajectories in PC1 and PC2. This automatically separates trajectories with the original labels. Here the color red is assigned to the first label encountered and the color blue to the second label encountered. To disable this behavior simply set `hue=None"`
cPCA.scatterplot(x='C1', y="C2", palette=('r', 'b'))
plt.savefig('contact_pca.png', transparent=True)
```
### Vizualization
Open PyMoL[2] and load the script with
`run /path/to/pmdlearn/vizualization/draw_network_pymol.py`

You can also add it in your .pymolrc for simplicity: 
```bash
echo "run /path/to/pmdlearn/vizualization/draw_network_pymol.py" >> ~/.pymolrc
```
Using the average contacts over the simulations, given one label you can draw an Amino Acid Network (AAN).
```python
draw_from_df('dpcn.dfp', weight="label")
```
Given two labels you can draw a perturbation network. The sense of subtraction is `w2-w1`:
```python
draw_from_df('dpcn.dfp', w2='holo', w1='apo', color_sign=True)
```
Using the eigenvector/feature importance saved from signal decomposition you can show these components.
```python
draw_from_df('cpcn.dfp', weight='C1', color_sign=True)
# To invert colors (PCA suffers from sign indeterminacy)
draw_from_df('cpcn.dfp', weight='C1', color_sign=-1)
# To use custom colors 
draw_from_df('cpcn.dfp', weight='C1', color_sign=('red', 'gold'))
```
A minimal working example is found in the vizualization folder. Try it with:
```bash
cd vizualization
pymol example.py
```

### Tutorial
In the notebook folder, there is a step by step notebook to reproduce the DPCN and CPCN analyzes for IGPS in [1]
### References
External : 

[1] MDAnalysis:

R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler, D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein. MDAnalysis: A Python package for the rapid analysis of molecular dynamics simulations. In S. Benthall and S. Rostrup, editors, Proceedings of the 15th Python in Science Conference, pages 98-105, Austin, TX, 2016. SciPy, doi:10.25080/majora-629e541a-00e.
N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein. MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations. J. Comput. Chem. 32 (2011), 2319-2327, doi:10.1002/jcc.21787. PMCID:PMC3144279

[2] PyMoL:

Schrödinger, L. & DeLano, W., 2020. PyMOL, Available at: http://www.pymol.org/pymol.


[3] DPCN analysis:  
Gheeraert, Aria, Lorenza Pacini, Victor S. Batista, Laurent Vuillon, Claire Lesieur, and Ivan Rivalta. "Exploring allosteric pathways of a v-type enzyme with dynamical perturbation networks." ***J. Phys. Chem. B*** 123, no. 16 (2019): 3452-3461.  

[4] cPCA analysis:  
Gheeraert, Aria, Laurent Vuillon, Laurent Chaloin, Olivier Moncorgé, Thibaut Very, Serge Perez, Vincent Leroux et al. "Singular Interface Dynamics of the SARS-CoV-2 Delta Variant Explained with Contact Perturbation Analysis." ***J. Chem. Info. Model.***  (2022).  


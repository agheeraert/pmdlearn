# pmdlearn
Protein Molecular Dynamics LEARNing 

## Installation
Using `pip`
```bash 
pip install git+https://github.com/agheeraert/pmdlearn
```
### Requirements
Pip manages all dependencies but a conda environment file is also available in this project root folder. You create a new conda environment with: 
``` bash
conda create --name <env> --file env.yml 
```
### Getting started
Load the module
```python
import pmdlearn as pm
```
Load a trajectory/topology in the Featurizer
```python
# Loading the 10000 last frames with a stride of 5 in the reference trajectory  
mdf_reference = pm.MDFeaturizer('topo.prmtop', 'traj_reference.dcd', label="reference", align=False, begin=-10000, stride=5)
# Loading the 10000 last frames with a stride of 5 in the perturbed trajectory  
mdf_perturbed = pm.MDFeaturizer('topo.prmtop', 'traj_perturbed.dcd', label="perturbed", align=False, begin=-10000, stride=5)
# This algorithm also support loading a single structure file
mdf_xray = pm.MDFeaturizer('code.pdb', label="code")
```
Extract contact features
```python
contacts_ref = mdf_reference.contacts()
contacts_pert = mdf_perturbed.contacts()
```
Add (concatenating) features together
```python
contacts = contacts_ref + contacts_pert
```
Build a Dynamical Perturbation Contact Network DataFrame and saving it for further use
```python
dpcn = contacts.average()
dpcn.to_pickle('dpcn.dfp')
```
Build a Contact PCA
```python
cPCA = contacts.ca(n_components=2)
influences = cPCA.get_influences()
influences.to_pickle('cpcn.dfp')
cPCA.scatterplot(palette=('r', 'b'))
plt.savefig('contact_pca.png', transparent=True)
```
### Vizualization
Add to your .pymolrc the vizualization script: 
```bash
echo "run /path/to/pmdlearn/vizualization/draw_network_pymol.py" >> ~/.pymolrc
```
Then in PyMOL to draw a Amino Acid Network (AAN)
```python
draw_from_df('dpcn.dfp', weight="label")
```
Or a Perturbation Network (make sur to use correct labels)
```python
draw_from_df('dpcn.dfp', w2='holo', w1='apo', color_sign=1)
```
Or a Principal Component Network
```python
draw_from_df('cpcn.dfp', weight='C1', color_sign=1)
# Inverted colors
draw_from_df('cpcn.dfp', weight='C1', color_sign=-1)
# Custom colors
draw_from_df('cpcn.dfp', weight='C1', color_sign=('red', 'gold'))
```
A minimal working example is found in the vizualization folder. Try it with:
```bash
cd vizualization
pymol example.py
```

### Tutorial
In the notebook folder, there is a step by step notebook to reproduce the DPCN and CPCN analyzes for IGPS in Gheeraert, Aria, et al. "Exploring allosteric pathways of a v-type enzyme with dynamical perturbation networks." ***J. Phys. Chem. B.*** 123.16 (2019): 3452-3461.


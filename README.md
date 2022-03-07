# pmdlearn
Protein Molecular Dynamics LEARNing 

## Installation
### Cloning the repository
If using conda, you should cd in your site-packages directory prior e.g.
```bash 
cd ~/anaconda3/lib/python3.7/site-packages
```
Then  clone the repo
```bash
git clone https://github.com/agheeraert/pmdlearn.git
```
### Add module to your python path (without using conda)
Without conda
```bash
echo "export PYTHONPATH=$PYTHONPATH:$(pwd)/pmdlearn" >> ~/.bashrc
```
### Getting started
Open python
```python
from pmdlearn import Model, MDFeaturizer, Features, MultiFeatures
```
Load a trajectory/topology in the Featurizer
```python
Featurizer = MDFeaturizer('topo.prmtop', 'traj.dcd', label="trajectory")
Featurizer = MDFeaturizer('1GPW.pdb', label="1GPW")
```
Extract contact features
```python
contacts = Featurizer.contacts()
```
Adding (concatenating) features together
```python
contacts_apo = Featurizer_apo.contacts()
contacts_holo = Featurizer_holo.contacts()
contacts = contacts_apo + contacts_holo
```
Building a Dynamical Perturbation Contact Network DataFrame
```python
dpcn = contacts.average()
dpcn.to_pickle('dpcn.dfp')
```
Building a Contact PCA
```python
cPCA = contacts.ca(n_components=2)
influences = cPCA.get_influences()
influences.to_pickle('cpcn.dfp')
```
### Vizualization
With PyMOL, add to your .pymolrc
```bash
echo run /path/to/pmdlearn/vizualization/draw_network_pymol.py
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
#Inverted colors
draw_from_df('cpcn.dfp', weight='C1', color_sign=-1)
#Custom colors
draw_from_df('cpcn.dfp', weight='C1', color_sign=('red', 'gold'))
```
A minimal working example is found in the vizualization folder try
```bash
cd vizualization
pymol example.py
```

### Tutorial
In the notebook folder, there is a step by step notebook to reproduce the DPCN and CPCN analyzes


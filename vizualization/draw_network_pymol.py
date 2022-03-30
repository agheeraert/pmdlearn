import enum
from pymol import cmd, stored, selector
from pymol.cgo import *
import pandas as pd
from scipy.sparse.dok import dok_matrix
import matplotlib as mpl
mpl.use('Qt5Agg')
import seaborn as sns 
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import networkx as nx
from networkx.algorithms.community import girvan_newman, modularity
import pickle as pkl

def minus_log(x):
    return -np.log(x)
    

def isfloat(value):
    if type(value) == list:
        value = value[0]
    try:
        float(value)
        return True
    except ValueError:
        return False

def getnum(string):
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
    if sele == None:
        sele = "polymer and not hydrogen"
    sele = "{} and {}".format(sele, top)
    stored.indexlist = []
    cmd.iterate(sele, 'stored.indexlist.append(index)')
    topmat = dok_matrix((len(map_indexes), len(pd.unique(list(map_residues.values())))))
    for id in stored.indexlist:
        topmat[map_indexes[id], map_residues[id]] = 1
    return topmat

def get_cca(df, weight='weight', source='node1', target='node2', cut_diam=3, smaller_max=False, color_compo=False):
    net = nx.from_pandas_edgelist(df.dropna(), source=source, target=target, edge_attr=True)
    net.remove_nodes_from(list(nx.isolates(net)))
    edge_list = sorted(net.edges(data=True), key=lambda t: abs(t[2].get(weight, 1)), reverse=True)
    connected_components = [[nx.number_connected_components(net), 0]]

    while len(net.nodes()) !=0:
        u, v, dic = edge_list.pop()
        net.remove_edge(u, v)
        net.remove_nodes_from(list(nx.isolates(net)))
        connected_components.append([nx.number_connected_components(net), abs(dic.get(weight, 1))])
    connected_components = np.array(connected_components)
    m = np.argmax(connected_components[::-1, 0])
    if smaller_max:
        smaller_max = connected_components[-m, 0] - smaller_max
        threshold = connected_components[np.where(connected_components[:, 0] == smaller_max)[0][-1]][1]
    else:
        threshold = connected_components[-m, 1]
    df = df.loc[df[weight].abs() > threshold]
    net = nx.from_pandas_edgelist(df.dropna(), source=source, target=target, edge_attr=True)
    components_list = [net.subgraph(c).copy() for c in nx.connected_components(net)] 
    if cut_diam > 0:
        robust = [list(c.nodes()) for c in components_list if nx.diameter(c)>=float(cut_diam)]
        net = net.subgraph([x for robust in list(robust) for x in robust])
    components_list = [net.subgraph(c).copy() for c in nx.connected_components(net)] 
    df = nx.to_pandas_edgelist(net, source='node1', target='node2')
    vps = [np.max(np.abs(list(nx.get_edge_attributes(c, weight).values()))) for c in components_list]
    ranking = np.argsort(vps)[::-1]
    if color_compo:
        node2compo = {}
        for i, color in zip(ranking, sns.color_palette("bright", len(ranking))):
            for a in components_list[i]:
                node2compo[a] = color
        df['color'] = df['node1'].map(node2compo)

    return df

def get_girvan_newman(df, weight='weight', source='node1', target='node2', color_compo=True, dist_func=minus_log):
    if dist_func is not None:
        df['_{}'.format(weight)] = dist_func(df[weight])
        old_weights = df[weight]
        net = nx.from_pandas_edgelist(df.dropna(), source=source, target=target, edge_attr='_{}'.format(weight))
    else:
        net = nx.from_pandas_edgelist(df.dropna(), source=source, target=target, edge_attr=weight)
    net.remove_nodes_from(list(nx.isolates(net)))
    comp = girvan_newman(net)
    max_modularity = 0
    for communities in comp:
        mod = modularity(net, communities, weight=weight)
        if mod >= max_modularity:
            out_communities = communities
            max_modularity = mod
    communities_list = [nx.subgraph(net, c).copy() for c in out_communities]
    if color_compo:
        n_colors = len(communities_list)
        if len(communities_list) <= 10:
            palette = sns.color_palette("bright", n_colors=n_colors)
        else:
            palette = sns.color_palette("husl", n_colors=n_colors)
        i2color = dict(enumerate(palette))
        node2compo = {}
        df = nx.to_pandas_edgelist(net, source=source, target=target)
        for i, c in enumerate(communities_list):
            for a in c:
                node2compo[a] = i
        df['community'] = df[source].map(node2compo)
        df['color'] = df['community'].map(i2color)
        df['community'] = df['community'].map(lambda i: 'C{}'.format(i+1))
    if dist_func is not None:
        df[weight] = old_weights
    return df


def draw_Network(path, reset_view=True, hide_nodes=True, **kwargs):
    G = pkl.load(open(path, 'rb'))
    view = cmd.get_view()
    draw(G.df, **kwargs)
    if reset_view:
        cmd.set_view(view)
    if hide_nodes:
        cmd.disable('*nodes')

def draw_from_df(path, reset_view=True, hide_nodes=True, **kwargs):
    view = cmd.get_view()
    df = pd.read_pickle(path)
    draw(df, **kwargs)
    if reset_view:
        cmd.set_view(view)
    if hide_nodes:
        cmd.disable('*nodes')

def draw_from_atommat(path, perturbation=None, sele=None, sele1=None, sele2=None, top=None, top_perturbation=None,
                      norm_expected=False, average_with=None, **kwargs):
    
    def _get_resmat(filepath, topo):
        load_mat = lambda X: load_npz(X) if X.split('.')[-1] == 'npz' else csr_matrix(np.load(X))
        mat = load_mat(filepath)
        stored.top_indexes, stored.top_residues = [], []
        cmd.iterate(topo, 'stored.top_indexes.append(index)')
        cmd.iterate(topo, 'stored.top_residues.append(resi)')
        resid_list = []
        k=0
        for elt1, elt2 in zip(stored.top_residues[:-1], stored.top_residues[1:]):
            resid_list.append(k)
            if elt1 != elt2:
                k+=1
        resid_list.append(k)
        map_indexes = dict(zip(stored.top_indexes, range(len(stored.top_indexes))))
        map_residues = dict(zip(stored.top_indexes, resid_list))
        top1 = create_topmat(sele1, topo, map_indexes, map_residues)
        top2 = create_topmat(sele2, topo, map_indexes, map_residues)
        resmat = (mat @ top1).transpose() @ top2
        resmat.setdiag(0)
        resmat.eliminate_zeros()
        if norm_expected:
            expected = (top1.sum(axis=1).transpose() @ top1).transpose() @ (top2.sum(axis=1).transpose() @ top2)
            resmat /= expected
            resmat[np.isnan(resmat)] = 0   
            resmat = csr_matrix(resmat)
        return resmat

    sele1 = "not hydrogen" if sele1 == None else sele1
    sele2 = "not hydrogen" if sele2 == None else sele2    

    if top == None:
        top = '{}* and polymer'.format(path.split('.')[0].split('_')[0])
    
    if sele1 == None and sele2 == None:
        sele1, sele2 = sele, sele


    resmat = _get_resmat(path, top)
    if perturbation !=None:
        if top_perturbation == None:
            top_perturbation = '{}* and polymer'.format(perturbation.split('.')[0].split('_')[0])
        resmat2 = _get_resmat(perturbation, top_perturbation)
        resmat = resmat2 - resmat

    elif average_with != None:
        if type(average_with) == str:
            average_with = [average_with]
        n_dynamics = len(average_with) + 1
        top_list = ['{}* and polymer'.format(filepath.split('.')[0].split('_')[0]) for filepath in average_with]
        resmat_list = [_get_resmat(_1, _2) for _1, _2 in zip(average_with, top_list)]
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
    nodes = pd.unique([resn+resi+':'+chain for resn, resi, chain in zip(stored.resnames, stored.resids, stored.chains)])
    id2node = dict(enumerate(nodes))
    df = nx.to_pandas_edgelist(nx.from_scipy_sparse_matrix(resmat))
    df.columns=['node1', 'node2', 'weight']
    df['node1'] = df['node1'].map(id2node)
    df['node2'] = df['node2'].map(id2node)

    draw(df, selection=top, **kwargs)
    

def draw(df, selection='polymer', group_by=None, color_by=None, color_by_list=None, color_sign=False, base_color=(0.75, 0.75, 0.75), r=1, 
                edge_norm=None, weight='weight', w1=None, w2=None, keep_previous=False, auto_patch=True, label='', threshold=None, labeling=None, 
                keep_interfaces=False, save_df=False, cmap_out=None, topk=None, to_print=[], cca=False, smaller_max=False, center='n. CA',
                reset_view=True, samewidth=False, induced=None, group_compo=False, color_compo=False, girvan_newman=False, dist_func=minus_log):
    """
    draws network on a selection from a pandas DataFrame
    DataFrame should be structured this way:
           \node1 label | node2 label | weight | color | other attributes
    index1
    index2..
    the group_by attribute allows to separate the drawing in different groups
    """

    if reset_view:
        view = cmd.get_view()

    if weight not in df.columns and not (w1 in df.columns and w2 in df.columns):
        raise NameError('Invalid weight. Valid weights are {}'.format(', '.join(df.columns[2:])))

    def _auto_patch(nodes, nodes_df):
        print(len(nodes), len(nodes_df))
        if len(nodes) == len(nodes_df):
            print('Auto_patching working (length of lists)')
            return nodes_df
        else:
            def _cutint(_):
                try:
                    return int(_)
                except:
                    return str(getnum(_))+':'+_.split(':')[-1]
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
                radius = row[weight]/edge_norm
            if type(row['color']) == str:
                row['color'] = mpl.colors.to_rgb(row['color'])
            objs+=[CYLINDER, *node2CA[row['node1']], *node2CA[row['node2']], radius, *row['color'], *row['color']]
        cmd.load_cgo(objs, '{}edges'.format(label))
        if type(base_color) == str:
            base_color = mpl.colors.to_rgb(base_color) 
        obj_nodes = [COLOR, *base_color]
        for u in nodelist:
            x, y, z = node2CA[u]
            obj_nodes += [SPHERE, x, y, z, r]
        cmd.load_cgo(obj_nodes, '{}nodes'.format(label))

    if not keep_previous:
        cmd.delete("*edges *nodes")
        cmd.label(selection=selection, expression="")


    #Get correspondance between 3D positions and labels
    selection += " and {}".format(center)
    stored.posCA, stored.resnames, stored.resids, stored.chains = [], [], [], []
    cmd.iterate_state(1, selector.process(selection), "stored.posCA.append([x,y,z])")
    cmd.iterate(selection, 'stored.resnames.append(resn)')
    cmd.iterate(selection, 'stored.resids.append(resi)')
    cmd.iterate(selection, 'stored.chains.append(chain)')
    nodes = [resn+resi+':'+chain for resn, resi, chain in zip(stored.resnames, stored.resids, stored.chains)]
    nodes_df = pd.unique(df[['node1', 'node2']].values.ravel('K'))


    if type(w1) != type(None) and type(w2) !=type(None):
        weight = '{}-{}'.format(w2, w1)
        df[weight] = df[w2] - df[w1]
    df = df.loc[df[weight] != 0]
    if not all(node in nodes for node in nodes_df):
        if auto_patch:
            nodes = _auto_patch(nodes, nodes_df)
        else: 
            notin = [node for node in nodes_df if node not in nodes]
            loc = (df['node1'].isin(notin)) | (df['node2'].isin(notin))
            df = df.loc[~loc]
    node2CA = dict(zip(nodes, stored.posCA))


    #Color by attribute
    if color_by is not None:
        attributes = pd.unique(df[color_by])
        n_colors = len(attributes)
        if color_by_list:
            palette = color_by_list
            print(''.join('{} colored in {}; '.format(u, v)  for u, v in zip(attributes, palette)))
        else:
            palette = sns.color_palette("bright", n_colors)
        attr2color = dict(zip(attributes, palette))
        df['color'] = df[color_by].map(attr2color)

    #Color by sign of weight
    elif color_sign: 
        if type(color_sign) == list:
            color1, color2 = color_sign
        elif color_sign == -1:
            color1, color2 = (0, 0, 1), (1, 0, 0)
        else:
            color1, color2 = (1, 0, 0), (0, 0, 1)
        
        print('Positive values in {} and negative values in {}'.format(color1, color2))
        weight2color = lambda X: color1 if X >= 0 else color2
        df['color'] = df[weight].map(weight2color)
    else:
        df['color'] = [base_color]*len(df['node1'])

    #Automatic normalization factor
    if edge_norm == None:
        edge_norm = np.max(np.abs(df[weight]))/float(r)
    else:
        edge_norm = float(edge_norm)

    #Apply threshold/topk/cca on weight
    if isinstance(threshold, (int, float, complex)):
        df = df.loc[df[weight].abs() >= threshold]
    elif isinstance(threshold, str):
        if threshold in df.columns:
            df = df.loc[df[weight].abs() >= df[threshold]]
        else:
            w2, w1 = threshold.split('-')
            df[threshold] = df[w2] - df[w1]
            df = df.loc[df[weight].abs() >= df[threshold].abs()]
    if topk:
        df = df.loc[df[weight].abs().sort_values(ascending=False).head(n=topk).index]
    if cca:
        df = get_cca(df, weight, smaller_max=smaller_max, color_compo=color_compo)
        
    if girvan_newman:
        df = get_girvan_newman(df, weight, color_compo=color_compo, dist_func=dist_func)
        group_by = 'community'
        print(df)

    if keep_interfaces:
        if type(keep_interfaces) == list:
            print('Keeping only a list of interfaces not yet implemented')
        else:
            getchain = lambda X: str(X[-1])
            df = df.loc[df['node1'].map(getchain) != df['node2'].map(getchain)]

    df = df.loc[df['color'].notna()] 
    df = df.loc[df[weight].notna()]

    if induced is not None:
        if isinstance(induced, str):
            induced = [induced]
        G = nx.from_pandas_edgelist(df, target='node1', source='node2', edge_attr=True)
        G = nx.compose_all([G.subgraph(nx.node_connected_component(G, node)) for node in induced if node in G.nodes()])
        df = nx.to_pandas_edgelist(G, target='node1', source='node2')
        print(list(G.nodes()))

    if group_compo:
        net = nx.from_pandas_edgelist(df, source="node1", target="node2", edge_attr=True)
        compo = {i: list(c) for i, c in enumerate(sorted(nx.connected_components(net), key=len, reverse=True))}
        components = np.zeros(len(df))
        for i, l in compo.items():
            ix = np.where(df['node1'].isin(l))[0]
            components[ix] = i+1
        df['component'] = ['C{}'.format(int(i)) for i in components]
        group_by = 'component'
            

    #Draws groups or all or in function of sign of weight
    if group_by != None:
        groups = pd.unique(df[group_by])
        for group in groups:
            _draw_df(df.loc[df[group_by]==group], label=group, samewidth=samewidth)
    else:
        if color_sign:
            _draw_df(df.loc[df[weight]>=0], label='pos_{}'.format(label if label != '' else weight), samewidth=samewidth)
            _draw_df(df.loc[df[weight]<0], label='neg_{}'.format(label if label != '' else weight), samewidth=samewidth)
        else:
            _draw_df(df, label=label, samewidth=samewidth)

    sel = pd.unique(df[['node1', 'node2']].values.ravel('K'))
    selnodes = ['first (resi {} and chain {})'.format(getnum(elt), str(elt).split(':')[-1]) for elt in sel]
    selnodes = ' or '.join(selnodes)
    #Labeling
    if labeling==1:
        cmd.label(selection=selnodes, expression="oneletter+resi")
    if labeling==3:
        cmd.label(selection=selnodes, expression="resn+resi")

    if save_df:
        pd.to_pickle(df, save_df)

    if cmap_out != None:
        net = nx.from_pandas_edgelist(df, source="node1", target="node2", edge_attr=True)
        cmap = nx.to_numpy_array(net, weight=weight)
        np.save(cmap_out, cmap)
    
    if 'ncompo' in to_print:
        net = nx.from_pandas_edgelist(df, source="node1", target="node2", edge_attr=True)
        print('Number of components {}'.format(nx.number_connected_components(net)))
    
    if reset_view:
        cmd.set_view(view)


def show_mut(sele1, sele2, representation="licorice", color=None, label="?mutations"):
    cmd.align(sele2, sele1)
    cmd.hide("everything", label)
    cmd.delete(label)
    sele1_CA = sele1 + ' and name CA'
    sele2_CA = sele2 +' and name CA'
    stored.res1, stored.res2, stored.resid, stored.chains = [], [], [], []
    cmd.iterate(sele1_CA, 'stored.res1.append(resn)')
    cmd.iterate(sele2_CA, 'stored.res2.append(resn)')
    cmd.iterate(sele2_CA, 'stored.resid.append(resi)')
    cmd.iterate(sele2_CA, 'stored.chains.append(chain)')
    res1, res2, resid, chains = map(np.array, [stored.res1, stored.res2, stored.resid, stored.chains])
    mutation_indexes = np.where(res1 != res2)
    mutations_resi = resid[mutation_indexes]
    mutations_chains = chains[mutation_indexes]
    selection = ["{} and resi {} and chain {}".format(sele2, resi, chain) for resi, chain in zip(mutations_resi, mutations_chains)]
    cmd.select(label, " or ".join(selection))
    cmd.show_as(representation=representation, selection=label)
    if color:
        cmd.color(color=color, selection=label)
    else:
        cmd.util.cbaw(label)

def draw_df_nodes(df, key="node", weight='weight', colors=['red', 'blue'], base_selection='name N+H', r=1, labeling=False, keep_previous=False, show_unassigned=False):
    df = pd.read_pickle(df)
    v2color = lambda X: colors[0] if X >= 0 else colors[1]
    if r==1:
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
        selection="{} and resi {} and chain {}".format(base_selection, node[3:-2], node[-1])
        cmd.select('temp', selection)
        cmd.show("sphere", "temp")
        cmd.set("sphere_scale", value=row[weight]/r, selection='temp')
        cmd.set("sphere_transparency", value=0, selection="temp")
        cmd.color(v2color(row[weight]), selection="temp")
        all_nodes.append(selection)
    if labeling==1:
        cmd.label(selection=' or '.join(all_nodes), expression="oneletter+resi")
    if labeling==3:
        cmd.label(selection=' or '.join(all_nodes), expression="resn+resi")

def continuous_color(df, key="node", weight="weight", w1=None, w2=None, base_selection='name CA', palette='blue_white_red', selection="polymer"):
    df = pd.read_pickle(df)
    try:
        to_selection = lambda X: "{} and resi {} and chain {}".format(base_selection, X[3:-2], X[-1])
        nodes = df[key].map(to_selection).values
    except TypeError:
        selection += " and {}".format(base_selection)
        nodes = selection
    if w1 == None and w2 == None:
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
        weights += [k-i]*(len(path)-1)
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

def continuous_color_from_df(df, source="node1", target="node2", weight='weight', base_selection='name CA', palette='blue_white_red'):
    df = pd.read_pickle(df)
    to_selection = lambda X: "{} and resi {} and chain {}".format(base_selection, X[3:-2], X[-1])
    # net = nx.from_pandas_edgelist(df.dropna(), source=source, target=target, edge_attr=weight)
    # mat = nx.to_numpy_array(net, weight=weight)
    # for i in range(mat.shape[0]): mat[i, i] = 0
    # print(mat) 
    # print(mat.nonzero())
    # scores = np.array(np.sum(mat, axis=1)/2).squeeze()
    unique_nodes = pd.unique(df[[source, target]].values.ravel('K'))
    selection = list(map(to_selection, unique_nodes))
    scores = []
    for node in unique_nodes:
        print(df[target] != df[source])
        loc = ((df[source] == node) | (df[target] == node)) & (df[target] != df[source])
        print(df.loc[loc])
        scores.append(np.sum(df.loc[loc][weight].values))
    print(scores)
    _color(scores, selection, palette=palette)

#    nodelist = pd.unique(df[['node1', 'node2']].values.ravel('K'))
    


cmd.extend("draw_from_df", draw_from_df)
cmd.extend("draw_from_atommat", draw_from_atommat)
cmd.extend("show_mut", show_mut)
cmd.extend("draw_df_nodes", draw_df_nodes)
cmd.extend("continuous_color", continuous_color)
cmd.extend("continuous_color_from_df", continuous_color)
cmd.extend("draw_shortest_paths", draw_shortest_paths)
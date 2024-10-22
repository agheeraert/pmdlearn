import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pymol import cmd
import numpy as np
from pymol.cgo import CYLINDER, COLOR, SPHERE, CONE

# Function to load DataFrame from a given path
def load_network_data(path):
    """
    Load the network data from a pickled file into a Pandas DataFrame.
    """
    df = pd.read_pickle(path)  # Adjust separator if necessary (e.g., comma or tab)
    return df

# Function to process and visualize the network in PyMOL
def visualize_network(df, weight_column, threshold, label, colors=[(0, 0, 1), (1, 0, 0)], normalization_factor=None, keep_interface=False, relative_threshold=False):
    """
    Visualize the network in PyMOL using CGO objects:
    - Positive edges as blue cylinders
    - Negative edges as red cylinders
    - Nodes as gray spheres

    Arguments:
    df -- Pandas DataFrame containing the network data
    weight_column -- The column to use for edge weight filtering
    threshold -- Minimum absolute value of weight for edges to be included
    colors -- List of colors for positive and negative edges
    normalization_factor -- If given, this will be used to normalize the cylinder radius, otherwise calculated from max weight
    """
    # Filter the DataFrame by weight threshold
    if relative_threshold:
        threshold = float(threshold)
        if not 0<=threshold<=1:
            raise ValueError("Relative threshold must be between 0 and 1 ({threshold}).")
        threshold = threshold*df[weight_column].abs().max()
    df_filtered = df.query(f'abs({weight_column}) >= {threshold}').copy()

    if keep_interface:
        loc = df_filtered['source'].str.split(':').str[-1] != df_filtered['target'].str.split(':').str[-1]
        df_filtered = df_filtered.loc[loc].copy()

    # Get unique nodes from the filtered DataFrame
    unique_nodes = set(df_filtered['source']).union(set(df_filtered['target']))

    # Dictionary to store coordinates of each node
    node_coords = {}

    # Iterate through unique nodes and fetch their coordinates
    for node in unique_nodes:
        resname, resid, chain = node.split(":")
        selection = f"resname {resname} and resid {resid} and chain {chain}"

        # Get the model for the current selection
        atoms = cmd.get_model(selection)

        # If the selection is not empty, save the coordinates of the first atom or CA
        if len(atoms.atom) > 0:
            atom = atoms.atom[0]  # Select the first atom (or CA if needed)
            node_coords[node] = atom.coord

    # Determine the max absolute weight if normalization_factor is not provided
    if normalization_factor is None:
        max_weight = df_filtered[weight_column].abs().max()
        normalization_factor = max_weight

    # Create CGO objects for positive and negative edges, and nodes
    positive_edges = []
    negative_edges = []
    nodes = []

    # Iterate through the filtered DataFrame rows
    for _, row in df_filtered.iterrows():
        source = row['source']
        target = row['target']
        weight = row[weight_column]

        # Ensure both nodes have associated coordinates
        if source in node_coords and target in node_coords:
            coord1 = node_coords[source]
            coord2 = node_coords[target]

            # Calculate the radius based on weight and normalization factor
            radius = abs(weight) / normalization_factor

            # Assign colors based on the weight
            if weight > 0:
                color = colors[0]  # Positive weight color
                # Create the cylinder (edge) between source and target
                positive_edges += [CYLINDER, *coord1, *coord2, radius, *color, *color]
            else:
                color = colors[1]  # Negative weight color
                # Create the cylinder (edge) between source and target
                negative_edges += [CYLINDER, *coord1, *coord2, radius, *color, *color]



    # Add nodes as gray spheres (gray spheres by default)
    for node, coord in node_coords.items():
        nodes += [COLOR, 0.8, 0.8, 0.8]  # Gray color for nodes
        nodes += [SPHERE, *coord, 1.0]  # 1.0 as default radius for spheres

    # Load the CGO objects into PyMOL
    cmd.load_cgo(positive_edges, f"{label}_pos_edges")
    cmd.load_cgo(negative_edges, f"{label}_neg_edges")
    cmd.load_cgo(nodes, f"{label}_nodes")


def get_cca(df, weight, cut_diam=3):
    net = nx.from_pandas_edgelist(df.dropna(), edge_attr=weight)
    net.remove_nodes_from(list(nx.isolates(net)))
    edge_list = sorted(net.edges(data=True),
                       key=lambda t: abs(t[2].get(weight, 1)), reverse=True)
    connected_components = [[nx.number_connected_components(net), 0]]
    while len(edge_list) != 0:
        u, v, dic = edge_list.pop()
        net.remove_edge(u, v)
        net.remove_nodes_from(list(nx.isolates(net)))
        cc = [nx.number_connected_components(net), abs(dic.get(weight, 1))]
        connected_components.append(cc)
    connected_components = np.array(connected_components)
    m = np.argmax(connected_components[::-1, 0])
    threshold = connected_components[-m, 1]
    df = df.loc[df[weight].abs() > threshold]
    net = nx.from_pandas_edgelist(df.dropna(), edge_attr=True)
    components_list = [net.subgraph(c).copy()
                       for c in nx.connected_components(net)]
    if int(cut_diam) > 0:
        robust = [list(c.nodes())
                  for c in components_list
                  if nx.diameter(c) >= float(cut_diam)]
        net = net.subgraph([x for robust in list(robust) for x in robust])
    components_list = [net.subgraph(c).copy()
                       for c in nx.connected_components(net)]
    
    df = nx.to_pandas_edgelist(net)
    components = [nx.to_pandas_edgelist(net) for net in components_list]
    return df, components

# PyMOL command to load and visualize the network directly from the script
def draw_network(path, weight='weight', threshold=0, relative_threshold=False, colors=[(0, 0, 1), (1, 0, 0)], normalization_factor=None, keep_previous=False, label=None, keep_interface=False,
        cca=False, cut_diam=3, group_cca=False):
    """
    PyMOL command to load the network data from a file and visualize it in PyMOL.

    Arguments:
    path -- Path to the network data file (CSV)
    weight_column -- The column to use for edge weight filtering (default: 'PC1')
    threshold -- Minimum absolute value of weight for edges to be included (default: 0.05)
    colors -- List of RGB colors for positive and negative edges (default: blue for positive, red for negative)
    normalization_factor -- If provided, it will normalize the edge radius; otherwise calculated from max weight
    """
    if not keep_previous:
        cmd.delete("*edges")
        cmd.delete("*nodes")
    if label is None:
        label = weight
    view = cmd.get_view()
    df = load_network_data(path)

    if normalization_factor is None:
        normalization_factor = df[weight].abs().max()
    if cca:
        df, components = get_cca(df, weight=weight, cut_diam=cut_diam)
        if group_cca:
            for i, component in enumerate(components, start=1):
                visualize_network(component, weight, 0, f"{label}_CC{i}", colors, normalization_factor, keep_interface=False, relative_threshold=False)
        else: 
            visualize_network(df, weight, 0, label, colors, normalization_factor, keep_interface, relative_threshold)
    else:
        visualize_network(df, weight, threshold, label, colors, normalization_factor, keep_interface, relative_threshold)
    cmd.set_view(view)

# Extend the pymol command to be callable directly within PyMOL
cmd.extend("draw_network", draw_network)

# Function to create a color based on the value between -1 and 1
def get_color_from_value(value, vmin=-1, vmax=1):
    """
    Map a value between -1 and 1 to a color using a blue-white-red scale.
    """
    norm_value = (value - vmin) / (vmax - vmin)  # Normalize the value between 0 and 1
    color = plt.cm.Reds(norm_value)  # Blue-White-Red colormap from matplotlib
    return color[:3]  # Return only RGB values (ignore alpha)

# Function to visualize nodes with different radii and optional color scaling
def visualize_nodes(df, size_column, label, normalization_factor=None, use_color_scale=True):
    """
    Visualize the nodes in PyMOL using CGO spheres with varying radii and optional color scaling.

    Arguments:
    df -- Pandas DataFrame containing the node data
    size_column -- The column to use for node size (radius)
    label -- Label for the PyMOL object
    normalization_factor -- Normalization factor for the radius (default: None)
    use_color_scale -- Whether to apply color scale based on a value between -1 and 1 (default: True)
    """
    # Get unique nodes from the DataFrame
    unique_nodes = df['source'].unique()

    # Dictionary to store coordinates of each node
    node_coords = {}

    if normalization_factor is None:
        normalization_factor = df[size_column].max()/2

    # Iterate through unique nodes and fetch their coordinates
    for node in unique_nodes:
        resname, resid, chain = node.split(":")
        selection = f"resname {resname} and resid {resid} and chain {chain}"

        # Get the model for the current selection
        atoms = cmd.get_model(selection)

        # If the selection is not empty, save the coordinates of the first atom or CA
        if len(atoms.atom) > 0:
            atom = atoms.atom[0]  # Select the first atom (or CA if needed)
            node_coords[node] = atom.coord

    # Create CGO object for nodes
    nodes = []

    # Iterate through the DataFrame rows
    for _, row in df.iterrows():
        node = row['source']  # Assuming 'source' column has the node name
        if node in node_coords:
            coord = node_coords[node]

            # Calculate the radius based on the size_column and normalization factor
            radius = row[size_column] / normalization_factor

            # Determine the color (either from a color scale or default gray)
            if use_color_scale:
                color = get_color_from_value(radius)
            else:
                color = (0.8, 0.8, 0.8)  # Default gray color

            # Add the node (sphere) to the CGO object
            nodes += [COLOR, *color]  # Apply color
            nodes += [SPHERE, *coord, radius]  # Apply position and radius

    # Load the CGO object into PyMOL
    cmd.load_cgo(nodes, f"{label}_nodes")

# PyMOL command to load and visualize nodes directly from the script
def draw_nodes(path, size='size', normalization_factor=None, use_color_scale=True, label=None):
    """
    PyMOL command to load node data from a file and visualize nodes with varying radii and optional color scaling.

    Arguments:
    path -- Path to the network data file (CSV or pickle)
    size -- The column to use for node size (radius) (default: 'size')
    normalization_factor -- Normalization factor for the radius (default: 1.0)
    use_color_scale -- Whether to apply color scale (default: True)
    """
    if label is None:
        label = size
    view = cmd.get_view()
    df = load_network_data(path)
    visualize_nodes(df, size, label, normalization_factor, use_color_scale)
    cmd.set_view(view)

# Extend the pymol command to be callable directly within PyMOL
cmd.extend("draw_nodes", draw_nodes)

# Function to visualize arrows based on direction vectors
def visualize_arrows(df, label, weight="weight", normalization_factor=None, use_color_scale=True):
    """
    Visualize the nodes with arrows in PyMOL using CGO arrows with varying lengths and optional color scaling.

    Arguments:
    df -- Pandas DataFrame containing the node data
    label -- Label for the PyMOL object
    use_color_scale -- Whether to apply color scale based on vector magnitude (default: True)
    """
    # Get unique nodes from the DataFrame
    unique_nodes = df['source'].unique()

    # Dictionary to store coordinates of each node
    node_coords = {}

    # Iterate through unique nodes and fetch their coordinates
    for node in unique_nodes:
        resname, resid, chain = node.split(":")
        selection = f"resname {resname} and resid {resid} and chain {chain}"

        # Get the model for the current selection
        atoms = cmd.get_model(selection)

        # If the selection is not empty, save the coordinates of the first atom (or CA)
        if len(atoms.atom) > 0:
            atom = atoms.atom[0]  # Select the first atom (or CA if needed)
            node_coords[node] = atom.coord

    if normalization_factor is None:
        vector_magnitudes = np.sqrt(np.square(df[[f'{weight}_x', f'{weight}_y', f'{weight}_z']]).sum(axis=1))
        vmin = np.min(vector_magnitudes)
        vmax = np.max(vector_magnitudes)
        normalization_factor = vmax/3
    # Create CGO object for arrows
    arrows = []

    # Iterate through the DataFrame rows
    for _, row in df.iterrows():
        node = row['source']  # Assuming 'source' column has the node name
        if node in node_coords:
            start_coord = node_coords[node]

            # Extract direction vectors (weight_x, weight_y, weight_z)
            direction_vector = [
                row[f'{weight}_x'],
                row[f'{weight}_y'],
                row[f'{weight}_z']
            ]

            # Normalize the direction vector for arrow length
            vector_magnitude = sum([x**2 for x in direction_vector])**0.5
            direction_vector = [x/normalization_factor for x in direction_vector]

            # Calculate the end point of the arrow
            end_coord = [
                start_coord[0] + direction_vector[0],
                start_coord[1] + direction_vector[1],
                start_coord[2] + direction_vector[2]
            ]

            end_coord_cone = [
                    start_coord[0] + (4/3*direction_vector[0]),
                    start_coord[1] + (4/3*direction_vector[1]),
                    start_coord[2] + (4/3*direction_vector[2])]

            # Determine the color (either from a color scale or default gray)
            if use_color_scale:
                color = get_color_from_value(vector_magnitude, vmin=vmin, vmax=vmax)
            else:
                color = (0.8, 0.8, 0.8)  # Default gray color

            # Add the arrow to the CGO object
            arrows += [CYLINDER, *start_coord, *end_coord, 0.2, *color, *color]
            arrows += [CONE, *end_coord, *end_coord_cone, 0.4, 0, *color, *color, 1, 1]

    # Load the CGO object into PyMOL
    cmd.load_cgo(arrows, label)

# PyMOL command to load and visualize arrows directly from the script
def draw_arrows(path, weight='weight', normalization_factor=None, use_color_scale=True, label=None):
    """
    PyMOL command to load node data from a file and visualize arrows with varying lengths and optional color scaling.

    Arguments:
    path -- Path to the network data file (CSV or pickle)
    normalization_factor -- Normalization factor for the arrow length (default: 1.0)
    use_color_scale -- Whether to apply color scale (default: True)
    """
    if label is None:
        label = f'{weight}_arrows'
    cmd.delete("*_arrows")
    view = cmd.get_view()
    df = load_network_data(path)
    visualize_arrows(df, label, weight, normalization_factor, use_color_scale)
    cmd.set_view(view)

# Extend the pymol command to be callable directly within PyMOL
cmd.extend("draw_arrows", draw_arrows)

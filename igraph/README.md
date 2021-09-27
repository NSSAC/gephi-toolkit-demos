# Contents

* [Dependencies](#dependencies)

* [Arguments](#arguments) (This details the command arguments.)
	
  1. [Required Arguments](#required-arguments)
  2. [Optional Arguments](#optional-arguments)
     1. [Value-Based Arguments](#value-arguments) (These are arguments that are more than a command-line switch)
        1. [Layout Algorithm Options](#layout-algorithm-options)
     2. [Boolean Switch Arguments](#boolean-switches) (These are arguments that just boolean switches.)
  3. [Example Invocations](#example-invocations)
  4. [Miscellaneous](#miscellaneous)

# Dependencies 
Please use conda on the igraph/requirements.txt file. When installing the dependencies, use the conda-forge channel.
```
conda create --name <env> --file igraph/requirements.txt -c conda-forge
```
# Arguments

## Required Arguments

* Input path: The path an input file this should be one accepted by the following readers:
	
	    * Read_DL - igraph._igraph.GraphBase.Read_DL
	    * Read_Edgelist - igraph._igraph.GraphBase.Read_Edgelist
	    * Read_GML - igraph._igraph.GraphBase.Read_GML
	    * Read_GraphDB - igraph._igraph.GraphBase.Read_GraphDB
	    * Read_GraphML - igraph._igraph.GraphBase.Read_GraphML
	    * Read_GraphMLz - igraph.Graph.Read_GraphMLz
	    * Read_Lgl - igraph._igraph.GraphBase.Read_Lgl
	    * Read_Ncol - igraph._igraph.GraphBase.Read_Ncol
	    * Read_Pajek - igraph._igraph.GraphBase.Read_Pajek
	    * Read_Pickle - igraph.Graph.Read_Pickle
	    * Read_Picklez - igraph.Graph.Read_Picklez
	    * Read_Picklez 0 - igraph.Graph.Read_Picklez 0
	* File extension should be one of:
		* lgl
		* adjacency
		* dimacs
		* dl
		* edgelist, edges, edge
		* graphviz, dot
		* gml
		* graphml
		* graphmlz
		* leda
		* ncol
		* pajek, net
		* pickle
    
* Output path: The path to the output plot. The file suffix will determine the type automatically. PDF is the recommended ending.
	* The file extension should be one of:
		* pdf
		* png
		* svg
		* ps or eps


## Optional Arguments

### Value Arguments

* algo - This sets the layout algorithm. Selection should come from the next section.
	
    #### Layout Algorithm Options:
    ~~~~
    layout_circle
		
	
    circle, circular
		
	
    Deterministic layout that places the vertices on a circle
    ----------	
    layout_drl
	
	
    drl
		
	
    The Distributed Recursive Layout algorithm for large graphs
    ----------	
    layout_fruchterman_reingold
		
	
    fr
		
	
    Fruchterman-Reingold force-directed algorithm
    ----------	
    layout_fruchterman_reingold_3d
		
	
    fr3d, fr_3d
		
	
    Fruchterman-Reingold force-directed algorithm in three dimensions
    ----------	
    layout_grid_fruchterman_reingold
		
	
    grid_fr
		
	
    Fruchterman-Reingold force-directed algorithm with grid heuristics for large graphs
    ----------	
    layout_kamada_kawai
		
	
    kk
		
	
    Kamada-Kawai force-directed algorithm
    ----------	
    layout_kamada_kawai_3d
		
	
    kk3d, kk_3d
		
	
    Kamada-Kawai force-directed algorithm in three dimensions
    ----------	
    layout_lgl
		
	
    large, lgl, large_graph
		
	
    The Large Graph Layout algorithm for large graphs
    ----------	
    layout_random
		
	
    random
		
	
    Places the vertices completely randomly
    ----------	
    layout_random_3d
		
	
    random_3d
		
	
    Places the vertices completely randomly in 3D
    ----------	
    layout_reingold_tilford
		
	
    rt, tree
		
	
    Reingold-Tilford tree layout, useful for (almost) tree-like graphs
    ----------	
    layout_reingold_tilford_circular
		
	
    rt_circular
	
    tree
		
	
    Reingold-Tilford tree layout with a polar coordinate post-transformation, useful for (almost) tree-like graphs
    ----------	
    layout_sphere
		
	
    sphere, spherical, circular_3d
		
	
    Deterministic layout that places the vertices evenly on the surface of a sphere
    ----------	
    ~~~~

  
* cluster - This takes a list of graph clustering algorithm names. They should be selected from: components the connected components, cohesive_blocks, community_edge_betweenness,
                          community_infomap, community_label_propagation, community_leading_eigenvector, community_leading_eigenvector_naive, community_leiden,
                          community_multilevel (a version of Louvain), community_optimal_modularity (exact solution, < 100 vertices), community_spinglass, community_walktrap. If none is selected
                          community_multilevel is used. The best is selected by modularity score.
  
  	*  **WARNING**: community_fastgreedy is not included because the return type does not have a modularity score.


* output width - Sets the output width in pixels.

* output height - Sets the output height in pixels.

* scale - A string that allows you to choose how to scale the nodes if at all.
  * degree - Scales nodes by their degree in the graph. If nodes are contracted, the degree
  of the nodes in the contracted graph is used.
  * Community only methods
    * comm_degree - Scales nodes in a contracted graph by the sum of the degree of the node's members in the original graph. 
    If not clustering is performed, this is an error.
    * comm_size - This is for contracted graphs only. This scales nodes by the size of their community.
  * Structural Properties:
    * K-Core
    * Clustering Coefficient
    * Betweenness
    * Closeness
    * Page Rank
    * Assortativity
    * Hub Score
    * Authority Score
    * Eccentricity
    * Constraint 
    * Harmonic Centrality

* color - This CLA may have three types of values. If the community contraction is applied with comm_coloring,
the coloring will be applied based on the community selected for the contraction. If there are more nodes than available
  colors, comm_coloring will not color each community uniquely.
1. comm_coloring - This colors the nodes by their community.  
2. An igraph supported color. One of: "red", "blue",
   "black", "brown", "green", "orange", "yellow", 
   "magenta", "lime", "indigo", "cyan".
3. A custom coloring scheme. - Not available yet.

* subgraph_nodes - This is a string that contains a path to a file containing \n separated node IDs.
If this argument is provided, the induced subgraph corresponding to these node IDs will be plotted instead of the full
graph. The induced subgraph is the set of nodes and edges such that a node is in the list of subgraph nodes and an edge
is incident on two nodes in the subgraph nodes. 
Otherwise, it has no effect. The argument should correspond with the node ID found the NCOL ( edgelist-like file 
passed in.)
* ego_node_center - If no argument for this is provided, it has no effect. Otherwise, it will plot only the subgraph
consisting of the ego network with the argument as the root of the ego network. The number hops from the root node is 
controlled by ego_node_distance. The argument should correspond with the node ID found the NCOL ( edgelist-like file 
passed in.)
* ego_node_distance - This an integer CLA that defaults to 1 and controls the ego network defined by ego_node_center. If
ego_node_center is not passed, this argument has no effect.
* edge_width - This is a string argument whose value, if present, will use the graph edge attributes to determine
edge width in the output plot. It is recommended to use an edge list with this argument. This should be a numerical 
attribute.
* edge_color - This is a string argument whose value, if present, will use the graph edge attributes to determine
edge color in the output plot. It is recommended to use an edge list with this argument. This should be a categorical
attribute.

#### Boolean Switches

* node_labels - If this flag is set, the node labels in the input edge file are plotted on the graph.
* node_labels_names - This provides the names of the values to be used in node labels. This is an optional argument, and
if node_labels is set, this will have the node name added to the list. Note that the requested node attribute values
must be in the input edges file. It is recommended to graphml or a similar format for attributes.
* contract - This will contract nodes into their communities as determined by the multilevel communities algorithm (Louvain-based).

* add_subgraph_boundary - This will add the all nodes and edges that are incident on an edge that connects to a node in 
the list of subgraph nodes specified by the subgraph_nodes parameter.

* drop_isolates - This removes isolates. It will be run prior to any clustering.

* directed - This switch will interpret the input file as a directed graph when provided. Otherwise, it is interpreted 
  as undirected.
  
* multi_edges - This switch will allow multi-edges in the output plot. If not provided, multi-edges will be removed from
the plot.
  
* self_loops - This switch will allow self-loops in the output plot. If not provided, self-loops will be removed from
the plot.
  
* node_labels - If the input file is an edge list file, the node IDs in the edge list will be saved and used as the name
attribute in the loaded igraph object. These names will be plotted in the output graph unless the number of nodes is too
  high.

# Example Invocations

**NOTE**: All invocations are made with the `igraph` directory as the root.
**NOTE**: You can validate the output plots against `demo_net_plots_validated`.


## Minimal Invocation

Here, we will see an example of how to plot with a minimal amount of configuration.

`
 python plot_graph.py --input_path demo_net_inputs/rec-amazon.edges --output_path demo_net_plots/rec-amazon.simple.pdf 
`

This takes the demo_net_inputs/rec-amazon.edges network and plots it using the pre-set identified above.


## Boolean Switch Example


There are several boolean switches that are described in the [switches section](#boolean-switches). We will demonstrate
the use of `--contract`. This switch can work in conjunction with the `--cluster` argument which is a list of clustering
algorithms to run. If `--cluster` is not present, we default to using the multi-level Louvain algorithm. Here is 
an example of that:

```
python plot_graph.py --input_path demo_net_inputs/rec-amazon.edges --output_path demo_net_plots/rec-amazon.simple.contracted.pdf --contract
```

If `--cluster` is present, we choose the algorithm that provides the best clustering as ranked by modularity score. Here is 
an example of that:

```
python plot_graph.py --input_path demo_net_inputs/rec-amazon.edges --output_path demo_net_plots/rec-amazon.simple.multi_cluster.pdf --contract --cluster community_multilevel community_leading_eigenvector
```


## Node Labelling Examples

To plot with node labels, first, ensure that you are using one of the edgelist formats or another format such GraphML
that can handle node attributes. GraphML supports rich attributes, so we start with an example of that. We first set the 
`--node_labels` flag without setting `--node_labels_names`. This will default to showing the `name` attribute.

```
python plot_graph.py --input_path demo_net_inputs/test.graphml --output_path demo_net_plots/test.graphml.node_label.pdf --node_labels
```

Now, we will provide a list of attributes with `--node_labels_names`:

```
python plot_graph.py --input_path demo_net_inputs/test.graphml --output_path demo_net_plots/test.graphml.node_label_multi.pdf --node_labels --node_labels_names Country label
```

## Subgraphs
This shows the two ways to plot a subgraph.

### Plotting Ego Networks

An ego network is the induced subgraph of a graph that contains the ego node and all nodes that are n-hops from the ego.
We show an example here with a 1-hop ego network:
```
python plot_graph.py --input_path demo_net_inputs/rec-amazon.edges --output_path demo_net_plots/test.ego.1_hop.png --ego_node_center 3
```
Notice this is just the node with name '3' and its neighbors.

Next, we see a 5-hop ego network of '3':
```
python plot_graph.py --input_path demo_net_inputs/rec-amazon.edges --output_path demo_net_plots/test.ego.5_hop.png --ego_node_center 3 --ego_node_distance 5
```

### Plotting Induced Subgraph

While ego networks are nice, we may desire a more general induced subgraph. We can do this by providing a nodes file like
the one seen in demo_net_inputs/test.nodes that lists a node name per line. Here is an example invocation of this:
```
python plot_graph.py --input_path demo_net_inputs/rec-amazon.edges --output_path demo_net_plots/test.subgraph.png --subgraph_nodes demo_net_inputs/test.nodes 
```

If we would like to include that nodes on the boundary of the induced subgraph, we add the boolean switch:

```
python plot_graph.py --input_path demo_net_inputs/rec-amazon.edges --output_path demo_net_plots/test.subgraph.with_boundary.png --subgraph_nodes demo_net_inputs/test.nodes --add_subgraph_boundary
```

## Scaling

By default, nodes are scaled uniformly according to the number of pixels in the plot. One may wish to alter that and
many options are offered as seen in the section that covers this parameter. All nodes are sizes are fit to a normal 
distribution when they are scale to avoid incredibly large or small nodes.

Here, I show an example of scaling by degree. 

``
python plot_graph.py --input_path demo_net_inputs/rec-amazon.edges --output_path demo_net_plots/test.subgraph.scaled.png --scale "degree" 
``

Here, I show any example of scaling by K-core

``
python plot_graph.py --input_path demo_net_inputs/rec-amazon.edges --output_path demo_net_plots/test.subgraph.scaled_k_core.png --scale "k_core" 
``

# Miscellaneous

**NOTE**: igraph has poor documentation at times.
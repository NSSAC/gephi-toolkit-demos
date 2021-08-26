import warnings
import traceback as tb

import igraph
import numpy as np

# Constants
ARROW_SIZE = 1
ARROW_WIDTH = 1
MAX_NODE_COUNT = 100


def load_induced_subgraph_nodes(subgraph_nodes_path: str):
    """
    This loads nodes for an induced subgraph.
    :param subgraph_nodes_path: The input path to the subgraph nodes.
    :return: The nodes to include in the induced subgraphs.
    """
    with open(subgraph_nodes_path, "r") as nodes_fp:
        nodes = nodes_fp.readlines()
    return list(map(lambda x: x.strip('\n'), nodes))


def get_induced_subgraph(G: igraph.Graph, node_list: list):
    """
    The list of nodes to get for the induced subgraph.
    :param G: The input graph.
    :param node_list: The node list for the induced subgraph.
    :return: The induced subgraph.
    """
    return G.induced_subgraph(vertices=node_list)


def get_ego_net(G: igraph.Graph, ego_node_center: int, ego_node_distance: int):
    """
    Get the ego node network given the input ego node center and ego node distance.
    :param G: The input graph G.
    :param ego_node_center: The center of the ego network.
    :param ego_node_distance: The distance from the center of the ego network.
    :return: A list of nodes in the ego network.
    """
    return G.neighborhood(vertices=ego_node_center, order=ego_node_distance)


def label_nodes(G: igraph.Graph, node_labels: bool, node_labels_names: list):
    """
    This is used perform any node labelling required.
    :param G: The graph to be plotted.
    :param node_labels: A boolean deciding whether or not to plot node labels.
    :param node_labels_names: A list of node attributes to use in node labels.
    :return:
    """
    # Set the plot attribute for node labels to the name attribute.
    if node_labels or node_labels_names:
        if not node_labels_names and G.vs["label"] == []:
            node_labels_names += ["name"]
        # Raise error if attempting to label when there are too many nodes
        if len(G.vs) > MAX_NODE_COUNT:
            raise ValueError("There are too many nodes in the graph to plot the labels.")
        # Plot the labels given by the user.
        if node_labels_names:
            labels = ["{" for i in range(len(G.vs))]
            for i in range(0, len(node_labels_names) - 1):
                label = node_labels_names[i]
                labels = [labels[j] + str(label) + " : " + G.vs[label][j] + ", " for j in range(len(labels))]
            last_label = node_labels_names[-1]
            labels = [labels[j] + str(last_label) + " : " + G.vs[last_label][j] + "}" for j in range(len(labels))]
            G.vs["label"] = labels
    else:
        G.vs["label"] = ["" for i in range(len(G.vs))]


def modify_edges(G: igraph.Graph, edge_width: str, edge_color: str):
    """
    This is used perform any node labelling required.
    :param G: The graph to be plotted.
    :param edge_width: The attribute to be used for edge width.
    :param edge_color: The attribute to be used for edge color.
    :return:
    """
    # Set the plot attribute for node labels to the name attribute.
    if edge_width is not None:
        G.es["width"] = G.es[edge_width]
    if edge_color is not None:
        # We are expecting a categorical attribute here, so we find a list of all values.
        values = set(G.es[edge_color])
        pal = igraph.drawing.colors.RainbowPalette(n=len(values))
        color_map = {value: color_int for color_int, value in enumerate(values)}
        colors_to_get = list(map(lambda x: pal.get(color_map[x]), G.es[edge_color]))
        G.es['color'] = colors_to_get


def cluster(G: igraph.Graph, algo_str):
    """
    This is a helper function to compute clusters using user-selected algorithms.
    :param G: The input graph.
    :param algo_str: A string saying what clustering algorithm to use.
    :return: A clustering object.
    """

    if algo_str == "components":
        return G.components()
    elif algo_str == "cohesive_blocks":
        return G.cohesive_blocks()
    elif algo_str == "community_edge_betweenness":
        return G.community_edge_betweenness()
    elif algo_str == "community_fastgreedy":
        return G.community_fastgreedy()
    elif algo_str == "community_infomap":
        return G.community_infomap()
    elif algo_str == "community_label_propagation":
        return G.community_label_propagation()
    elif algo_str == "community_leading_eigenvector":
        return G.community_leading_eigenvector()
    elif algo_str == "community_leading_eigenvector_naive":
        return G.community_leading_eigenvector_naive()
    elif algo_str == "community_leiden":
        return G.community_leiden()
    elif algo_str == "community_multilevel":
        return G.community_multilevel()
    elif algo_str == "community_optimal_modularity":
        return G.community_optimal_modularity()
    elif algo_str == "community_spinglass":
        return G.community_spinglass()
    elif algo_str == "community_walktrap":
        return G.community_walktrap()
    else:
        print(algo_str)
        raise ValueError("Invalid clustering algorithm name.")


def load_graph(input_path: str, directed: bool, multi_edges: bool, self_loops: bool):
    """
    A helper function used to perform the graph loading part of the plot.
    :param input_path: A string path to the input graph files.
    :param directed: A boolean that decides whether the graph is interpreted as directed.
    :param multi_edges: A boolean that decides whether the graph allows multi-edges.
    :param self_loops: A boolean that decides whether the graph allows self-loops.
    :return: The loaded igraph Graph object.
    """
    G = igraph.Graph()
    try:
        if input_path.split(".")[-1] in ["graphml", "graphmlz", "pickle", "gml", "dimacs"]:
            G = igraph.Graph.Load(input_path)
        else:
            if input_path.split(".")[-1] in ["edges", "edge", "edgelist", "ncol"]:
                G = igraph.Graph.Read_Ncol(input_path, directed=directed, weights='if_present')

            G = igraph.Graph.Load(input_path, directed=directed)

        # If a graph is not a simple, the graph should have multi-edges and self-loops removed if they are not allowed.
        if not multi_edges or not self_loops:
            if not G.is_simple():
                G = G.simplify(multiple=not multi_edges, loops=not self_loops)
                warnings.warn(UserWarning("WARNING: The input graph had either self-loops or multi-edges. "
                                          "You set allow multi-edges to : " + str(multi_edges) + ". You set allow "
                                                                                                 "self-loops to: " + str(
                    self_loops) + ". Those you set to false caused "
                                  "multi-edges and/or self-loops to be removed."))
    except Exception as e:
        tb.print_exc()
        print(e)
        print("Failed to load the graph from: " + input_path)
        exit(1)
    finally:
        return G


def compute_best_clustering(G: igraph.Graph, clusterings: list):
    """
    Computes the best clustering using modularity score for input graph.
    :param G: The input graph.
    :param clusterings: The list of igraph clustering algorithms to try:
    :return: The highest modularity score clustering found.
    """
    G.to_undirected()
    warnings.warn(UserWarning("Clustering will convert the graph to undirected."))
    # Find the best clustering from all of those provided.
    best_cluster = None
    best_score = 0
    for clustering in clusterings:
        if best_cluster is None:
            best_cluster = cluster(G, clustering)
            best_score = float(best_cluster.modularity)
        else:
            curr_cluster = cluster(G, clustering)
            curr_score = float(curr_cluster.modularity)
            if best_score < float(curr_score):
                best_cluster = curr_cluster
                best_score = curr_score
    return best_cluster


def set_arrow_sizes(visual_style):
    """
    This sets the sizes of arrows using the constants defined above.
    :param visual_style: The visual sytle dictionary being used in the plot method.
    :return: None
    """
    # Compute the total number of pixels in the output plot

    visual_style["edge_arrow_size"] = ARROW_SIZE
    visual_style["edge_arrow_width"] = ARROW_WIDTH


def get_vertex_size(G, output_width: int, output_height: int):
    """
    This computes the vertex size based on the output size and graph properties.
    :param G: The graph to plot.
    :param output_width: The output width in pixels
    :param output_height: The output height in pixels
    :return: The vertex size in pixels
    """

    # Compute the total number of pixels in the output plot
    total_pixels = output_width * output_height
    # Scale the vertex and arrow size based on the number of output pixels
    vertex_size = min(total_pixels / ((G.vcount() * 6) + 1), 15)
    return vertex_size


def scale_nodes(scale: str, G: igraph.Graph, old_G: igraph.Graph, best_cluster, vertex_size: float, directed: bool):
    """
    This performs the node scaling if required.
    :param G: The current graph.
    :param scale: The scaling argument. Indicates which type of scaling to perform.
    :param old_G: The graph before contraction.
    :param best_cluster: The best clustering that was found.
    :param vertex_size: The base vertex size in pixels
    :param directed: The directed argument is used to determine node structural properties.
    :return: None
    """
    if scale:

        if scale == "comm_degree":
            if old_G is None:
                raise ValueError("The old_G is None and scaling is being performed by community degree.")

            # Compute inter-community degree
            sizes = np.fromiter(
                (sum([sum(1 for neighbor in old_G.vs[node].neighbors() if neighbor not in best_cluster[comm.index])
                      for node in best_cluster[comm.index]]) for comm in
                 G.vs),
                dtype=float)
        elif scale == "comm_size":
            sizes = best_cluster.sizes()
            # Scale the nodes by the size of their communities

        elif scale == "degree":
            deg = G.degree()
            sizes = deg

        elif scale == "k_core":
            k_cores = G.coreness(shell_index='all')
            sizes = k_cores

        elif scale == "clustering_coefficient":
            clustering_coefficients = G.transitivity_local_undirected()
            sizes = clustering_coefficients

        elif scale == "betweenness":
            betweenness = G.betweenness(directed=directed)
            sizes = betweenness

        elif scale == "closeness":
            closeness = G.closeness()
            sizes = closeness

        elif scale == "page_rank":
            page_rank = G.pagerank(directed=directed)
            sizes = page_rank

        elif scale == "assortativity":
            assortativity = G.assortativity(directed=directed)
            sizes = assortativity

        elif scale == "hub_score":
            hub_score = G.hub_score()
            sizes = hub_score

        elif scale == "authority_score":
            authority_score = G.authority_score()
            sizes = authority_score

        elif scale == "eccentricity":
            eccentricity = G.eccentricity()
            # The higher a node's eccentricity, smaller it should be.
            eccentricity = list(map(lambda x: 1 / x, eccentricity))
            sizes = eccentricity

        elif scale == "constraint":
            constraint = G.constraint()
            # The higher a node's constraint, smaller it should be.
            constraint = list(map(lambda x: 1 / x, constraint))
            sizes = constraint

        elif scale == "harmonic_centrality":
            harmonic_centrality = G.harmonic_centrality()
            sizes = harmonic_centrality

        else:
            raise Exception(scale + " is not a valid scaling style.")
        sizes = (((sizes - np.mean(sizes)) / ((2 * np.std(sizes)) + 1)) * vertex_size) + vertex_size
        G.vs["size"] = sizes
    else:
        G.vs["size"] = vertex_size

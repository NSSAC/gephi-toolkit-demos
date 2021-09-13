"""
This is the driver file for plotting graphs used in the GraphPlot tool.
"""

__author__ = "Henry Carscadden"
__contact__ = "hlc5v@virginia.edu"

import numpy as np
import igraph
import argparse

from plotting_methods import load_graph, get_vertex_size, set_arrow_sizes, compute_best_clustering, scale_nodes, \
    label_nodes, get_ego_net, get_induced_subgraph, load_induced_subgraph_nodes, modify_edges

# Constants
OUTPUT_FORMATS = ["pdf", "png", "svg", "ps", "eps"]
INPUT_FORMATS = ["lgl", "adjacency", "dimacs", "dl", "edgelist", "edges", "edge", "graphviz", "dot", "gml", "graphml",
                 "graphmlz", "leda", "ncol", "pajek", "net", "pickle"]

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", required=True, type=str, help="Path to input file.")
parser.add_argument("--output_path", required=True, type=str, help="Path to output file.")

# All of the below are optional CLAs
parser.add_argument("-algo", "--layout_algorithm", required=False, default="auto", type=str, help="The layout algorithm"
                                                                                                  "that iGraph should "
                                                                                                  "use.")
parser.add_argument('--contract', dest="contract", action='store_true', default=False, required=False,
                    help="If this flag is provided, the script will "
                         "attempt"
                         "to contract nodes into their clusters. "
                         "Recommended for larger graphs ~100k+.")
parser.add_argument('--directed', dest="directed", action='store_true', default=False, required=False,
                    help="If this flag is provided, the script will "
                         "interpret the input Graph as directed.")
parser.add_argument("--color", required=False, type=str, help="This CLA may have three types of values. 1."
                                                              "comm_coloring - This colors the nodes by their "
                                                              "community. "
                                                              '2. An igraph supported color. One of: "red", "blue", '
                                                              '"black", "brown", "green", "orange", "yellow", '
                                                              '"magenta", "lime", "indigo", "cyan"'
                                                              "3. A custom coloring scheme. - Not available yet.")
parser.add_argument("--cluster", required=False,
                    type=str, nargs="*", default=["community_multilevel"], help="This takes a list of graph "
                                                                                "clustering algorithm names. "
                                                                                "They should be selected from: "
                                                                                "components the connected "
                                                                                "components, "
                                                                                "cohesive_blocks, "
                                                                                "community_edge_betweenness, "
                                                                                "community_fastgreedy, "
                                                                                "community_infomap, "
                                                                                "community_label_propagation, "
                                                                                "community_leading_eigenvector, "
                                                                                "community_leading_eigenvector_naive, "
                                                                                "community_leiden, "
                                                                                "community_multilevel "
                                                                                "(a version of Louvain), "
                                                                                "community_optimal_modularity "
                                                                                "(exact solution, < 100 "
                                                                                "vertices), "
                                                                                "community_spinglass, "
                                                                                "community_walktrap. "
                                                                                "If none is selected "
                                                                                " community_multilevel is used.")
parser.add_argument("--output_width", required=False, default=2000,
                    type=int, help="Specify the output width in pixels.")
parser.add_argument("--output_height", required=False, default=1000,
                    type=int, help="Specify the output height in pixels.")
parser.add_argument("--scale", required=False, type=str, help="This string argument that takes three possible values"
                                                              ": degree, comm_degree, and comm_size. These determine"
                                                              " how nodes are scaled if at all.")
# This flag manages isolates.
parser.add_argument("--drop_isolates", action='store_true', dest="drop_isolates", required=False, default=False,
                    help="If this flag is provided, the script will drop"
                         "isolates from the graph plot.")
# These flags are to manage whether or not multi-edges/self-loops are allowed.
parser.add_argument("--multi_edges", action='store_true', dest="multi_edges", required=False, default=False,
                    help="Remove multi-edges if this flag is not set.")
parser.add_argument("--self_loops", action='store_true', dest="self_loops", required=False, default=False,
                    help="Remove self loops if this flag is not set.")
# This flag manages whether or not node labels are shown.
parser.add_argument("--node_labels", action="store_true", dest="node_labels", required=False, default=False,
                    help="If this flag is set, the node labels in the input edge file are plotted on the graph.")
parser.add_argument("--node_labels_names", required=False,
                    type=str, nargs="*", default=[],
                    help="This provides the names of the values to be used in node labels.")
# This flag manages whether or not edge labels are shown.
parser.add_argument("--edge_width", type=str, dest="edge_width", required=False, default=None,
                    help="The edge attribute controlling edge width.")
parser.add_argument("--edge_color", type=str, dest="edge_color", required=False, default=None,
                    help="The edge attribute controlling color.")
# Ego net parameters
parser.add_argument("--ego_node_center", type=str, required=False, default=None,
                    help="An integer with the node ID used for the root "
                         "of an ego network subplot.")
parser.add_argument("--ego_node_distance", type=int, default=1, required=False, help="The maximum distance from the "
                                                                                     "root of the ego network.")
# A file containing nodes in an induced subgraph.
parser.add_argument("--subgraph_nodes", type=str, default=None, required=False, help="This should a string that "
                                                                                     "contains the path to node IDs to "
                                                                                     "include in an induced subgraph.")
parser.add_argument("--add_subgraph_boundary", action='store_true', dest="add_subgraph_boundary", required=False,
                    default=False, help="Add the boundary of the induced subgraph if requested.")


def main():
    args = parser.parse_args()
    # Path Arguments
    input_path = args.input_path
    # Graph Metadata
    directed = args.directed
    # Check if the format is valid.
    if input_path.split(".")[-1].lower() not in INPUT_FORMATS:
        print("The input file extension is " + input_path.split(".")[-1].lower() + " is not a supported input format.")
        raise TypeError("The input file extension should be one of: " + str(INPUT_FORMATS))

    # Check if the format is valid.
    output_path = args.output_path
    if output_path.split(".")[-1].lower() not in OUTPUT_FORMATS:
        print(
            "The input file extension is " + output_path.split(".")[-1].lower() + " is not a supported output format.")
        raise TypeError("The output file extension should be one of: " + str(OUTPUT_FORMATS))
    # The ego network parameters
    ego_node_center = args.ego_node_center
    ego_node_distance = args.ego_node_distance
    # Get the induced subgraph path
    subgraph_nodes = args.subgraph_nodes
    add_subgraph_boundary = args.add_subgraph_boundary
    if add_subgraph_boundary and subgraph_nodes is None:
        raise ValueError("--add_subgraph_boundary requires that --subgraph_nodes be set first.")

    # Modifications to the plot
    contract = args.contract
    color = args.color
    scale = args.scale
    if directed and (scale == "clustering_coefficient"):
        raise ValueError("Scaling by clustering coefficient is not allowed with directed graphs.")
    # If we are not contracting into communities and these options are set, this is a problem.
    if (scale == "comm_degree" or scale == "comm_size") and not contract:
        raise ValueError("If scaling by community traits is requested (i.e. comm_degree or comm_size), then,"
                         "contract must also be true.")
    # Boolean switches to decide what is allowed in the graph.
    drop_isolates = args.drop_isolates
    self_loops = args.self_loops
    multi_edges = args.multi_edges

    print(f"Contract: {contract} Drop Isolates: {drop_isolates}")
    # Get the output size
    output_width, output_height = int(args.output_width), int(args.output_height)
    # Layout algorithm
    layout_algorithm = args.layout_algorithm
    # The clustering algorithm to try.
    clusterings = args.cluster
    # Should the input file be interpreted with node labels for the output plot
    edge_width = args.edge_width
    edge_color = args.edge_color
    # Should the input file be interpreted with edge labels for the output plot
    node_labels = args.node_labels
    node_labels_names = args.node_labels_names
    # Ensure that plotting node labels is actually possible.
    if node_labels or node_labels_names:
        if contract:
            raise ValueError(
                "The --node_labels and --contract flags cannot both be set. This would result in invalid node "
                "labels.")
    # Ensure that plotting edge labels is actually possible.
    if edge_width is not None:
        if contract:
            raise ValueError(
                "The --edge_labels and --contract flags cannot both be set. This would result in invalid edge "
                "labels.")

    # Initialize the visual style object that will be used to set the plot parameters.
    visual_style = {}

    colors = ["red", "blue", "black", "brown", "green", "orange", "yellow", "magenta", "lime", "indigo", "cyan"]

    print("Beginning Graph Loading.")
    # Attempt to load the graph

    G = load_graph(input_path=input_path,
                   directed=directed, multi_edges=multi_edges, self_loops=self_loops)
    print("Graph finished loading.")
    print("======================")

    if subgraph_nodes is not None and ego_node_center is not None:
        raise ValueError("Both subgraph_nodes and ego_node_center parameters cannot be used together.")

    induced_graph_nodes = None
    # Get the ego network
    if ego_node_center is not None:
        ego_node_center = G.vs.find(name=ego_node_center)
        induced_graph_nodes = get_ego_net(G=G, ego_node_center=ego_node_center, ego_node_distance=ego_node_distance)

    # Get the induced graph nodes
    if subgraph_nodes is not None:
        induced_graph_nodes = load_induced_subgraph_nodes(subgraph_nodes_path=subgraph_nodes)
        induced_graph_nodes = list(G.vs.select(name_in=induced_graph_nodes))
        boundary_nodes = []
        # Add in the boundary nodes if requested.
        if add_subgraph_boundary:
            boundary_node_lists = G.neighborhood(vertices=induced_graph_nodes, mindist=1)
            for node_list in boundary_node_lists:
                boundary_nodes += node_list
            induced_graph_nodes += boundary_nodes
            induced_graph_nodes = set(induced_graph_nodes)

    if subgraph_nodes is not None or ego_node_center is not None:
        G = get_induced_subgraph(G=G, node_list=induced_graph_nodes)

    # This finds the vertex size.
    vertex_size = get_vertex_size(G=G, output_width=output_width, output_height=output_height)

    # Set the arrow sizes
    set_arrow_sizes(visual_style=visual_style)

    # Drop isolates if requested.
    if drop_isolates:
        G.delete_vertices(G.vs.select(_degree=0))

    # Adding node labels
    label_nodes(G=G, node_labels=node_labels, node_labels_names=node_labels_names)
    # Adding edge labels
    modify_edges(G=G, edge_width=edge_width, edge_color=edge_color)

    # Find the best clustering based on modularity score.
    best_cluster = None
    if color == "comm_coloring" or contract:
        print("Starting clustering computation.")
        best_cluster = compute_best_clustering(G=G, clusterings=clusterings)
        print("Finishing clustering algorithm run.")
        print("====================")

    old_G = None
    if contract:

        if scale != "degree":
            old_G = G.copy()
        # Set G to the cluster that was identified earlier.
        G = best_cluster.cluster_graph()

        print("Finished contracting graph.")
        print("====================")

        # If the coloring is by community, pick a random color for each community which is now a node.
        if color == "comm_coloring":
            G.vs['color'] = np.random.choice(colors, size=(G.vcount(),), replace=True)

    if color == "comm_coloring" and not contract:
        pal = igraph.drawing.colors.ClusterColoringPalette(len(best_cluster))
        G.vs['color'] = pal.get_many(best_cluster.membership)
    elif color in colors:
        G.vs["color"] = color

    print("Running Layout Algorithm: " + str(layout_algorithm))
    print("====================")
    layout = G.layout(layout_algorithm)

    print("Finished running layout algorithm.")
    print("====================")

    print("Setting the node sizes.")
    print("====================")

    scale_nodes(scale=scale, G=G, old_G=old_G, best_cluster=best_cluster, vertex_size=vertex_size, directed=directed)

    print("Finished setting the node sizes in the plot.")
    print("====================")

    # Plot the graph with the requested settings and layout.
    igraph.plot(G, layout=layout,
                bbox=(output_width, output_height),
                target=output_path, **visual_style)
    print("Finished plotting the graph.")
    print("Done.")
    print("====================")


if __name__ == '__main__':
    main()

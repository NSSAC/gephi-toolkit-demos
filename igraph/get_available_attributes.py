import argparse
import json

from plotting_methods import load_graph

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", required=True, type=str, help="Path to input file.")
# These flags are to manage whether or not multi-edges/self-loops are allowed.
parser.add_argument("--multi_edges", action='store_true', dest="multi_edges", required=False, default=False,
                    help="Remove multi-edges if this flag is not set.")
parser.add_argument("--self_loops", action='store_true', dest="self_loops", required=False, default=False,
                    help="Remove self loops if this flag is not set.")
parser.add_argument('--directed', dest="directed", action='store_true', default=False, required=False,
                    help="If this flag is provided, the script will "
                         "interpret the input Graph as directed.")

# Constants
INPUT_FORMATS = ["lgl", "adjacency", "dimacs", "dl", "edgelist", "edges", "edge", "graphviz", "dot", "gml", "graphml",
                 "graphmlz", "leda", "ncol", "pajek", "net", "pickle"]


# Get the node attributes and edge attributes
def main():
    args = parser.parse_args()
    # Path Arguments
    input_path = args.input_path
    # Check if the format is valid.
    if input_path.split(".")[-1].lower() not in INPUT_FORMATS:
        print("The input file extension is " + input_path.split(".")[-1].lower() + " is not a supported input format.")
        raise TypeError("The input file extension should be one of: " + str(INPUT_FORMATS))
    # Graph Metadata
    directed = args.directed
    self_loops = args.self_loops
    multi_edges = args.multi_edges

    G = load_graph(input_path=input_path,
                   directed=directed, multi_edges=multi_edges, self_loops=self_loops)

    # Add all missing node labels and edge labels to the set.
    node_labels = set()
    for node in G.vs:
        for node_label in node.attributes():
            node_labels.add(node_label)
    edge_labels = set()
    for edge in G.es:
        for edge_label in edge.attributes():
            edge_labels.add(edge_label)
    # Output the edge attributes discovered
    with open(input_path + ".edge_attributes.json", "w") as fp:
        json.dump(list(edge_labels), fp)
    # Output the node attributes discovered
    with open(input_path + ".node_attributes.json", "w") as fp:
        json.dump(list(node_labels), fp)


if __name__ == '__main__':
    main()
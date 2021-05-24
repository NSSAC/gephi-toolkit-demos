import numpy as np

import igraph
import warnings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", required=True, type=str, help="Path to input file.")
parser.add_argument("--output_path", required=True, type=str, help="Path to output file.")
parser.add_argument("-algo", "--layout_algorithm", required=False, default="auto", type=str, help="The layout algorithm"
                                                                                                  "that iGraph should "
                                                                                                  "use.")
parser.add_argument('--contract', required=False, type=bool, help="If this flag is provided, the script will "
                                                                  "attempt"
                                                                  "to contract nodes into their clusters. "
                                                                  "Recommended for larger graphs ~100k+.")
parser.add_argument("--color", required=False, type=bool, help="If this flag is provided, the script will try different"
                                                               "clustering, then, color the nodes according to "
                                                               "cluster.")
parser.add_argument("--output_width", required=False, default=2000,
                    type=int, help="Specify the output width in pixels.")
parser.add_argument("--output_height", required=False, default=1000,
                    type=int, help="Specify the output height in pixels.")
parser.add_argument("--scale", required=False, type=bool, help="If this flag is provided, the script will scale"
                                                               "the vertices by proportionally to their degree.")



def main():
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    # Modifications to the plot
    contract = args.contract
    color = args.color
    scale = args.scale
    # Set the output size
    output_width, output_height = int(args.output_width), int(args.output_height)


    layout_algorithm = args.layout_algorithm

    G = igraph.Graph.Load(input_path)
    clustering_methods = [G.community_multilevel, G.community_leiden]

    if contract:
        warnings.warn(UserWarning("Contract will convert the graph to undirected."))
        # TODO: Provide a more sophisticated contraction logic.
        G.to_undirected()
        best_score, best_cluster = 0, None
        for clustering in clustering_methods:
            cluster = clustering()
            if best_score > cluster.modularity:
                best_cluster = cluster
                best_score = clustering
        G = best_cluster.cluster_graph()

    if color:
        warnings.warn(UserWarning("Contract will convert the graph to undirected."))
        G.to_undirected()
        best_score, best_cluster = 0, None
        for clustering in clustering_methods:
            cluster = clustering()
            if best_score > cluster.modularity:
                best_cluster = cluster
                best_score = clustering
        pal = igraph.drawing.colors.ClusterColoringPalette(len(best_cluster))
        G.vs['color'] = pal.get_many(best_cluster.membership)

    deg = G.degree()
    layout = G.layout(layout_algorithm)
    visual_style = {}
    total_pixels = output_width * output_height
    vertex_size = min(total_pixels / ((G.vcount() * 6) + 1), 15)
    arrow_size = min(total_pixels / ((G.ecount() * 45) + 1), 15)
    visual_style["edge_arrow_size"] = arrow_size

    if scale:
        G.vs["size"] = (((deg - np.mean(deg)) / ((2 * np.std(deg)) + 1)) * vertex_size) + vertex_size
    else:
        visual_style["vertex_size"] = vertex_size

    igraph.plot(G, layout=layout,
                bbox=(output_width, output_height),
                target=output_path, **visual_style)


if __name__ == '__main__':
    main()

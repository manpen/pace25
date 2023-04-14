#!/usr/bin/env python3
import sys
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

COLOR_NORMAL_NODE = (0.5, 0.5, 1)
COLOR_REMOVED_NODE = (0.9, 0.9, 0.9)
COLOR_REMOVING = (1, 0.5, 0.5)
COLOR_SURVIVING = (1, 0, 0)


def read_graph(path):
    G = nx.Graph()
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('p') or line.startswith('c'):
                assert line.startswith('p tww ')
                n, m = [int(x) for x in line.split() if x !=
                        "" and x not in ["p", "tww"]]
                G.add_nodes_from(range(1, n+1))
            else:
                G.add_edge(int(line.split()[0]), int(line.split()[1]))

    return G


def read_cs(filename):
    with open(filename, 'r') as f:
        return [(int(x.split()[0]), int(x.split()[1])) for x in f.readlines() if not x.startswith('c')]


def get_edge_color(graph, e):
    return graph.get_edge_data(e[0], e[1])["col"]


def get_edge_colors(graph, edges):
    return [get_edge_color(graph, e) for e in edges]


def merge(rem, sur):
    graph.nodes()[rem]["col"] = COLOR_REMOVED_NODE
    graph.nodes()[sur]["col"] = COLOR_NORMAL_NODE

    nr = set(graph.neighbors(rem))
    ns = set(graph.neighbors(sur))

    union = nr.union(ns)
    common = set((w for w in nr.intersection(ns) if
                  ["k", "k"] == get_edge_colors(graph, [(rem, w), (sur, w)])))

    graph.remove_edges_from([(rem, w) for w in nr])
    graph.remove_edges_from([(sur, w) for w in ns])

    for w in union:
        if w in [sur, rem]:
            continue
        graph.add_edge(sur, w, col=('k' if w in common else 'r'))


def red_degree_of(graph, u):
    n = [v for v in graph.neighbors(u) if get_edge_color(graph, (u, v)) == 'r']
    return len(n)


def visualize_merge_sequence(graph, seq, pos, output_path):
    nx.set_node_attributes(graph, COLOR_NORMAL_NODE, "col")
    nx.set_edge_attributes(graph, 'k', "col")

    tww = 0
    with PdfPages(output_path) as pdf:
        for (step, (sur, rem)) in enumerate(seq):
            plt.figure()
            active = set([sur, rem])
            dist = nx.shortest_path_length(graph, source=sur, target=rem)

            nx.draw_networkx_edges(graph, pos, alpha=0.5,
                                   edge_color=get_edge_colors(graph, graph.edges()))

            edge_list = [(u, v) for (u, v) in graph.edges()
                         if u in active or v in active]

            graph.nodes()[rem]["col"] = COLOR_REMOVING
            graph.nodes()[sur]["col"] = COLOR_SURVIVING

            nx.draw(graph, pos, with_labels=True,
                    edgelist=edge_list,
                    edge_color=get_edge_colors(graph, edge_list),
                    node_color=[graph.nodes[u]["col"] for u in graph.nodes()])

            for u in graph.nodes():
                tww = max(tww, red_degree_of(graph, u))

            plt.title("Step: %02d. Merge %d into %d (distance %d); TWW before: %d" %
                      (step + 1, rem, sur, dist, tww))

            pdf.savefig()
            plt.clf()

            merge(rem, sur)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        graph_path = sys.argv[1]
        cs_path = sys.argv[2]
        output_path = sys.argv[3]

    elif 2 <= len(sys.argv) <= 3:
        cs_path = sys.argv[1]
        assert (".gr." in cs_path)
        graph_path = cs_path
        while not graph_path.endswith(".gr"):
            graph_path = graph_path[:-1]

        if len(sys.argv) == 3:
            output_path = sys.argv[2]
        else:
            output_path = cs_path[len(graph_path)+1:] + ".pdf"

    else:
        print("""Usage: %s [graph] <contraction sequence> <pdf-output>)
If graph is not specified, it is assumed that the contraction
sequence file is named like the graph file, but with an additional suffix""" % sys.argv[0])
        sys.exit(1)

    graph = read_graph(graph_path)
    seq = read_cs(cs_path)
    print("Graph %s loaded with %d nodes and %d edges" %
          (graph_path, graph.number_of_nodes(), graph.number_of_edges()))

    if nx.is_bipartite(graph):
        pos = nx.bipartite_layout(graph, nx.bipartite.sets(graph)[0])
    else:
        pos = nx.spring_layout(graph)

    visualize_merge_sequence(graph, seq, pos, output_path)

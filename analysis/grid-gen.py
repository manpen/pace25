import networkx as nx

n = 6
m = 6
G = nx.grid_2d_graph(n, m)


def to_num(a, b):
    assert (a < n)
    assert (b < m)
    return a * n + b


print("p tww", G.number_of_nodes(), G.number_of_edges())
for (u, v) in G.edges():

    print(to_num(*u)+1, to_num(*v)+1)

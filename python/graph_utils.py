#!/usr/bin/env python3
from torch_geometric.utils import remove_isolated_nodes
import torch
from torch_geometric.data import Data
import spectrum
from itertools import compress
from heapq import heappush, heappop, _heappop_max

def dict_graph_to_f_t_c_graph(graph_dict):
    _from = []
    _to = []
    _char = []

    for key in graph_dict.keys():
        for edge in graph_dict[key]:
            _from.append(key)
            _to.append(edge[0])
            _char.append(edge[1])
    return _from, _to, _char


def torch_graph_to_f_t_c_graph(graph, edge_aa):
    edge_index = graph.edge_index
    edge_index = edge_index.tolist()

    _from = edge_index[0]
    _to = edge_index[1]
    assert len(_from) == len(edge_aa) and len(_from) == len(_to)
    return _from, _to, edge_aa


def torch_graph_to_dict_graph(graph, weights=None, reverse_edges=False, return_edges=False):
    """
    Parse from torch geometric graph to dict graph {'source' -> [(to_node1, weight), ...}

    """
    if not reverse_edges:
        source = graph.edge_index[0].tolist()
        to = graph.edge_index[1].tolist()
    else:
        source = graph.edge_index[1].tolist()
        to = graph.edge_index[0].tolist()

    graph_dict = dict.fromkeys(list(set(source + to)), [])


    chars = spectrum.Spectrum.decode_sequence(graph.edge_attr.cpu())

    for i, s in enumerate(source):
        t = to[i]
        if weights == None:
            graph_dict[s].append(t, chars[i])
        else:
            graph_dict[s].append((t, chars[i], weights[i], i))

    if return_edges:
        return graph_dict, list(zip(source, to)), chars

    return graph_dict



def bfs(graph, start_node):
    visited = [start_node]
    queue = [start_node]

    while queue:
        s = queue.pop(0)
        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
    return visited

def filter_nodes(graph, node_pairs, edges_aa, verbose=False):
    """
    Remove nodes that are not part of possible paths between node pairs
    Any node that is part of at least one path between a node pair is kept.#!/usr/bin/env

    """
    graph_reverse = torch_graph_to_dict_graph(graph, reverse_edges=True)
    graph_d = torch_graph_to_dict_graph(graph)
    all_nodes = set(graph_d.keys())
    keep_nodes = set([])

    for (source, sink) in node_pairs:
        from_source = set(bfs(graph_d, source))
        from_sink = set(bfs(graph_reverse, sink))

        keep_nodes.update(from_source.intersection(from_sink))

    to_remove = all_nodes - keep_nodes

    to_remove = torch.tensor(list(to_remove))
    if verbose:
        print("Removing {n_remove} % of nodes".format(n_remove=len(to_remove) / len(all_nodes) * 100))
    edge_index = graph.edge_index.clone()
    edge_index = torch.transpose(edge_index, 0, 1)

    from_nodes_remove = edge_index[:,0].apply_(lambda x : x in to_remove)
    to_nodes_remove = edge_index[:,1].apply_(lambda x : x in to_remove)
    remove_mask = ~torch.logical_or(from_nodes_remove, to_nodes_remove)

    edges_aa = list(compress(edges_aa, remove_mask.tolist()))

    edge_index = graph.edge_index.transpose(0, 1)[remove_mask, :]
    edge_attr = graph.edge_attr[remove_mask]
    edge_index = torch.transpose(edge_index, 0, 1)
    edge_index, edge_attr, mask = remove_isolated_nodes(edge_index, edge_attr=edge_attr,
                                                        num_nodes=len(all_nodes))

    x = graph.x[mask]
    new_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return new_graph, mask.numpy(), edges_aa



def filter_spectrum(graph, peaks, intensities, edges_aa, node_pairs):
    """
    Filter spectrum peaks not belonging to main paths
    """

    graph, mask, edges_aa = filter_nodes(graph, node_pairs, edges_aa, verbose=True)
    peaks = list(compress(peaks, mask))
    intensities = list(compress(intensities, mask))

    return graph, peaks, intensities, edges_aa


def remove_redundancy(_from, _to, _char):

    _new_from, _new_to, _new_char = list(zip(*list(set(list(zip(_from, _to, _char))))))
    return _new_from, _new_to, _new_char


def paths_number_source_sink(dict_graph, source_sink_tuple, n_paths={}, visited=[], flag=None):
    source, sink = source_sink_tuple
    visited = visited + [source]
    if source == sink:
        return 1
    else:
        if source not in n_paths:
            paths = [paths_number_source_sink(dict_graph, (next_node, sink), visited=visited) \
                                   for next_node in dict_graph[source] if next_node not in visited]
            paths.append(0)
            n_paths[source] = sum(paths)
    return n_paths[source]


class Path(object):
    def __init__(self, cost, path):
        self.cost = cost
        self.path = list(path)

    def __lt__(self, other):
        return self.cost < other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def get(self):
        return (self.cost, list(self.path))


def k_best_bellmanford(dict_graph, edges, source_sink_tuple, k=1, verbose=False):
    source, targets = source_sink_tuple
    targets = [] + [targets]
    # each path is a tuple (cost, (edge_0, edge_1, ...))
    nodes = max(dict_graph.keys())

    max_dist = [0 for i in range(nodes + 1 )]
    count_paths = [0 for i in range(nodes + 1 )]
    max_dist[source] = 0
    paths = [[] for i in range(nodes + 1)]
    heappush(paths[source], Path(0, []))

    for iteration in range(nodes * k):
        if iteration % 10 == 0:
            print(iteration / (nodes * k))
        for node in range(nodes + 1):
            for edge in dict_graph[node]:
                to, char, cost_to, edge_i = edge
                for path in paths[node]:
                    cost, path_from = path.get()
                    if count_paths[to] >= k:
                        if max_dist[to] < cost + cost_to :
                            continue
                    max_dist[to] = max(max_dist[to], cost + cost_to)
                    heappush(paths[to], Path(cost + cost_to, path_from + [to]))
                    count_paths[to] += 1
                    if count_paths[to] > k:
                        _heappop_max(paths[to])
                        max_dist[to] = max(paths[to]).get()[0]
                        count_paths[to] -= 1

    return paths[targets[0]]



def k_best_dijkstra(dict_graph, edges, source_sink_tuple, k=5, verbose=False):
    """
    Compute k shortests paths with Dijkstra approach
    """
    source, targets = source_sink_tuple
    targets = [] + [targets]
    # each path is a tuple (cost, (edge_0, edge_1, ...))
    paths = {}
    nodes = max(dict_graph.keys())

    n_paths = {}
    for node in range(nodes + 1):
        n_paths[node] = 0

    paths = [[] for target in range(len(targets))]

    B = []
    heappush(B, Path(0, []))

    while len(B) > 0 and any([len(paths_t) < k for paths_t in paths]):
        path_obj = heappop(B)
        cost, path = path_obj.get()
        if len(path) > 0: # if it not the source
            if len(path) > 2:
                print(path)
            node = edges[path[-1]][1]
        else:
            node = source
        n_paths[node] += 1

        if node in targets:
            paths[targets.index(node)].append((cost, path))
        # print(n_paths)
        # print(node)
        if n_paths[node] <= k:
            for edge in dict_graph[node]:
                to, char, cost_to, edge_i = edge
                new_path = path + [edge_i]
                heappush(B, Path(cost_to + cost, new_path))

    return paths

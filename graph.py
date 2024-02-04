"""
This module defines classes for nodes and edges, and provides composited functions for graph visualization and analysis.
"""

import csv
import glob
import pickle
import powerlaw
import numpy as np
import seaborn as sns
import networkx as nx
from typing import Tuple, List, Set
import matplotlib.pyplot as plt
import community  # python-louvain package


###### Graph Representation ######

class Node:
    '''
    Represents a node in a graph.
    '''
    def __init__(self, node_type:str, name:str):
        self.node_type = node_type
        self.name = name


class Politician(Node):
    """
    Represents a politician node. 
    """
    def __init__(self, name:str, party:str, state:str):
        super().__init__(name=name, node_type='Politician')
        self.party = party
        self.state = state


class Committee(Node):
    """
    Represents a committee node.
    """
    def __init__(self, name:str, state:str):
        super().__init__(name=name, node_type='Committee')
        self.state = state     


class Policy(Node):
    """
    Represents a policy node.
    """
    def __init__(self, name:str, date:str, synopsis:str, outcome:str):
        super().__init__(name=name, node_type='Policy')
        self.date = date
        self.synopsis = synopsis
        self.outcome = outcome
        

class Donor(Node):
    """
    Represents a donor node.
    """
    def __init__(self, name:str):
        super().__init__(name=name, node_type='Donor')


class InterestGroup(Node):
    """
    Represents a interest group node.
    """
    def __init__(self, name:str):
        super().__init__(name=name, node_type='InterestGroup')


class Edge:
    """
    Represents an edge connecting two nodes in a graph.
    """
    def __init__(self, source_node:str, target_node:str, weight:float, edge_type:str):
        self.source_node = source_node
        self.target_node = target_node
        self.weight = weight
        self.edge_type = edge_type


class Vote(Edge): 
    """
    Represents a directed relationship, i.e. a politician voting yes for a bill. It can also be a weighted relationship, with a default weight of 1.
    """
    def __init__(self, source_node:str, target_node:str, weight:float=1):
        super().__init__(source_node, target_node, weight, edge_type='Vote') 


class Sponsor(Edge):
    """
    Represents a directed relationship, i.e. a politician or committee sponsoring a bill. It can also be a weighted relationship, with a default weight of 1.
    """
    def __init__(self, source_node:str, target_node:str, weight:float=1):
        super().__init__(source_node, target_node, weight, edge_type='Sponsor') 


class Donate(Edge):
    """
    Represents a directed and weighted relationship, i.e., a donor contributes to a politician. The weight is the donation amount.
    """
    def __init__(self, source_node:str, target_node:str, weight:float):
        super().__init__(source_node, target_node, weight, edge_type='Donate')


class Rate(Edge):
    """
    Represents a directed and weighted relationship, i.e., an interest group rates a politician. The weight is the rating number.
    """
    def __init__(self, source_node:str, target_node:str, weight:float):
        super().__init__(source_node, target_node, weight, edge_type='Rate')
   
        
class Consensus(Edge):
    """
    Represents the relationship between the positions of politicians on a particular bill. This is actually a two-way relationship, so the connection between 
    them needs to be created twice. It can also be a weighted relationship, with a default weight of 1.
    """
    def __init__(self, source_node:str, target_node:str, weight:float=1):
        super().__init__(source_node, target_node, weight, edge_type='Consensus')


###### Graph Construction ######

def read_pkl(directory) -> Tuple[List, List]:
    all_nodes = []
    all_edges = []
    # get a list of all the .pkl files in the directory
    pkl_files = glob.glob(f'{directory}/*.pkl')
    # iterate through the files and append the data to the list
    for file_name in pkl_files:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            if file_name.endswith("nodes.pkl"):
                all_nodes.extend(data)
            elif file_name.endswith("edges.pkl"):
                all_edges.extend(data)
    return all_nodes, all_edges


def add_nodes(g:nx.Graph, all_nodes:List[Node]) -> None:
    """
    Add all nodes to a graph.
    """
    for node in all_nodes:
        if isinstance(node, Politician):
            g.add_node(node.name, node_type=node.node_type, party=node.party, state=node.state)
        if isinstance(node, Committee):
            g.add_node(node.name, node_type=node.node_type, state=node.state)
        if isinstance(node, Policy):
            g.add_node(node.name, node_type=node.node_type, date=node.date, synopsis=node.synopsis, outcome=node.outcome)
        if isinstance(node, Donor):
            g.add_node(node.name, node_type=node.node_type)
        if isinstance(node, InterestGroup):
            g.add_node(node.name, node_type=node.node_type)


def add_edges(g:nx.Graph, all_edges:List[Edge]) -> None:
    """
    Add all edges to a graph.
    """
    # batch normalization
    normalize_edge_weight(all_edges, edge_type='Donate')
    normalize_edge_weight(all_edges, edge_type='Rate')
    lst = [(edge.source_node, edge.target_node, {'weight':edge.weight, 'edge_type':edge.edge_type}) for edge in all_edges]
    g.add_edges_from(lst)


def normalize_edge_weight(edge_list:List[Edge], edge_type:str) -> None:
    # get the maximum edge weight of edges
    lst = [edge.weight for edge in edge_list if edge.edge_type == edge_type]
    if lst: # if lst is not empty
        max_weight = max(lst)
        # loop through the edges and normalize the weights of edges
        for edge in edge_list:
            if edge.edge_type == edge_type:
                edge.weight /= max_weight


###### Graph Visulization ######

def draw_whole_graph(g:nx.Graph, nodes:List[Node], edges:List[Edge], color_map:dict) -> None:
    # add all nodes to graph
    add_nodes(g, nodes)
    # add all edges to graph
    add_edges(g, edges)
    # draw the whole graph
    draw_graph(g, 'The Whole Graph', color_map)


def draw_donation_graph(g:nx.Graph, color_map:dict) -> None:
    # create a donation subgraph
    g_donation= nx.DiGraph()
    for n, attr in g.nodes(data=True):
        if attr['node_type'] != 'InterestGroup' and attr['node_type'] !='Policy' and attr['node_type'] !='Committee':
            g_donation.add_node(n, **attr)
    for u, v, attr in g.edges(data=True):
        if u in g_donation.nodes() and v in g_donation.nodes():
            g_donation.add_edge(u, v, **attr)
    draw_graph(g_donation, 'The Donation Subgraph', color_map, with_labels=True)


def draw_rating_graph(g:nx.Graph, color_map:dict) -> None:
    # create a rating subgraph
    g_rating = nx.DiGraph()
    for n, attr in g.nodes(data=True):
        if attr['node_type'] != 'Donor' and attr['node_type'] !='Policy' and attr['node_type'] !='Committee':
            g_rating.add_node(n, **attr)
    for u, v, attr in g.edges(data=True):
        if u in g_rating.nodes() and v in g_rating.nodes():
            g_rating.add_edge(u, v, **attr)
    # Draw the donation subgraph
    draw_graph(g_rating, 'The Rating Subgraph', color_map, with_labels=True)


def draw_sponsorship_graph(g:nx.Graph, color_map:dict) -> None:
    # create a sponsorship subgraph
    g_sponsor= nx.DiGraph()
    for n, attr in g.nodes(data=True):
        if attr['node_type'] != 'InterestGroup' and attr['node_type'] !='Donor':
            g_sponsor.add_node(n, **attr)
    for u, v, attr in g.edges(data=True):
        if u in g_sponsor.nodes() and v in g_sponsor.nodes():
            g_sponsor.add_edge(u, v, **attr)
    draw_graph(g_sponsor, 'The Sponsorship Subgraph', color_map, pos=nx.kamada_kawai_layout(g_sponsor, weight='weight'), with_labels=True)


def draw_graph(g:nx.Graph, title:str, color_map:dict, pos:dict=None, with_labels:bool=False) -> None:
    """
    Customized graph drawing function.
    """
    labels = nx.get_node_attributes(g, 'node_type')
    # map attribute values to colors using a dictionary or a function
    node_colors = [color_map.get(labels[n], 'gray') for n in g.nodes()]
    # draw the graph with node colors
    plt.figure(figsize=(10, 6))
    if not pos:
        pos = nx.spring_layout(g, weight='weight', seed=123)
    nx.draw(g, pos, node_color=node_colors, node_size=80)
    # create a custom legend
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_map.values()]
    plt.legend(markers, color_map.keys(), numpoints=1, title='Node Type', loc='lower right')
    if with_labels:
        # get the degrees of all nodes in the graph
        degrees = dict(g.degree())
        # sort the degrees dictionary by value in descending order
        sorted_degrees = dict(sorted(degrees.items(), key=lambda item: item[1], reverse=True))
        # get the top 5 nodes by degree
        top_nodes = list(sorted_degrees.keys())[:5]
        # draw the graph with the top 5 nodes labeled
        nx.draw_networkx_labels(g, pos, font_size=8, labels={node: node if node in top_nodes else '' for node in g.nodes()})
    # show the plot
    plt.title(title)
    plt.show()


###### Centrality ######

def centrality(g:nx.Graph, politician_name:str, state:str, top_n:int=5) -> Tuple[List, List, List, List]:
    """
    One function to calculate Degree Centrality, Betweenness Centrality, Closeness Centrality and Pagerank Centrality.
    """
    dc = []
    bc = []
    cc = []
    rc = []
    print(f'Centrality Metrics of the {politician_name} ({state}) Network')
    # Degree Centrality
    degree_centrality = nx.degree_centrality(g)
    sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    print('\nDegree Centrality')
    print('-----------------')
    for key, value in sorted_degree[:top_n]:
        print(f'{key}: {value}')
        dc.append(key)
    # Betweenness Centrality
    betweenness_centrality = nx.betweenness_centrality(g)
    sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
    print('\nBetweenness Centrality')
    print('----------------------')
    for key, value in sorted_betweenness[:top_n]:
        print(f'{key}: {value}')
        bc.append(key)
    # Closeness Centrality
    closeness_centrality = nx.closeness_centrality(g)
    sorted_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
    print('\nCloseness Centrality')
    print('--------------------')
    for key, value in sorted_closeness[:top_n]:
        print(f'{key}: {value}')
        cc.append(key)
    # Pagerank Centrality
    pagerank_centrality = nx.pagerank(g)
    sorted_pagerank = sorted(pagerank_centrality.items(), key=lambda x: x[1], reverse=True)
    print('\nPagerank Centrality')
    print('--------------------')
    for key, value in sorted_pagerank[:top_n]:
        print(f'{key}: {value}')
        rc.append(key)

    return dc, bc, cc, rc


###### Power Law ######

def plot_power_law(g:nx.Graph) -> None:
    # extract the degree distribution of the graph
    degree_sequence = sorted([d for _, d in g.degree()], reverse=True)
    # fit the degree distribution to a power law model
    fit = powerlaw.Fit(degree_sequence, discrete=True, method='TruncatedPowerLaw')
    # plot the degree distribution and the power law fit
    fig = fit.plot_pdf(color='b', linewidth=2)
    fit.power_law.plot_pdf(color='b', linestyle='--', ax=fig)
    # add labels and a legend to the plot
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.legend(['Graph data', 'Power law fit'], loc='upper right')
    # show the plot
    plt.show()


###### Sensitivity Analysis ######

def calculate_y(*args) -> list:
    # convert each argument to a set, the first arg should be the base
    sets = [set(arg) for arg in args]
    y = []
    for s in sets:
        score = jaccard_sim([sets[0], s])[1, 0]
        y.append(score)
    return y


def plot_sensitivity(x:list, y_lst:List[list]) -> None:
    _, ax = plt.subplots()
    labels = ['Degree', 'Betweenness', 'Closeness', 'Pagerank'] 
    for y, label in zip(y_lst, labels):
        ax.plot(x, y, label=label)
    ax.set_xticks(x)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Jaccard Similarity Index')
    ax.legend()
    plt.show()


###### Centality Metrics Comparison ######

def jaccard_sim(sets:List[Set]) -> np.array:
    # Compute the Jaccard similarity index between all pairs of sets   
    jaccard_mat = np.zeros((len(sets), len(sets)))
    for i in range(len(sets)):
        for j in range(len(sets)):
            jaccard_mat[i, j] = len(sets[i].intersection(sets[j])) / len(sets[i].union(sets[j]))
    return jaccard_mat

def plot_heatmap(jaccard_mat:np.array, labels:list) -> None:
    ax = sns.heatmap(jaccard_mat, cmap='crest', annot=True, xticklabels=labels, yticklabels=labels)
    ax.tick_params(axis='x', labelrotation=-45)
    ax.xaxis.tick_top()


###### Community Detection ######

def community_detection(g:nx.Graph) -> None:
    g_undirected = g.to_undirected()
    # Compute the communities using the Louvain algorithm
    partition = community.best_partition(g_undirected)
    # Print the communities
    for i, comm in enumerate(set(partition.values())):
        members = [node for node in partition.keys() if partition[node] == comm]
        print(f"Community {i}: {members}")


###### Save Neighbors ######

def save_node_neighbors(g:nx.Graph, node_type:str, path:str) -> None:
    target_nodes = [n for n, attr in g.nodes(data=True) if attr.get('node_type') == node_type]
    # find the neighbors of each target node
    neighbors_dict = {}
    for node in target_nodes:
        neighbors = set(g.neighbors(node))
        neighbors_dict[node] = neighbors
    # Convert dictionary to list of tuples
    neighbor_list = list(neighbors_dict.items())
    # Sort list based on length of neighbor set
    sorted_list = sorted(neighbor_list, key=lambda x: len(x[1]), reverse=True)

    rows = []
    for node, neighbors in sorted_list:
        for neighbor in neighbors:
            rows.append([node, neighbor, g.get_edge_data(node, neighbor)['weight']])
  
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'{node_type}', 'Recipient', 'Normalized Weight'])
        writer.writerows(rows)


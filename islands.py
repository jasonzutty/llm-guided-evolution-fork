import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
import math
import heapq
import shutil
import os

'''
To use this class for migration, at the end of each generation: 
1) Create Individual objects (class defined below) for each individual (inputting the individual file path and ranked fitness)
2) Create Island objects (class defined below) for each island (inputting the island folder path and list of individual objects under the island)
3) Generate a graph to represent the island migration topology using the generate_graph_topology method (input the list of islands); this can be reused in future generations
4) Perform island migration by running the migrate function (input the island map and graph topology)

Note that at the time of this comment, there is still work to be done to ensure all of these steps are functional
'''

class Topology(Enum):
    RING = "Ring"
    FULL = "Fully Connected"
    BROADCAST = "Broadcast"
    LATTICE = "Lattice"
    TORUS = "Torus"

class Individual:
    def __init__(self, name, rank):
        # This is the filename of the individual
        self.name = name
        self.rank = rank

    def __lt__(self, other):
        # For heap sorting by lowest rank
        return self.rank < other.rank

class Island:
    def __init__(self, path: str, individuals: list[Individual]):
        self.path = path
        self.individuals = individuals
        heapq.heapify(self.individuals)
    
    def remove_best(self) -> Individual:
        return heapq.heappop(self.individuals)
    
    def add_individual(self, individual) -> None:
        heapq.heappush(self.individuals, individual)

def generate_graph_topology(islands, topology, n=1, w=1) -> nx.Graph:
    """
    Generates a graph of islands based on the specified topology.

    Parameters:
        islands (list): List of Island objects.
        topology (Topology): The desired topology (Topology.RING, Topology.FULLY_CONNECTED, Topology.BROADCAST,
                            Topology.LATTICE, topology.TORUS).
        n (int, optional): For RING/TORUS topologies, specifies how many neighbors each node connects to on each side.
                           Default is 1 for a simple ring/torus.
        w (int, optional): Weight of all nodes in the generated graph, represents number of individuals migrated each migration cycle

    Returns:
        nx.Graph: A NetworkX graph object representing the islands and their connections.
    """
    # Using an undirected graph for current topologies, can be expanded if topologies use directed edges
    G = nx.Graph()
    num_islands = len(islands)

    for island in islands:
        G.add_node(island.path)

    if topology == Topology.RING:
        # Note any n greater than |islands|/2 is a fully connected graph
        if n < 1 or n >= num_islands // 2:
            raise ValueError(f"Invalid n for RING topology. Must be 1 <= n < {num_islands // 2}.")

        for i in range(num_islands):
            for j in range(1, n + 1):
                # Connect each node to its j-th neighbor on both sides
                G.add_edge(islands[i].path, islands[(i + j) % num_islands].path, weight=w)
                G.add_edge(islands[i].path, islands[(i - j) % num_islands].path, weight=w)

    elif topology == Topology.FULL:
        for i in range(num_islands):
            for j in range(i + 1, num_islands):
                G.add_edge(islands[i].path, islands[j].path, weight=w)

    elif topology == Topology.BROADCAST:
        central_island = islands[0]  # The first island acts as the central hub
        for i in range(1, num_islands):
            G.add_edge(central_island.path, islands[i].path, weight=w)
    elif topology == Topology.LATTICE:
        # Ensure n is a perfect square
        if num_islands < 1 or (math.isqrt(num_islands))**2 != num_islands:
            raise ValueError(f"Invalid n for LATTICE topology. Must be a positive perfect square.")

        # Calculate grid dimensions (rows and columns)
        grid_size = math.isqrt(num_islands)

        # Create lattice connections (4 neighbors for interior nodes)
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate the node index in the grid
                current_island = islands[i * grid_size + j]

                # Connect to the right neighbor
                if j < grid_size - 1:  # Avoid out of bounds for right neighbor
                    right_island = islands[i * grid_size + (j + 1)]
                    G.add_edge(current_island.path, right_island.path, weight=w)

                # Connect to the bottom neighbor
                if i < grid_size - 1:  # Avoid out of bounds for bottom neighbor
                    bottom_island = islands[(i + 1) * grid_size + j]
                    G.add_edge(current_island.path, bottom_island.path, weight=w)
    elif topology == Topology.TORUS:
        # Ensure n is positive/even
        if num_islands < 1 or num_islands % 2 == 1:
            raise ValueError(f"Invalid n for LATTICE topology. Must be a positive even integer.")

        i2 = num_islands // 2

        r1 = islands[:i2]
        r2 = islands[i2:]

        # Create Single n-Ring topologies
        for i in range(len(r1)):
            for j in range(1, n + 1):
                G.add_edge(r1[i].path, r1[(i + j) % (num_islands // 2)].path, weight=w)
                G.add_edge(r1[i].path, r1[(i - j) % (num_islands // 2)].path, weight=w)

        for i in range(len(r2)):
            for j in range(1, n + 1):
                G.add_edge(r2[i].path, r2[(i + j) % (num_islands // 2)].path, weight=w)
                G.add_edge(r2[i].path, r2[(i - j) % (num_islands // 2)].path, weight=w)

        for i in range(len(r1)):
            G.add_edge(r1[i].path, r2[i].path, weight=w)
            G.add_edge(r2[i].path, r1[i].path, weight=w)

    else:
        raise ValueError("Invalid topology. Use values from the Topology enum.")

    return G

def migrate(topology: nx.Graph, islands_list: list[Island]) -> None:
    """
    Migrates the most fit individuals across islands based on the specified topology.

    Parameters:
        topology (nx.graph): The desired topology represented as a weighted graph
        islands (dict): Map of Island object values mapped to a str ID key.

    Returns:
        nx.Graph: A NetworkX graph object representing the islands and their connections.
    """
    # Map of each island to a list of individuals migrating to it (and their source island)
    islands: dict[str, Island] = {i.path: i for i in islands_list}

    migration: dict[Island, list[tuple[Individual, Island]]] = {island: [] for island in islands.keys()}

    for i, j, d in topology.edges(data=True):
        for k in range(d['weight']):
            migration[i].append((islands[j].remove_best(), j))
            migration[j].append((islands[i].remove_best(), i))

    for i in migration:
        for j in migration[i]:
            #move_file(j[0].name, j[1].path, i.path)
            islands[i].add_individual(j[0])

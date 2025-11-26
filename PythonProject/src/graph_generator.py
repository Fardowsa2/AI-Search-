# src/graph_generator.py
import networkx as nx
import random
import numpy as np
from typing import List, Tuple


class GraphGenerator:
    @staticmethod
    def generate_random_graph(n_nodes: int, branching_factor: float,
                              weight_range: Tuple[float, float] = (1, 10),
                              seed: int = None) -> nx.Graph:
        """Generate random connected graph with specified parameters"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        G = nx.Graph()

        # Add nodes
        for i in range(n_nodes):
            G.add_node(f"Node_{i}")

        # Ensure connectivity - create a spanning tree first
        nodes = list(G.nodes())
        random.shuffle(nodes)

        for i in range(1, len(nodes)):
            parent = random.choice(nodes[:i])
            weight = random.uniform(weight_range[0], weight_range[1])
            G.add_edge(parent, nodes[i], weight=weight)

        # Add additional edges to achieve desired branching factor
        target_edges = int(branching_factor * n_nodes)
        current_edges = G.number_of_edges()

        while current_edges < target_edges and current_edges < (n_nodes * (n_nodes - 1)) // 2:
            u, v = random.sample(nodes, 2)
            if not G.has_edge(u, v):
                weight = random.uniform(weight_range[0], weight_range[1])
                G.add_edge(u, v, weight=weight)
                current_edges += 1

        # Add random positions for visualization
        for node in G.nodes():
            G.nodes[node]['pos'] = (random.random() * 100, random.random() * 100)

        return G

    @staticmethod
    def generate_grid_world(size: int, obstacle_density: float,
                            connectivity: str = "4", weighted: bool = False,
                            seed: int = None) -> nx.Graph:
        """Generate grid world with obstacles"""
        if seed is not None:
            random.seed(seed)

        G = nx.Graph()

        # Add grid nodes
        for i in range(size):
            for j in range(size):
                node_id = f"{i},{j}"
                G.add_node(node_id, pos=(i, j), obstacle=False)

        # Add obstacles
        all_nodes = list(G.nodes())
        n_obstacles = int(obstacle_density * len(all_nodes))
        obstacle_nodes = random.sample(all_nodes, n_obstacles)

        for node in obstacle_nodes:
            G.nodes[node]['obstacle'] = True

        # Add edges based on connectivity
        for i in range(size):
            for j in range(size):
                node_id = f"{i},{j}"
                if G.nodes[node_id]['obstacle']:
                    continue

                # 4-connectivity (up, down, left, right)
                neighbors = []
                if i > 0:
                    neighbors.append(f"{i - 1},{j}")
                if i < size - 1:
                    neighbors.append(f"{i + 1},{j}")
                if j > 0:
                    neighbors.append(f"{i},{j - 1}")
                if j < size - 1:
                    neighbors.append(f"{i},{j + 1}")

                # 8-connectivity (add diagonals)
                if connectivity == "8":
                    if i > 0 and j > 0:
                        neighbors.append(f"{i - 1},{j - 1}")
                    if i > 0 and j < size - 1:
                        neighbors.append(f"{i - 1},{j + 1}")
                    if i < size - 1 and j > 0:
                        neighbors.append(f"{i + 1},{j - 1}")
                    if i < size - 1 and j < size - 1:
                        neighbors.append(f"{i + 1},{j + 1}")

                # Add edges to non-obstacle neighbors
                for neighbor in neighbors:
                    if not G.nodes[neighbor]['obstacle']:
                        weight = random.uniform(1, 10) if weighted else 1.0
                        G.add_edge(node_id, neighbor, weight=weight)

        return G
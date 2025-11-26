# src/heuristics.py 
import math
import networkx as nx


def euclidean_h(G: nx.Graph, goal: str):
    """Heuristic h(n) = Euclidean distance from node n to the goal using stored positions."""
    goal_pos = G.nodes[goal].get("pos")

    def h(n: str) -> float:
        pn = G.nodes[n].get("pos")
        if not goal_pos or not pn:
            return 0.0
        dx = pn[0] - goal_pos[0]
        dy = pn[1] - goal_pos[1]
        return math.hypot(dx, dy)

    return h


def manhattan_h(G: nx.Graph, goal: str):
    """Manhattan distance heuristic for grid worlds"""
    goal_pos = G.nodes[goal].get("pos")

    def h(n: str) -> float:
        pn = G.nodes[n].get("pos")
        if not goal_pos or not pn:
            return 0.0
        return abs(pn[0] - goal_pos[0]) + abs(pn[1] - goal_pos[1])

    return h


def chebyshev_h(G: nx.Graph, goal: str):
    """Chebyshev distance heuristic for 8-connected grid worlds"""
    goal_pos = G.nodes[goal].get("pos")

    def h(n: str) -> float:
        pn = G.nodes[n].get("pos")
        if not goal_pos or not pn:
            return 0.0
        return max(abs(pn[0] - goal_pos[0]), abs(pn[1] - goal_pos[1]))

    return h


def zero_h(G: nx.Graph, goal: str):
    """Zero heuristic (falls back to uniform cost search)"""

    def h(n: str) -> float:
        return 0.0

    return h

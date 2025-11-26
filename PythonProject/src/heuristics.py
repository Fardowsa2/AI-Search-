# src/heuristics.py - FIXED
import math
import networkx as nx


def euclidean_h(G: nx.Graph, goal: str):
    """Heuristic h(n) = Euclidean distance from node n to the goal using stored positions."""
    goal_data = G.nodes.get(goal, {})
    goal_pos = goal_data.get("pos") if goal_data else None

    def h(n: str) -> float:
        node_data = G.nodes.get(n, {})
        pn = node_data.get("pos") if node_data else None
        if not goal_pos or not pn:
            return 0.0
        dx = pn[0] - goal_pos[0]
        dy = pn[1] - goal_pos[1]
        return math.hypot(dx, dy)

    return h


def manhattan_h(G: nx.Graph, goal: str):
    """Manhattan distance heuristic for grid worlds"""
    goal_data = G.nodes.get(goal, {})
    goal_pos = goal_data.get("pos") if goal_data else None

    def h(n: str) -> float:
        node_data = G.nodes.get(n, {})
        pn = node_data.get("pos") if node_data else None
        if not goal_pos or not pn:
            return 0.0
        return abs(pn[0] - goal_pos[0]) + abs(pn[1] - goal_pos[1])

    return h


def chebyshev_h(G: nx.Graph, goal: str):
    """Chebyshev distance heuristic for 8-connected grid worlds"""
    goal_data = G.nodes.get(goal, {})
    goal_pos = goal_data.get("pos") if goal_data else None

    def h(n: str) -> float:
        node_data = G.nodes.get(n, {})
        pn = node_data.get("pos") if node_data else None
        if not goal_pos or not pn:
            return 0.0
        return max(abs(pn[0] - goal_pos[0]), abs(pn[1] - goal_pos[1]))

    return h


def zero_h(G: nx.Graph, goal: str):
    """Zero heuristic (falls back to uniform cost search)"""

    def h(n: str) -> float:
        return 0.0

    return h


# Add this function to get the appropriate heuristic
def get_heuristic(G: nx.Graph, goal: str, heuristic_name: str = "euclidean"):
    """Get the specified heuristic function"""
    heuristics = {
        "euclidean": euclidean_h,
        "manhattan": manhattan_h,
        "chebyshev": chebyshev_h,
        "zero": zero_h
    }

    heuristic_func = heuristics.get(heuristic_name.lower(), euclidean_h)
    return heuristic_func(G, goal)
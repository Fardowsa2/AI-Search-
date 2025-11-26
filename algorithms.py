# src/algorithms.py
"""
Search Algorithm Implementations

This module contains implementations of five search algorithms:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Iterative Deepening DFS (IDDFS)
- Greedy Best-First Search
- A* Search

All algorithms return both the path found and comprehensive metrics including
runtime, memory usage, nodes expanded, and solution quality.
"""

from collections import deque
import heapq
import time
import tracemalloc
from typing import Dict, List, Tuple, Callable, Optional
import networkx as nx

# Type alias for search results
SearchResult = Tuple[List[str], Dict[str, float]]


def reconstruct_path(parents: Dict[str, Optional[str]], start: str, goal: str) -> List[str]:
    """
    Reconstruct the path from start to goal using parent pointers.

    Args:
        parents: Dictionary mapping each node to its parent in the search tree
        start: Starting node
        goal: Goal node

    Returns:
        List of nodes from start to goal, or empty list if no path exists
    """
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parents.get(cur)
    path.reverse()
    return path if path and path[0] == start else []


def path_cost(G: nx.Graph, path: List[str]) -> float:
    """
    Calculate the total cost of a path in the graph.

    Args:
        G: Graph with edge weights
        path: List of nodes representing the path

    Returns:
        Total cost of the path, or 0 if path is invalid
    """
    if not path or len(path) < 2:
        return 0.0
    return sum(G[a][b].get("weight", 1.0) for a, b in zip(path, path[1:]))


def run_with_metrics(fn: Callable[[], Tuple[List[str], Dict]], frontier_sizes: List[int]) -> SearchResult:
    """
    Wrapper function to measure runtime and memory usage of search algorithms.

    Args:
        fn: Search algorithm function to execute
        frontier_sizes: List to track frontier size at each step

    Returns:
        Tuple of (path, metrics dictionary)
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    path, meta = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    meta.setdefault("runtime_sec", dt)
    meta.setdefault("peak_tracemalloc_bytes", peak)
    if frontier_sizes:
        meta["frontier_peak_size"] = max(frontier_sizes)

    return path, meta


def bfs(G: nx.Graph, start: str, goal: str) -> SearchResult:
    """
    Breadth-First Search algorithm.

    Properties:
    - Complete: Yes
    - Optimal: Yes (for uniform cost)
    - Time Complexity: O(b^d)
    - Space Complexity: O(b^d)

    Args:
        G: Graph to search
        start: Starting node
        goal: Goal node

    Returns:
        Tuple of (path, metrics)
    """
    parents = {start: None}
    q = deque([start])
    explored = {start}
    frontier_sizes = [1]
    nodes_expanded = 0

    def _run():
        nonlocal nodes_expanded
        while q:
            frontier_sizes.append(len(q))
            u = q.popleft()
            nodes_expanded += 1

            if u == goal:
                break

            for v in G.neighbors(u):
                if v not in explored:
                    explored.add(v)
                    parents[v] = u
                    q.append(v)

        path = reconstruct_path(parents, start, goal)
        meta = {
            "algorithm": "BFS",
            "nodes_expanded": nodes_expanded,
            "path_cost": path_cost(G, path),
            "solution_depth": len(path) - 1 if path else -1
        }
        return path, meta

    return run_with_metrics(_run, frontier_sizes)


def dfs(G: nx.Graph, start: str, goal: str) -> SearchResult:
    """
    Depth-First Search algorithm.

    Properties:
    - Complete: No (can get stuck in infinite paths)
    - Optimal: No
    - Time Complexity: O(b^m) where m is maximum depth
    - Space Complexity: O(bm)

    Args:
        G: Graph to search
        start: Starting node
        goal: Goal node

    Returns:
        Tuple of (path, metrics)
    """
    stack = [start]
    parents = {start: None}
    explored = {start}
    frontier_sizes = [1]
    nodes_expanded = 0

    def _run():
        nonlocal nodes_expanded
        while stack:
            frontier_sizes.append(len(stack))
            u = stack.pop()
            nodes_expanded += 1

            if u == goal:
                break

            for v in G.neighbors(u):
                if v not in explored:
                    explored.add(v)
                    parents[v] = u
                    stack.append(v)

        path = reconstruct_path(parents, start, goal)
        meta = {
            "algorithm": "DFS",
            "nodes_expanded": nodes_expanded,
            "path_cost": path_cost(G, path),
            "solution_depth": len(path) - 1 if path else -1
        }
        return path, meta

    return run_with_metrics(_run, frontier_sizes)


def iddfs(G: nx.Graph, start: str, goal: str, max_depth: int = 50) -> SearchResult:
    """
    Iterative Deepening Depth-First Search.

    Properties:
    - Complete: Yes
    - Optimal: Yes (for uniform cost)
    - Time Complexity: O(b^d)
    - Space Complexity: O(bd)

    Args:
        G: Graph to search
        start: Starting node
        goal: Goal node
        max_depth: Maximum depth to search

    Returns:
        Tuple of (path, metrics)
    """
    nodes_expanded_total = 0
    frontier_peak = 0
    parents_global: Dict[str, Optional[str]] = {}

    def dls(limit: int) -> bool:
        """Depth-Limited Search helper function."""
        nonlocal nodes_expanded_total, frontier_peak, parents_global
        stack = [(start, 0)]
        parents = {start: None}
        visited = {start}

        while stack:
            frontier_peak = max(frontier_peak, len(stack))
            u, d = stack.pop()
            nodes_expanded_total += 1

            if u == goal:
                parents_global = parents
                return True

            if d < limit:
                for v in G.neighbors(u):
                    if v not in visited:
                        visited.add(v)
                        parents[v] = u
                        stack.append((v, d + 1))
        return False

    def _run():
        for depth in range(max_depth + 1):
            if dls(depth):
                path = reconstruct_path(parents_global, start, goal)
                meta = {
                    "algorithm": "IDDFS",
                    "nodes_expanded": nodes_expanded_total,
                    "path_cost": path_cost(G, path),
                    "solution_depth": len(path) - 1 if path else -1,
                    "max_depth_reached": depth
                }
                return path, meta

        return [], {
            "algorithm": "IDDFS",
            "nodes_expanded": nodes_expanded_total,
            "solution_depth": -1
        }

    return run_with_metrics(_run, [frontier_peak])


def greedy_best_first(G: nx.Graph, start: str, goal: str, h) -> SearchResult:
    """
    Greedy Best-First Search algorithm.

    Properties:
    - Complete: No (can get stuck in infinite paths)
    - Optimal: No
    - Time Complexity: O(b^m)
    - Space Complexity: O(b^m)

    Args:
        G: Graph to search
        start: Starting node
        goal: Goal node
        h: Heuristic function h(n) -> float

    Returns:
        Tuple of (path, metrics)
    """
    pq = [(h(start), start)]
    parents = {start: None}
    visited = set()
    nodes_expanded = 0
    frontier_sizes = [1]

    def _run():
        nonlocal nodes_expanded
        while pq:
            frontier_sizes.append(len(pq))
            _, u = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)
            nodes_expanded += 1

            if u == goal:
                break

            for v in G.neighbors(u):
                if v not in visited:
                    parents.setdefault(v, u)
                    heapq.heappush(pq, (h(v), v))

        path = reconstruct_path(parents, start, goal)
        meta = {
            "algorithm": "GreedyBestFirst",
            "nodes_expanded": nodes_expanded,
            "path_cost": path_cost(G, path),
            "solution_depth": len(path) - 1 if path else -1
        }
        return path, meta

    return run_with_metrics(_run, frontier_sizes)


def astar(G: nx.Graph, start: str, goal: str, h) -> SearchResult:
    """
    A* Search algorithm.

    Properties:
    - Complete: Yes
    - Optimal: Yes (with admissible heuristic)
    - Time Complexity: O(b^d)
    - Space Complexity: O(b^d)

    Args:
        G: Graph to search
        start: Starting node
        goal: Goal node
        h: Heuristic function h(n) -> float

    Returns:
        Tuple of (path, metrics)
    """
    open_pq = [(h(start), 0.0, start)]  # (f = g + h, g, node)
    g = {start: 0.0}
    parents = {start: None}
    closed = set()
    nodes_expanded = 0
    frontier_sizes = [1]

    def _run():
        nonlocal nodes_expanded
        while open_pq:
            frontier_sizes.append(len(open_pq))
            _, g_u, u = heapq.heappop(open_pq)

            if u in closed:
                continue
            nodes_expanded += 1

            if u == goal:
                break

            closed.add(u)
            for v in G.neighbors(u):
                w = G[u][v].get("weight", 1.0)
                tentative = g_u + w

                if v not in g or tentative < g[v]:
                    g[v] = tentative
                    parents[v] = u
                    heapq.heappush(open_pq, (tentative + h(v), tentative, v))

        path = reconstruct_path(parents, start, goal)
        meta = {
            "algorithm": "A*",
            "nodes_expanded": nodes_expanded,
            "path_cost": path_cost(G, path),
            "solution_depth": len(path) - 1 if path else -1
        }
        return path, meta

    return run_with_metrics(_run, frontier_sizes)

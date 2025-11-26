# src/benchmark.py 
import statistics as stats
import pandas as pd
import networkx as nx
from typing import Callable, List, Tuple, Dict
import time
import tracemalloc
import matplotlib.pyplot as plt
import seaborn as sns

from algorithms import bfs, dfs, iddfs, greedy_best_first, astar
from heuristics import euclidean_h, manhattan_h, chebyshev_h, zero_h


class BenchmarkSuite:
    def __init__(self):
        self.results = []

    def single_run(self, G: nx.Graph, start: str, goal: str, algorithm: str,
                   heuristic_name: str = "euclidean") -> Dict:
        """Run single algorithm and return detailed metrics"""
        h_func = self._get_heuristic(G, goal, heuristic_name)

        algorithms = {
            "BFS": lambda: bfs(G, start, goal),
            "DFS": lambda: dfs(G, start, goal),
            "IDDFS": lambda: iddfs(G, start, goal),
            "Greedy": lambda: greedy_best_first(G, start, goal, h_func),
            "A*": lambda: astar(G, start, goal, h_func),
        }

        if algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        path, metrics = algorithms[algorithm]()

        return {
            'algorithm': algorithm,
            'heuristic': heuristic_name,
            'path_found': len(path) > 0,
            'path_length': len(path) if path else 0,
            'path_cost': metrics.get('path_cost', float('inf')),
            'nodes_expanded': metrics.get('nodes_expanded', 0),
            'runtime_sec': metrics.get('runtime_sec', 0),
            'peak_memory_bytes': metrics.get('peak_tracemalloc_bytes', 0),
            'frontier_peak_size': metrics.get('frontier_peak_size', 0),
            'solution_depth': metrics.get('solution_depth', -1)
        }

    def batch_compare(self, G: nx.Graph, start: str, goal: str,
                      algorithms: List[str] = None, repeats: int = 5,
                      heuristic_name: str = "euclidean") -> pd.DataFrame:
        """Compare multiple algorithms with statistical analysis"""
        if algorithms is None:
            algorithms = ["BFS", "DFS", "IDDFS", "Greedy", "A*"]

        rows = []
        for algo in algorithms:
            runtimes, memories, nodes_expanded, path_costs, path_lengths = [], [], [], [], []
            successes = 0

            for _ in range(repeats):
                result = self.single_run(G, start, goal, algo, heuristic_name)

                if result['path_found']:
                    successes += 1
                    runtimes.append(result['runtime_sec'])
                    memories.append(result['peak_memory_bytes'])
                    nodes_expanded.append(result['nodes_expanded'])
                    path_costs.append(result['path_cost'])
                    path_lengths.append(result['path_length'])
                else:
                    # Use large values for failed paths for comparison
                    runtimes.append(float('inf'))
                    memories.append(float('inf'))
                    nodes_expanded.append(float('inf'))
                    path_costs.append(float('inf'))
                    path_lengths.append(0)

            success_rate = successes / repeats

            # Calculate statistics (handle cases with all failures)
            if successes > 0:
                runtime_mean = stats.mean([r for r in runtimes if r != float('inf')])
                runtime_std = stats.pstdev([r for r in runtimes if r != float('inf')]) if successes > 1 else 0
                memory_mean = stats.mean([m for m in memories if m != float('inf')])
                nodes_mean = stats.mean(nodes_expanded)
                cost_mean = stats.mean(path_costs)
            else:
                runtime_mean = runtime_std = memory_mean = nodes_mean = cost_mean = float('inf')

            rows.append({
                'algorithm': algo,
                'success_rate': success_rate,
                'runtime_mean_s': runtime_mean,
                'runtime_std_s': runtime_std,
                'memory_mean_bytes': memory_mean,
                'nodes_expanded_mean': nodes_mean,
                'path_cost_mean': cost_mean,
                'path_length_mean': stats.mean(path_lengths) if successes > 0 else 0
            })

        return pd.DataFrame(rows)

    def complexity_analysis(self, graph_generator: Callable,
                            complexity_settings: List[Dict],
                            start_goal_generator: Callable,
                            algorithms: List[str] = None) -> pd.DataFrame:
        """Analyze algorithm performance across different complexity settings"""
        all_results = []

        for setting in complexity_settings:
            print(f"Testing setting: {setting}")

            for seed in range(5):  # 5 different seeds per setting
                G = graph_generator(seed=seed, **setting)
                start, goal = start_goal_generator(G)

                df_batch = self.batch_compare(G, start, goal, algorithms, repeats=1)

                for _, row in df_batch.iterrows():
                    result = row.to_dict()
                    result.update(setting)
                    result['seed'] = seed
                    all_results.append(result)

        return pd.DataFrame(all_results)

    def _get_heuristic(self, G: nx.Graph, goal: str, name: str):
        heuristics = {
            "euclidean": euclidean_h,
            "manhattan": manhattan_h,
            "chebyshev": chebyshev_h,
            "zero": zero_h
        }
        return heuristics.get(name, euclidean_h)(G, goal)

    def plot_comparison(self, df: pd.DataFrame, metrics: List[str] = None):
        """Create comparison plots"""
        if metrics is None:
            metrics = ['runtime_mean_s', 'memory_mean_bytes', 'nodes_expanded_mean']

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            # Filter out infinite values for plotting
            plot_df = df[df[metric] < float('inf')]

            if not plot_df.empty:
                sns.barplot(data=plot_df, x='algorithm', y=metric, ax=axes[i])
                axes[i].set_title(f'{metric} Comparison')
                axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate textual analysis report"""
        report = "BENCHMARK REPORT\n"
        report += "=" * 50 + "\n\n"

        # Find best algorithm for each metric
        metrics = ['runtime_mean_s', 'memory_mean_bytes', 'nodes_expanded_mean']

        for metric in metrics:
            valid_df = df[df[metric] < float('inf')]
            if not valid_df.empty:
                best = valid_df.loc[valid_df[metric].idxmin()]
                report += f"Best {metric}: {best['algorithm']} ({best[metric]:.4f})\n"

        report += "\nANALYSIS:\n"
        report += "- BFS: Complete, optimal for uniform cost, high memory\n"
        report += "- DFS: Incomplete for infinite spaces, low memory\n"
        report += "- IDDFS: Complete, optimal, good memory, slower\n"
        report += "- Greedy: Fast but not optimal, good for simple heuristics\n"
        report += "- A*: Complete and optimal with admissible heuristic\n"

        return report

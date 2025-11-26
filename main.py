# src/main.py
from src.algorithms import bfs, dfs, iddfs, greedy_best_first, astar, path_cost
from src.graph_loader import load_set1_graph
from src.graph_generator import GraphGenerator
from src.heuristics import get_heuristic
from src.visualization import draw_path
from src.benchmark import BenchmarkSuite
import networkx as nx
import matplotlib.pyplot as plt


def display_sample_nodes(G, count=10):
    """Display sample nodes from the graph"""
    nodes = list(G.nodes())
    print(f"\nGraph has {len(nodes)} nodes")
    if len(nodes) <= count:
        print(f"All nodes: {nodes}")
    else:
        print(f"Sample nodes: {nodes[:count]}")
        print(f"... and {len(nodes) - count} more")


def single_algorithm_mode(G):
    """Run single algorithm with visualization"""
    nodes = list(G.nodes())
    display_sample_nodes(G)

    while True:
        print("\n" + "=" * 50)
        print("SINGLE ALGORITHM MODE")
        print("=" * 50)

        # Get start and goal
        start = input("Start node (or 'back' to return): ").strip()
        if start.lower() == 'back':
            return

        goal = input("Goal node (or 'back' to return): ").strip()
        if goal.lower() == 'back':
            return

        if start not in G or goal not in G:
            print(f"Error: One or both nodes not found in graph. Please use exact names.")
            continue

        # Choose algorithm
        print("\nAvailable algorithms:")
        print("1. BFS (Breadth-First Search)")
        print("2. DFS (Depth-First Search)")
        print("3. IDDFS (Iterative Deepening DFS)")
        print("4. Greedy Best-First Search")
        print("5. A* Search")

        algo_choice = input("Choose algorithm (1-5, default 5): ").strip() or "5"

        algorithms = {
            "1": "BFS", "2": "DFS", "3": "IDDFS", "4": "Greedy", "5": "A*"
        }
        algo = algorithms.get(algo_choice, "A*")

        # Additional parameters for IDDFS
        max_depth = 50
        if algo == "IDDFS":
            depth_input = input(f"Max depth for IDDFS (default {max_depth}): ").strip()
            if depth_input.isdigit():
                max_depth = int(depth_input)

        # Choose heuristic for informed searches
        heuristic_name = "euclidean"
        if algo in ["Greedy", "A*"]:
            print("\nAvailable heuristics:")
            print("1. Euclidean distance")
            print("2. Manhattan distance")
            print("3. Chebyshev distance")
            print("4. Zero heuristic (uniform cost)")

            heuristic_choice = input("Choose heuristic (1-4, default 1): ").strip() or "1"
            heuristic_map = {
                "1": "euclidean",
                "2": "manhattan",
                "3": "chebyshev",
                "4": "zero"
            }
            heuristic_name = heuristic_map.get(heuristic_choice, "euclidean")
            print(f"Using {heuristic_name} heuristic...")

        # Run algorithm
        print(f"\nRunning {algo} with {heuristic_name} heuristic...")

        # Get the heuristic function
        h = get_heuristic(G, goal, heuristic_name)

        algorithm_functions = {
            "BFS": lambda: bfs(G, start, goal),
            "DFS": lambda: dfs(G, start, goal),
            "IDDFS": lambda: iddfs(G, start, goal, max_depth=max_depth),
            "Greedy": lambda: greedy_best_first(G, start, goal, h),
            "A*": lambda: astar(G, start, goal, h),
        }

        path, metrics = algorithm_functions[algo]()

        # Display results
        print(f"\n{'=' * 50}")
        print(f"RESULTS: {algo} from {start} to {goal}")
        print(f"{'=' * 50}")
        print(f"Path found: {'Yes' if path else 'No'}")
        if path:
            print(f"Path: {' -> '.join(path)}")
            print(f"Path length: {len(path)} nodes")
            print(f"Path cost: {metrics.get('path_cost', 'N/A'):.4f}")
        print(f"Nodes expanded: {metrics.get('nodes_expanded', 'N/A')}")
        print(f"Solution depth: {metrics.get('solution_depth', 'N/A')}")
        print(f"Runtime: {metrics.get('runtime_sec', 'N/A'):.6f} seconds")
        print(f"Peak memory: {metrics.get('peak_tracemalloc_bytes', 'N/A'):,} bytes")
        if 'frontier_peak_size' in metrics:
            print(f"Max frontier size: {metrics.get('frontier_peak_size', 'N/A')}")

        # Ask for visualization
        viz = input("\nShow visualization? (y/n): ").strip().lower()
        if viz == 'y':
            try:
                draw_path(G, path, title=f"{algo} Path: {start} → {goal}")
            except Exception as e:
                print(f"Visualization error: {e}")
                print("But the search completed successfully!")

        # Ask to continue
        print(f"\n{'=' * 50}")
        again = input("Run another search with same graph? (y/n): ").strip().lower()
        if again != 'y':
            break


def batch_comparison_mode(G):
    """Run batch comparison of all algorithms"""
    nodes = list(G.nodes())
    display_sample_nodes(G)

    while True:
        print("\n" + "=" * 50)
        print("BATCH COMPARISON MODE")
        print("=" * 50)

        # Get start and goal
        start = input("Start node (or 'back' to return): ").strip()
        if start.lower() == 'back':
            return

        goal = input("Goal node (or 'back' to return): ").strip()
        if goal.lower() == 'back':
            return

        if start not in G or goal not in G:
            print(f"Error: One or both nodes not found in graph.")
            continue

        # Choose heuristic
        print("\nAvailable heuristics for informed searches:")
        print("1. Euclidean distance")
        print("2. Manhattan distance")
        print("3. Chebyshev distance")
        print("4. Zero heuristic (uniform cost)")

        heuristic_choice = input("Choose heuristic (1-4, default 1): ").strip() or "1"
        heuristics = {
            "1": "euclidean", "2": "manhattan", "3": "chebyshev", "4": "zero"
        }
        heuristic_name = heuristics.get(heuristic_choice, "euclidean")

        # Choose number of repeats
        repeats = input("Number of runs per algorithm (default 5): ").strip()
        repeats = int(repeats) if repeats.isdigit() else 5

        print(f"\nRunning batch comparison with {repeats} repeats...")
        benchmark_suite = BenchmarkSuite()
        df = benchmark_suite.batch_compare(G, start, goal, repeats=repeats, heuristic_name=heuristic_name)

        print(f"\n{'=' * 60}")
        print(f"BENCHMARK RESULTS: {start} → {goal} ({repeats} runs)")
        print(f"{'=' * 60}")
        print(df.to_string(index=False, float_format='%.6f'))

        # Generate and display report
        report = benchmark_suite.generate_report(df)
        print(f"\n{report}")

        # Create visualizations
        viz = input("\nShow comparison charts? (y/n): ").strip().lower()
        if viz == 'y':
            benchmark_suite.plot_comparison(df)

        # Save results option
        save = input("\nSave results to CSV? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("Filename (default 'benchmark_results.csv'): ").strip() or "benchmark_results.csv"
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")

        # Ask to continue
        print(f"\n{'=' * 50}")
        again = input("Run another batch comparison with same graph? (y/n): ").strip().lower()
        if again != 'y':
            break


def graph_selection_mode():
    """Handle graph selection with looping"""
    current_graph = None

    while True:
        print("\n" + "=" * 50)
        print("GRAPH SELECTION")
        print("=" * 50)
        print("1. Use preset graph (Kansas cities)")
        print("2. Generate random graph")
        print("3. Generate grid world")
        print("4. Use current graph")
        print("5. Back to main menu")

        choice = input("Choose option (1-5): ").strip()

        if choice == "1":
            try:
                G = load_set1_graph("data/Adjacencies.txt", "data/coordinates.csv")
                current_graph = G
                print(f"✓ Loaded Kansas cities graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                return G
            except Exception as e:
                print(f"Error loading graph: {e}")
                continue

        elif choice == "2":
            try:
                n_nodes = int(input("Number of nodes (default 20): ") or 20)
                branching = float(input("Branching factor (default 1.5): ") or 1.5)
                seed = int(input("Random seed (default 42): ") or 42)

                G = GraphGenerator.generate_random_graph(
                    n_nodes=n_nodes,
                    branching_factor=branching,
                    seed=seed
                )
                current_graph = G
                print(f"✓ Generated random graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                return G
            except Exception as e:
                print(f"Error generating graph: {e}")
                continue

        elif choice == "3":
            try:
                size = int(input("Grid size (default 10): ") or 10)
                obstacles = float(input("Obstacle density (0-1, default 0.2): ") or 0.2)
                connectivity = input("Connectivity (4/8, default 4): ") or "4"
                weighted = input("Weighted edges? (y/n, default n): ").strip().lower() == 'y'
                seed = int(input("Random seed (default 42): ") or 42)

                G = GraphGenerator.generate_grid_world(
                    size=size,
                    obstacle_density=obstacles,
                    connectivity=connectivity,
                    weighted=weighted,
                    seed=seed
                )
                current_graph = G
                # Count non-obstacle nodes
                free_nodes = sum(1 for n in G.nodes() if not G.nodes[n].get('obstacle', False))
                print(f"✓ Generated {size}x{size} grid: {free_nodes} free nodes, {obstacles * 100}% obstacles")
                return G
            except Exception as e:
                print(f"Error generating grid: {e}")
                continue

        elif choice == "4":
            if current_graph is not None:
                print(f"✓ Using current graph: {current_graph.number_of_nodes()} nodes")
                return current_graph
            else:
                print("No graph loaded yet. Please select option 1, 2, or 3 first.")
                continue

        elif choice == "5":
            return None
        else:
            print("Invalid choice. Please try again.")


def main():
    """Main program with continuous looping"""
    print("🎯 AI Search Algorithm Lab")
    print("   Implement, Visualize, and Compare Search Algorithms")

    current_graph = None

    while True:
        print("\n" + "=" * 60)
        print("MAIN MENU")
        print("=" * 60)
        print("1. Select/Change Graph")
        if current_graph:
            print(f"   Current: {current_graph.number_of_nodes()} nodes, {current_graph.number_of_edges()} edges")
        else:
            print("   Current: No graph selected")
        print("2. Single Algorithm Search (with visualization)")
        print("3. Batch Algorithm Comparison (benchmarking)")
        print("4. Exit")
        print("=" * 60)

        choice = input("Choose option (1-4): ").strip()

        if choice == "1":
            # Graph selection
            G = graph_selection_mode()
            if G is not None:
                current_graph = G

        elif choice == "2":
            # Single algorithm mode
            if current_graph is None:
                print("Please select a graph first (option 1)")
                continue
            single_algorithm_mode(current_graph)

        elif choice == "3":
            # Batch comparison mode
            if current_graph is None:
                print("Please select a graph first (option 1)")
                continue
            batch_comparison_mode(current_graph)

        elif choice == "4":
            print("\nThank you for using the AI Search Algorithm Lab!")
            print("Goodbye! 👋")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()

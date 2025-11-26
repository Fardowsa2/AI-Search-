```markdown
# 🎯 AI Search Algorithms Lab

A comprehensive implementation, visualization, and benchmarking suite for five fundamental search algorithms in artificial intelligence.

## 📋 Overview

This project provides a complete framework for understanding, comparing, and visualizing search algorithms including:
- **Breadth-First Search (BFS)**
- **Depth-First Search (DFS)**
- **Iterative Deepening DFS (IDDFS)**
- **Greedy Best-First Search**
- **A* Search**

Featuring interactive visualization, comprehensive benchmarking, and support for multiple graph types including real-world geographic data.

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install networkx matplotlib pandas numpy seaborn

# Run the program
python src/main.py
```

### Immediate Demo
```bash
python src/main.py
# 1 → 1 (Select Kansas cities)
# 2 → Single algorithm mode  
# Start: Wichita
# Goal: Topeka
# Algorithm: A*
```

## 🏗️ Project Structure
```
ai-search-algorithms-lab/
├── src/
│   ├── main.py              # Main program interface
│   ├── algorithms.py        # All search algorithm implementations
│   ├── graph_loader.py      # Load Kansas cities dataset
│   ├── graph_generator.py   # Random graph and grid generation
│   ├── heuristics.py        # Heuristic functions
│   ├── benchmark.py         # Performance comparison
│   └── visualization.py     # Graph visualization
├── data/
│   ├── Adjacencies.txt      # Kansas cities road connections
│   └── coordinates.csv      # Geographic coordinates
└── requirements.txt
```

## ✨ Features

### 🔍 Algorithms
- **BFS**: Complete, optimal for uniform costs
- **DFS**: Memory-efficient but not complete
- **IDDFS**: Combines DFS memory with BFS completeness
- **Greedy**: Fast heuristic-based search
- **A***: Optimal informed search

### 🌐 Graph Types
- **Kansas Cities**: Real geographic data (46 cities)
- **Random Graphs**: Customizable size and connectivity
- **Grid Worlds**: Maze-like environments with obstacles

### 📊 Benchmarking
- Runtime and memory analysis
- Statistical comparison across algorithms
- Success rate calculations
- Visual charts and CSV export

### 🎨 Visualization
- Interactive graph display
- Path highlighting
- Color-coded node states
- Real-time algorithm animation

## 🎮 Usage

### Single Algorithm Search
```
MAIN MENU → "1" (Select Graph) → "1" (Kansas cities)
MAIN MENU → "2" (Single Algorithm)
Start: Wichita
Goal: Topeka
Algorithm: A*
```

### Batch Comparison
```
MAIN MENU → "1" (Select Graph) → "2" (Random graph, 50 nodes)
MAIN MENU → "3" (Batch Comparison)
Compare all 5 algorithms with statistical analysis
```

### Grid World
```
MAIN MENU → "1" (Select Graph) → "3" (Grid world)
Size: 15x15, Obstacles: 30%, Connectivity: 4
```

## 📈 Performance Summary

| Algorithm | Optimal | Complete | Time | Memory | Best For |
|-----------|---------|----------|------|--------|----------|
| BFS | ✅ | ✅ | Medium | High | Guaranteed optimal |
| DFS | ❌ | ❌ | Fast | Low | Memory constraints |
| IDDFS | ✅ | ✅ | Slow | Medium | Unknown depth |
| Greedy | ❌ | ❌ | Very Fast | Low | Quick solutions |
| A* | ✅ | ✅ | Fast | Medium | Optimal + efficient |

## 🛠️ Requirements

- Python 3.8+
- networkx >= 3.0
- matplotlib >= 3.5
- pandas >= 1.4
- numpy >= 1.21
- seaborn >= 0.11

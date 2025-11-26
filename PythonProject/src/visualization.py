# src/visualization.py - UPDATED
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider


class SearchVisualizer:
    def __init__(self, G, algorithms_dict):
        self.G = G
        self.algorithms = algorithms_dict
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)

        # Colors for different states
        self.colors = {
            'start': 'green',
            'goal': 'red',
            'frontier': 'yellow',
            'explored': 'lightblue',
            'current': 'orange',
            'obstacle': 'black',
            'path': 'purple',
            'normal': 'lightgray'
        }

        self.legend_patches = [
            patches.Patch(color=color, label=state)
            for state, color in self.colors.items()
        ]

        self.current_algorithm = None
        self.search_states = []
        self.current_step = 0
        self.is_playing = False
        self.animation = None
        self.speed = 500

        self.setup_controls()
        self.draw_graph()

    def setup_controls(self):
        # Control buttons
        ax_play = plt.axes([0.15, 0.05, 0.1, 0.04])
        ax_pause = plt.axes([0.26, 0.05, 0.1, 0.04])
        ax_step = plt.axes([0.37, 0.05, 0.1, 0.04])
        ax_restart = plt.axes([0.48, 0.05, 0.1, 0.04])
        ax_speed = plt.axes([0.15, 0.12, 0.3, 0.02])

        self.btn_play = Button(ax_play, 'Play')
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_step = Button(ax_step, 'Step')
        self.btn_restart = Button(ax_restart, 'Restart')
        self.slider_speed = Slider(ax_speed, 'Speed', 50, 2000, valinit=500)

        self.btn_play.on_clicked(self.play)
        self.btn_pause.on_clicked(self.pause)
        self.btn_step.on_clicked(self.step)
        self.btn_restart.on_clicked(self.restart)
        self.slider_speed.on_changed(self.set_speed)

    def draw_graph(self):
        self.ax.clear()

        # Get positions - use stored or generate
        pos = {}
        for node in self.G.nodes():
            if 'pos' in self.G.nodes[node]:
                pos[node] = self.G.nodes[node]['pos']
            else:
                # Generate fallback position if missing
                pos[node] = (0, 0)

        # If no positions available, generate layout
        if not pos or all(p == (0, 0) for p in pos.values()):
            pos = nx.spring_layout(self.G, seed=42)

        # Draw edges
        nx.draw_networkx_edges(self.G, pos, ax=self.ax, alpha=0.3)

        # Draw nodes with default colors
        node_colors = ['lightgray'] * len(self.G.nodes())
        nx.draw_networkx_nodes(self.G, pos, ax=self.ax, node_color=node_colors, node_size=300)

        # Draw labels
        nx.draw_networkx_labels(self.G, pos, ax=self.ax, font_size=8)

        # Store positions for animation
        self.pos = pos
        self.node_colors = {node: 'lightgray' for node in self.G.nodes()}

        self.update_display()
        self.ax.legend(handles=self.legend_patches, loc='upper right')

    def update_display(self):
        self.ax.clear()

        # Draw edges
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, alpha=0.3)

        # Draw nodes with current colors
        node_colors_list = [self.node_colors[node] for node in self.G.nodes()]
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax,
                               node_color=node_colors_list, node_size=300)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax, font_size=8)

        # Draw edge weights if available and graph is small
        if self.G.number_of_edges() < 50:
            edge_labels = {(u, v): f"{d.get('weight', 1):.1f}"
                           for u, v, d in self.G.edges(data=True)}
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels, ax=self.ax, font_size=6)

        self.ax.set_title(f"Step {self.current_step}/{len(self.search_states)}")
        self.ax.legend(handles=self.legend_patches, loc='upper right')

    def load_algorithm(self, algorithm_name, start, goal, path):
        """Simple visualization of the found path"""
        self.current_algorithm = algorithm_name
        self.start = start
        self.goal = goal
        self.path = path

        # Reset colors
        self.node_colors = {node: 'lightgray' for node in self.G.nodes()}
        self.node_colors[start] = self.colors['start']
        self.node_colors[goal] = self.colors['goal']

        # Highlight path
        if path:
            for node in path:
                if node not in [start, goal]:
                    self.node_colors[node] = self.colors['path']

        self.update_display()

    def play(self, event=None):
        self.is_playing = True
        self.animate()

    def pause(self, event=None):
        self.is_playing = False
        if self.animation:
            self.animation.event_source.stop()

    def step(self, event=None):
        if self.current_step < len(self.search_states) - 1:
            self.current_step += 1
            # self.update_state(self.search_states[self.current_step])

    def restart(self, event=None):
        self.current_step = 0
        if self.search_states:
            # self.update_state(self.search_states[0])
            pass
        self.pause()

    def set_speed(self, val):
        self.speed = val
        if self.animation:
            self.animation.event_source.interval = val

    def animate(self):
        def update(frame):
            if self.is_playing and self.current_step < len(self.search_states) - 1:
                self.current_step += 1
                # self.update_state(self.search_states[self.current_step])
            else:
                self.pause()

        self.animation = FuncAnimation(self.fig, update, interval=self.speed, cache_frame_data=False)

    def show(self):
        plt.show()


def draw_path(G: nx.Graph, path, title="Best Path"):
    """Simple path visualization function"""
    # Get positions - handle missing positions
    pos = {}
    for node in G.nodes():
        if 'pos' in G.nodes[node]:
            pos[node] = G.nodes[node]['pos']

    # If no positions available, generate layout
    if not pos:
        pos = nx.spring_layout(G, seed=42)
        print("Generated fallback layout for visualization")

    plt.figure(figsize=(12, 8))

    # Draw base graph
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.5, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=6)

    # Highlight path if exists
    if path:
        # Highlight path nodes
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=120, node_color='red')

        # Highlight path edges
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color='red')

        # Highlight start and goal
        if path:
            nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_size=200, node_color='green')
            nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_size=200, node_color='orange')

        plt.title(f"{title} (Path length: {len(path)}, Cost: {path_cost(G, path):.2f})")
    else:
        plt.title(title + " (no solution)")

    plt.tight_layout()
    plt.show()


def path_cost(G: nx.Graph, path: list) -> float:
    """Calculate total cost of a path"""
    if not path or len(path) < 2:
        return 0.0
    total_cost = 0.0
    for i in range(len(path) - 1):
        if G.has_edge(path[i], path[i + 1]):
            total_cost += G[path[i]][path[i + 1]].get('weight', 1.0)
    return total_cost
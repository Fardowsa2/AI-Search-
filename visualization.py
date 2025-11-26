# src/visualization.py 
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
import time
from collections import deque


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
        self.search_states = deque()
        self.current_step = 0
        self.is_playing = False
        self.animation = None
        self.speed = 500  # ms between steps

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

        # Use stored positions or spring layout
        if all("pos" in self.G.nodes[n] for n in self.G.nodes):
            pos = {n: self.G.nodes[n]["pos"] for n in self.G.nodes}
        else:
            pos = nx.spring_layout(self.G, seed=42)

        # Draw nodes and edges
        nx.draw_networkx_edges(self.G, pos, ax=self.ax, alpha=0.3)
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

        # Draw edge weights if available
        edge_labels = {(u, v): f"{d.get('weight', 1):.1f}"
                       for u, v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels, ax=self.ax)

        self.ax.set_title(f"Step {self.current_step}/{len(self.search_states)}")
        self.ax.legend(handles=self.legend_patches, loc='upper right')

    def load_algorithm(self, algorithm_name, start, goal):
        self.current_algorithm = algorithm_name
        self.start = start
        self.goal = goal

        # Reset colors
        self.node_colors = {node: 'lightgray' for node in self.G.nodes()}
        self.node_colors[start] = self.colors['start']
        self.node_colors[goal] = self.colors['goal']

        # Generate search states for visualization
        self.search_states = self.generate_search_states(algorithm_name, start, goal)
        self.current_step = 0

        self.update_display()

    def generate_search_states(self, algorithm_name, start, goal):
        # This would capture the state at each step of the search
        # For now, return mock states - you'd need to modify algorithms to yield states
        states = []

        # Add initial state
        states.append({
            'current': start,
            'frontier': [start],
            'explored': set(),
            'path': []
        })

        # This would be populated by instrumented search algorithms
        # You'd need to modify your algorithms to yield state at each step
        return deque(states)

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
            self.update_state(self.search_states[self.current_step])

    def restart(self, event=None):
        self.current_step = 0
        if self.search_states:
            self.update_state(self.search_states[0])
        self.pause()

    def set_speed(self, val):
        self.speed = val
        if self.animation:
            self.animation.event_source.interval = val

    def update_state(self, state):
        # Update node colors based on current state
        for node in self.G.nodes():
            if node == state.get('current'):
                self.node_colors[node] = self.colors['current']
            elif node in state.get('frontier', []):
                self.node_colors[node] = self.colors['frontier']
            elif node in state.get('explored', set()):
                self.node_colors[node] = self.colors['explored']
            elif node in state.get('path', []):
                self.node_colors[node] = self.colors['path']
            elif node in [self.start, self.goal]:
                self.node_colors[node] = self.colors['start'] if node == self.start else self.colors['goal']
            else:
                self.node_colors[node] = self.colors['normal']

        self.update_display()

    def animate(self):
        def update(frame):
            if self.is_playing and self.current_step < len(self.search_states) - 1:
                self.current_step += 1
                self.update_state(self.search_states[self.current_step])
            else:
                self.pause()

        self.animation = FuncAnimation(self.fig, update, interval=self.speed, cache_frame_data=False)

    def show(self):
        plt.show()

# src/graph_loader.py - UPDATED
from pathlib import Path
import math
import re
import networkx as nx
import pandas as pd


def _find_col(columns, keywords):
    """
    Return the first column whose lowercase name contains any of the keywords.
    """
    cl = [c.strip() for c in columns]
    for c in cl:
        name = c.lower().strip()
        for kw in keywords:
            if kw in name:
                return c
    return None


def _infer_name_lat_lon(df: pd.DataFrame):
    """
    Try hard to infer the 'name', 'lat', and 'lon' columns from the CSV.
    """
    cols = list(df.columns)

    # name-like columns
    name_col = _find_col(cols, ["name", "city", "town", "place", "label"])
    # latitude-like columns
    lat_col = _find_col(cols, ["lat", "latitude", "y"])
    # longitude-like columns
    lon_col = _find_col(cols, ["lon", "long", "longitude", "lng", "x"])

    # If lat/lon still missing, try heuristics:
    if lat_col is None or lon_col is None:
        # Look for numeric candidates
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        # If we have at least 2 numeric columns, assume the first two are coordinates
        if len(numeric_cols) >= 2:
            # Keep name_col if we already have it; otherwise, assume the first non-numeric is the name
            if name_col is None:
                non_numeric = [c for c in cols if c not in numeric_cols]
                name_col = non_numeric[0] if non_numeric else cols[0]
            # Pick the first two numeric columns for lat/lon
            lat_col, lon_col = numeric_cols[0], numeric_cols[1]

    # Final fallbacks if name is still None
    if name_col is None:
        name_col = cols[0]

    return name_col, lat_col, lon_col


def load_set1_graph(adj_path: str, coord_path: str) -> nx.Graph:
    """
    Build an undirected weighted graph from Adjacencies.txt and coordinates.csv.
    Handles missing positions by generating fallback positions.
    """
    G = nx.Graph()

    # Dictionary to store node positions
    positions = {}

    # Load coordinates if file exists
    if Path(coord_path).exists():
        try:
            coords_df = pd.read_csv(coord_path)
            name_col, lat_col, lon_col = _infer_name_lat_lon(coords_df)

            # Add nodes with positions
            for _, row in coords_df.iterrows():
                name = str(row[name_col]).strip()
                if not name or name.lower() == "nan":
                    continue

                try:
                    lon = float(row[lon_col])
                    lat = float(row[lat_col])
                    positions[name] = (lon, lat)
                    G.add_node(name, pos=(lon, lat))
                except (ValueError, TypeError):
                    # If coords are invalid, still add the node without position
                    G.add_node(name)
                    print(f"Warning: Invalid coordinates for {name}")
        except Exception as e:
            print(f"Warning: Could not load coordinates: {e}")
    else:
        print(f"Warning: Coordinates file {coord_path} not found")

    # Load adjacencies
    if Path(adj_path).exists():
        with open(adj_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = re.split(r"[,\s]+", s)
                if len(parts) < 2:
                    continue
                a, b = parts[0].strip(), parts[1].strip()

                # Add nodes if they don't exist
                if a not in G:
                    G.add_node(a)
                if b not in G:
                    G.add_node(b)

                # Calculate weight based on available positions
                pos_a = positions.get(a)
                pos_b = positions.get(b)
                if pos_a and pos_b:
                    dx = pos_a[0] - pos_b[0]
                    dy = pos_a[1] - pos_b[1]
                    w = math.hypot(dx, dy)
                else:
                    w = 1.0  # Default weight

                if not G.has_edge(a, b):
                    G.add_edge(a, b, weight=w)
    else:
        print(f"Error: Adjacency file {adj_path} not found")
        return G

    # Add missing positions for nodes without coordinates
    missing_positions = [node for node in G.nodes() if 'pos' not in G.nodes[node]]
    if missing_positions:
        print(f"Warning: {len(missing_positions)} nodes missing positions, generating fallback layout...")
        # Generate positions for all nodes using spring layout
        all_positions = nx.spring_layout(G, seed=42)
        for node in G.nodes():
            G.nodes[node]['pos'] = all_positions[node]

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G
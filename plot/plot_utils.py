import networkx as nx

def netron_style_layout_balanced(G, vertical_spacing=100, horizontal_spacing=150):
    """
    Netron-style layout for DAGs with balanced left-right branch placement.

    Args:
        G: A NetworkX directed acyclic graph (DAG).
        vertical_spacing: Vertical distance between layers (y-axis).
        horizontal_spacing: Horizontal distance between branches (x-axis).

    Returns:
        pos: A dict mapping each node to its (x, y) plot position.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG")

    # Step 1: Find the longest path ("spine") using dynamic programming
    top_order = list(nx.topological_sort(G))
    dist = {node: 0 for node in top_order}
    pred = {node: None for node in top_order}

    for node in top_order:
        for succ in G.successors(node):
            if dist[node] + 1 > dist[succ]:
                dist[succ] = dist[node] + 1
                pred[succ] = node

    end_node = max(dist, key=dist.get)
    spine = []
    n = end_node
    while n is not None:
        spine.append(n)
        n = pred[n]
    spine.reverse()

    # Step 2: Assign depth based on topological layer
    node_depth = {}
    for node in top_order:
        preds = list(G.predecessors(node))
        if not preds:
            node_depth[node] = 0
        else:
            node_depth[node] = max(node_depth[p] + 1 for p in preds)

    # Step 3: Place spine nodes vertically at x=0
    pos = {}
    for node in spine:
        y = -node_depth[node] * vertical_spacing
        pos[node] = (0, y)

    # Step 4: Place side branches to left and right
    occupied_slots = set()  # (depth, x) tuples to avoid overlaps

    # Pre-fill occupied positions for spine nodes
    for node in spine:
        x, y = pos[node]
        depth = node_depth[node]
        occupied_slots.add((depth, x))

    for node in top_order:
        if node in pos:
            continue  # already placed on the spine

        depth = node_depth[node]
        y = -depth * vertical_spacing

        # Heuristic: determine placement side based on predecessors
        preds = list(G.predecessors(node))
        pred_xs = [pos[p][0] for p in preds if p in pos]

        side = None
        if pred_xs:
            avg_x = sum(pred_xs) / len(pred_xs)
            if avg_x > horizontal_spacing / 2:
                side = "right"
            elif avg_x < -horizontal_spacing / 2:
                side = "left"

        # Try to place node on preferred side, or alternate outward from spine
        max_attempts = 10
        for i in range(1, max_attempts):
            if side == "left":
                x_try = -i * horizontal_spacing
            elif side == "right":
                x_try = i * horizontal_spacing
            else:
                # If side unclear, try left then right outward from center
                x_try = (-1)**i * ((i + 1) // 2) * horizontal_spacing

            if (depth, x_try) not in occupied_slots:
                pos[node] = (x_try, y)
                occupied_slots.add((depth, x_try))
                break
        else:
            # Fallback if all slots are taken: place far right
            fallback_x = (max_attempts + 1) * horizontal_spacing
            pos[node] = (fallback_x, y)
            occupied_slots.add((depth, fallback_x))

    return pos

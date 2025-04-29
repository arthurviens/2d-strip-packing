import networkx as nx

def netron_style_layout_with_spine(G, vertical_spacing=100, horizontal_spacing=150):
    """
    Netron-style layout for DAGs: center the longest path (spine), place others around.

    Args:
        G: A NetworkX DAG.
        vertical_spacing: Y-distance between layers.
        horizontal_spacing: X-distance between sibling branches.

    Returns:
        pos: Dict mapping node -> (x, y)
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG")

    # Step 1: Longest path via dynamic programming over topological order
    top_order = list(nx.topological_sort(G))
    dist = {node: 0 for node in top_order}
    pred = {node: None for node in top_order}

    for node in top_order:
        for succ in G.successors(node):
            if dist[node] + 1 > dist[succ]:
                dist[succ] = dist[node] + 1
                pred[succ] = node

    # Find end of longest path
    end_node = max(dist, key=lambda n: dist[n])
    # Reconstruct path backwards
    spine = []
    n = end_node
    while n is not None:
        spine.append(n)
        n = pred[n]
    spine.reverse()

    # Step 2: Assign vertical layer (depth)
    node_depth = {}
    for node in top_order:
        preds = list(G.predecessors(node))
        if not preds:
            node_depth[node] = 0
        else:
            node_depth[node] = max(node_depth[p] + 1 for p in preds)

    # Step 3: Place spine nodes on x = 0
    pos = {}
    layer_to_spine_x = {}

    for node in spine:
        y = -node_depth[node] * vertical_spacing
        pos[node] = (0, y)
        layer_to_spine_x[node_depth[node]] = 0

    # Step 4: Place side nodes to left/right of spine, respecting predecessor side if any
    side_counters = {}  # depth -> alternation counter

    for node in top_order:
        if node in pos:
            continue  # already placed (on spine)

        depth = node_depth[node]
        preds = list(G.predecessors(node))
        pred_xs = [pos[p][0] for p in preds if p in pos]

        # Try to respect the side of already-placed predecessors
        if pred_xs:
            avg_x = sum(pred_xs) / len(pred_xs)
            if avg_x > horizontal_spacing / 2:
                side = "right"
            elif avg_x < -horizontal_spacing / 2:
                side = "left"
            else:
                side = None
        else:
            side = None

        # Decide placement
        counter = side_counters.get(depth, 0)
        if side == "left":
            direction = -1
            index = sum(1 for x in pred_xs if x < 0) + 1
        elif side == "right":
            direction = 1
            index = sum(1 for x in pred_xs if x > 0) + 1
        else:
            direction = -1 if counter % 2 == 0 else 1  # alternate
            index = counter // 2 + 1

        x = direction * index * horizontal_spacing
        y = -depth * vertical_spacing
        pos[node] = (x, y)

        side_counters[depth] = counter + 1


    return pos

"""Sparse connected grid and up to K shortest simple routes; standard library only."""
from __future__ import annotations

import heapq
import math
import random
from collections import deque

GRID_X, GRID_Y, STEP_M = 30, 40, 50
SURFACES = ("paved", "gravel", "dirt")


def neighbors(cell):
    x, y = cell
    for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        p = (x + dx, y + dy)
        if 0 <= p[0] < GRID_X and 0 <= p[1] < GRID_Y:
            yield p


def generate_graph(rng: random.Random, node_count=200):
    """Grow a connected subgraph, then add local cycles and two dead-end dumps."""
    if not (60 <= node_count <= GRID_X * GRID_Y):
        raise ValueError("node_count must be within [60, 1200]")
    center = (GRID_X // 2, GRID_Y // 2)
    chosen = {center}
    tree_edges = set()
    tip = center
    while len(chosen) < node_count:
        options = [p for p in neighbors(tip) if p not in chosen]
        if not options or rng.random() < 0.20:
            frontier = [p for p in sorted(chosen) if any(n not in chosen for n in neighbors(p))]
            tip = rng.choice(frontier)
            options = [p for p in neighbors(tip) if p not in chosen]
        new = rng.choice(options)
        chosen.add(new)
        tree_edges.add(tuple(sorted((tip, new))))
        tip = new

    physical = set(tree_edges)
    # A few cycles make real choices possible; roads remain sparse.
    for p in sorted(chosen):
        for q in neighbors(p):
            pair = tuple(sorted((p, q)))
            if q in chosen and pair not in physical and rng.random() < 0.37:
                physical.add(pair)

    # Smooth terrain. Scaling is global, so every neighbouring grade is <= 8%.
    phase1, phase2, phase3 = (rng.uniform(-math.pi, math.pi) for _ in range(3))
    def raw_height(p):
        x, y = p
        return (15 * math.sin(x / 9 + phase1) + 18 * math.cos(y / 11 + phase2)
                + 9 * math.sin((x + y) / 8 + phase3) + 0.3*x - 0.2*y)
    all_cells = [(x, y) for x in range(GRID_X) for y in range(GRID_Y)]
    vals = {p: raw_height(p) for p in all_cells}
    max_step = max(abs(vals[p] - vals[q]) for p in all_cells for q in neighbors(p))
    scale = min(1.0, (STEP_M * .077) / max_step)  # small rounding margin

    # Correlated covers: neighboring links share the same terrain patches.
    patch = [[rng.uniform(-1, 1) for _ in range(7)] for _ in range(6)]
    def cover(p, q):
        x, y = (p[0]+q[0])/2, (p[1]+q[1])/2
        fx, fy = x / 6.0, y / 7.0
        ax, ay = min(4, int(fx)), min(5, int(fy))
        u, v = max(0, min(1, fx-ax)), max(0, min(1, fy-ay))
        s = (patch[ax][ay]*(1-u)*(1-v) + patch[ax+1][ay]*u*(1-v)
             + patch[ax][ay+1]*(1-u)*v + patch[ax+1][ay+1]*u*v)
        return "paved" if s < -0.11 else "dirt" if s > 0.11 else "gravel"

    cells = sorted(chosen)
    id_of = {p: i for i, p in enumerate(cells)}
    nodes = [dict(node_id=f"n-{i:04d}", kind="junction", x_m=p[0]*STEP_M,
                  y_m=p[1]*STEP_M, z_m=round(vals[p]*scale, 4)) for i,p in enumerate(cells)]
    edges = []
    adj = {i: [] for i in range(len(nodes))}
    def add_directed(u, v, surface, length=STEP_M):
        eid = f"e-{len(edges):05d}"
        slope = (nodes[v]['z_m'] - nodes[u]['z_m']) / length
        edge = dict(edge_id=eid, from_node=nodes[u]['node_id'], to_node=nodes[v]['node_id'],
                    length_m=length, surface=surface, u=u, v=v, slope=slope,
                    physical=(min(u,v), max(u,v)))
        edges.append(edge)
        adj[u].append((v, len(edges)-1))
    for p, q in sorted(physical):
        u, v = id_of[p], id_of[q]
        surface = cover(p, q)
        add_directed(u, v, surface)
        add_directed(v, u, surface)

    # Choose 3 dispersed loading sites using shortest-path hop distances.
    def bfs(source):
        q = deque([source]); d = {source: 0}
        while q:
            u = q.popleft()
            for v,_ in adj[u]:
                if v not in d:
                    d[v] = d[u]+1
                    q.append(v)
        return d
    candidates = [i for i in adj if len(adj[i]) >= 3]
    if len(candidates) < 5:
        candidates = list(adj)
    first = rng.choice(candidates)
    chosen_loads = [first]
    for _ in range(2):
        ds = [bfs(p) for p in chosen_loads]
        pick = max((i for i in candidates if i not in chosen_loads),
                   key=lambda i: (min(d[i] for d in ds), sum(d[i] for d in ds), -i))
        chosen_loads.append(pick)
    for i in chosen_loads:
        nodes[i]['kind'] = 'load'

    # Dumps are distinct dead-end vertices; one physical two-way road each.
    dists_to_load = [bfs(i) for i in chosen_loads]
    dump_anchors = []
    original_node_ids = list(range(node_count))
    for k in range(2):
        occupied = {(n['x_m'], n['y_m']) for n in nodes}
        def has_free_neighbor(idx):
            x, y = nodes[idx]['x_m'], nodes[idx]['y_m']
            return any((x+dx*STEP_M, y+dy*STEP_M) not in occupied
                       for dx,dy in ((1,0),(0,1),(-1,0),(0,-1)))
        candidates = [i for i in original_node_ids
                      if i not in chosen_loads + dump_anchors and has_free_neighbor(i)]
        dprevious = bfs(dump_anchors[0]) if dump_anchors else None
        anchor = max(candidates, key=lambda i: (
            min(d[i] for d in dists_to_load) + (dprevious[i] if dprevious else 0), -i))
        dump_anchors.append(anchor)
        new_id = len(nodes)
        x0, y0 = nodes[anchor]['x_m'], nodes[anchor]['y_m']
        occupied = {(n['x_m'],n['y_m']) for n in nodes}
        possible = [(x0+dx*STEP_M,y0+dy*STEP_M) for dx,dy in ((1,0),(0,1),(-1,0),(0,-1))]
        location = next((p for p in possible if p not in occupied), None)
        if location is None:
            raise ValueError('No free location to attach dump vertex')
        # Same altitude keeps the dead-end slope valid and routing simple.
        nodes.append(dict(node_id=f"n-{new_id:04d}",kind="dump",x_m=location[0],
                          y_m=location[1],z_m=nodes[anchor]['z_m']))
        adj[new_id] = []
        add_directed(anchor, new_id, 'gravel')
        add_directed(new_id, anchor, 'gravel')
    for values in adj.values():
        values.sort()
    return nodes, edges, adj, chosen_loads, [len(nodes)-2, len(nodes)-1]


def shortest_path(adj, edges, start, dest, weight, forbidden_nodes=frozenset(), forbidden_edges=frozenset()):
    """Dijkstra: returns a tuple of edge indexes (or None)."""
    if start in forbidden_nodes or dest in forbidden_nodes:
        return None
    if start == dest:
        return ()
    heap = [(0., start)]
    dist = {start: 0.}
    prev = {}
    while heap:
        cost, u = heapq.heappop(heap)
        if cost > dist[u] + 1e-9:
            continue
        if u == dest:
            result = []
            while u != start:
                eidx = prev[u]
                result.append(eidx)
                u = edges[eidx]['u']
            return tuple(reversed(result))
        for v, eidx in adj[u]:
            if v in forbidden_nodes or eidx in forbidden_edges:
                continue
            new_cost = cost + weight(edges[eidx])
            if new_cost + 1e-9 < dist.get(v, float('inf')):
                dist[v] = new_cost
                prev[v] = eidx
                heapq.heappush(heap, (new_cost, v))
    return None


def k_shortest_paths(adj, edges, source, target, weight, k=5):
    """Yen's algorithm, exact K shortest *simple* paths for nonnegative weights."""
    first = shortest_path(adj, edges, source, target, weight)
    if first is None:
        return []
    selected = [first]
    candidates = []
    seen = {first}
    for _ in range(1, k):
        previous = selected[-1]
        vertices = [source] + [edges[e]['v'] for e in previous]
        for i in range(len(previous)):
            prefix = previous[:i]
            blocked_nodes = set(vertices[:i])
            blocked_edges = {route[i] for route in selected if len(route) > i and route[:i] == prefix}
            suffix = shortest_path(adj, edges, vertices[i], target, weight, blocked_nodes, blocked_edges)
            if suffix is not None:
                candidate = prefix + suffix
                if candidate not in seen:
                    cost = sum(weight(edges[e]) for e in candidate)
                    heapq.heappush(candidates, (cost, candidate))
                    seen.add(candidate)
        if not candidates:
            break
        selected.append(heapq.heappop(candidates)[1])
    return selected

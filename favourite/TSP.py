"""
Full implementation: Held-Karp (DP) with precedence support + robust fallback
when there are pinned (fixed-position) slots (uses corrected brute-force).

Usage notes:
- Vertices are 0..V-1 (indices into locations list in your view code).
- fixed_position: list of length V where fixed_position[pos] is either None or the vertex index
  that must be placed at slot `pos` (slot numbering 0..V-1).
- precedence_constraints: list of pairs (a, b) meaning "a must appear before b" where a,b
  are vertex indices in 0..V-1.
- start / end: either None or a vertex index (0..V-1). If provided, they will be respected.

Behavior:
- If any pinned (fixed_position contains not-None), algorithm uses a corrected permutation approach
  (safe, exact, but factorial in free slots count).
- If no pinned: uses Held-Karp DP (O(n^2 * 2^n)) and enforces precedence constraints correctly.

This module fixes earlier bugs: candidate construction, _calculate_path_cost index usage,
proper mapping of precedence when some nodes are excluded (start/end), and cycle checks.

"""

from itertools import permutations
from typing import List, Optional, Tuple, Dict
import math, requests


def _check_fixed_positions(perm: List[int], fixed_position: List[Optional[int]]) -> bool:
    """perm is full-length list of vertex indices placed in order.
    fixed_position[pos] is either None or a vertex index that must appear at pos.
    """
    if len(perm) != len(fixed_position):
        return False
    return all(fp is None or perm[pos] == fp for pos, fp in enumerate(fixed_position))


def _check_start_end(perm: List[int], start: Optional[int], end: Optional[int]) -> bool:
    if start is not None and perm[0] != start:
        return False
    if end is not None and perm[-1] != end:
        return False
    return True


def _check_precedence(perm: List[int], precedence_constraints: List[Tuple[int, int]]) -> bool:
    pos_map = {node: idx for idx, node in enumerate(perm)}
    try:
        return all(pos_map[before] < pos_map[after] for before, after in precedence_constraints)
    except KeyError:
        # If a node referenced in constraints is not present in perm => invalid
        return False


def _has_cycle(precedence_constraints: List[Tuple[int, int]], n_vertices: int) -> bool:
    # Simple cycle detection via Kahn's algorithm on vertices 0..n_vertices-1
    g = {i: [] for i in range(n_vertices)}
    indeg = [0] * n_vertices
    for a, b in precedence_constraints:
        if a < 0 or a >= n_vertices or b < 0 or b >= n_vertices:
            continue
        g[a].append(b)
        indeg[b] += 1

    q = [i for i in range(n_vertices) if indeg[i] == 0]
    seen = 0
    while q:
        v = q.pop()
        seen += 1
        for w in g[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    return seen != n_vertices


class Graph:
    def __init__(self, vertices: int):
        self.V = vertices
        self.adjacency_matrix = [[math.inf for _ in range(vertices)] for _ in range(vertices)]

    def add_edge(self, u: int, v: int, w: float):
        self.adjacency_matrix[u][v] = w

    def _calculate_path_cost(self, path: List[int]) -> float:
        """Sum cost along given path (list of vertex indices)."""
        if not path or len(path) < 2:
            return 0.0
        return sum(self.adjacency_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))

    # ----------------------------- Held-Karp w/ precedence (no pinned) -----------------------------
    def _held_karp_with_precedence(self,
                                   nodes: List[int],
                                   precedence: List[Tuple[int, int]],
                                   start_node: Optional[int],
                                   end_node: Optional[int]) -> Tuple[Optional[List[int]], Optional[float]]:
        """Solve TSP path visiting exactly `nodes` in any order (these are vertex indices).

        nodes: list of vertex indices to be visited (does NOT include start_node or end_node).
        precedence: pairs (a,b) where a must come before b; a,b are vertex indices in global graph.
        start_node / end_node: if provided are vertex indices (may or may not be in nodes).

        Returns (path_list, cost) where path_list includes start_node (if provided), nodes in order,
        then end_node (if provided).

        Approach:
        - We map nodes -> positions 0..n-1 for DP.
        - Precedence constraints that involve vertices outside the set {start_node} U nodes U {end_node}
          are considered invalid (raise or return None).
        - Enforce constraint for each 'last' chosen: all its predecessors must be already in the mask
          (or be the start_node already placed before sequence began).
        """
        n = len(nodes)
        if n == 0:
            # trivial: only possibly edge start->end
            if start_node is None and end_node is None:
                return [], 0.0
            if start_node is None:
                # path is just end_node
                return [end_node], 0.0
            if end_node is None:
                return [start_node], 0.0
            return [start_node, end_node], self.adjacency_matrix[start_node][end_node]

        # map global vertex -> local index for nodes
        global_to_local: Dict[int, int] = {v: i for i, v in enumerate(nodes)}

        # Preprocess precedence: produce local-index pairs (x,y) where x,y in 0..n-1
        # Cases:
        # - if a == start_node, treat as satisfied before any DP selection (start is before everything)
        # - if b == start_node -> impossible
        # - if a == end_node -> impossible (end must come after a but end is after whole sequence)
        # - if b == end_node: requires a appears somewhere in nodes or equals start
        prec_local: List[Tuple[int, int]] = []
        predecessors_of_local: Dict[int, List[int]] = {i: [] for i in range(n)}

        for a, b in precedence:
            # invalid references
            if (start_node is not None and b == start_node) or (end_node is not None and a == end_node):
                # b before start or end before a -> impossible constraint
                return None, None

            if a == start_node:
                # satisfied by construction (start before others). If b is in nodes or equals end_node, OK.
                if b in global_to_local:
                    # nothing to add: just ensure that when placing b, it's not in first position if start not provided
                    # but since start is placed before sequence, b can be anywhere in nodes.
                    pass
                else:
                    # b could be end_node, that's fine
                    pass
                continue

            if b == end_node:
                # constraint a -> end means a must be visited somewhere in nodes (or be start)
                # we'll check later that a is reachable.
                if a in global_to_local:
                    # interpret as local_a must be in mask at end (full mask)
                    local_a = global_to_local[a]
                    predecessors_of_local.setdefault(-1, []).append(local_a)  # -1 marks 'end' requirement
                else:
                    # a might be start_node; if not, impossible
                    if a != start_node:
                        return None, None
                continue

            if a in global_to_local and b in global_to_local:
                la = global_to_local[a]
                lb = global_to_local[b]
                predecessors_of_local[lb].append(la)
                prec_local.append((la, lb))
            else:
                # constraint involves vertices not in nodes and not start/end -> impossible
                # (This is a strict stance; alternatively you could allow constraints referencing vertices outside selection)
                return None, None

        # DP table: dp[mask][last] = (cost, prev)
        # mask ranges 1..(1<<n)-1
        dp: Dict[Tuple[int, int], Tuple[float, Optional[int]]] = {}

        # initialize base cases: choose any single node as first in sequence
        for i in range(n):
            # if node i has predecessors that are not start (i.e. must be present before i), then i cannot be first
            preds = predecessors_of_local.get(i, [])
            if preds:
                # if any pred requires another node to be before i, we cannot place i as first
                continue
            # cost from start_node (if provided) else 0
            cost_from_start = 0.0
            if start_node is not None:
                cost_from_start = self.adjacency_matrix[start_node][nodes[i]]
            dp[(1 << i, i)] = (cost_from_start, None)

        # iterate masks
        full_mask = (1 << n) - 1
        for mask in range(1, 1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                key = (mask, last)
                if key not in dp:
                    # might be uninitialized because impossible to reach due to precedence
                    continue
                cost_here, _ = dp[key]
                # try to extend by next node 'nxt'
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    # nxt can be added only if all predecessors of nxt are in (mask)
                    preds = predecessors_of_local.get(nxt, [])
                    if any(not (mask & (1 << p)) for p in preds):
                        continue
                    new_mask = mask | (1 << nxt)
                    new_cost = cost_here + self.adjacency_matrix[nodes[last]][nodes[nxt]]
                    nk = (new_mask, nxt)
                    if nk not in dp or new_cost < dp[nk][0]:
                        dp[nk] = (new_cost, last)

        # After filling, find best last to connect to end_node (if provided)
        best_cost = math.inf
        best_last = None
        for last in range(n):
            key = (full_mask, last)
            if key not in dp:
                continue
            cost_here = dp[key][0]
            # check any predecessor requirement that involves 'end' (stored under -1)
            end_preds = predecessors_of_local.get(-1, [])
            if any(not (full_mask & (1 << p)) for p in end_preds):
                # required predecessor for end not satisfied
                continue
            # add edge to end_node if present else finish at nodes[last]
            cost_total = cost_here
            if end_node is not None:
                cost_total += self.adjacency_matrix[nodes[last]][end_node]
            if cost_total < best_cost:
                best_cost = cost_total
                best_last = last

        if best_last is None:
            return None, None

        # reconstruct path
        path_local = []
        mask = full_mask
        cur = best_last
        while cur is not None:
            path_local.append(nodes[cur])
            prev = dp[(mask, cur)][1]
            mask ^= (1 << cur)
            cur = prev
        path_local.reverse()

        # add start / end
        full_path = []
        if start_node is not None:
            full_path.append(start_node)
        full_path.extend(path_local)
        if end_node is not None:
            full_path.append(end_node)

        return full_path, best_cost

    # ----------------------------- Permutation fallback (pinned) -----------------------------------
    def _permutation_search(self, fixed_position: List[Optional[int]],
                            precedence_constraints: List[Tuple[int, int]],
                            start: Optional[int], end: Optional[int]) -> Tuple[Optional[List[int]], Optional[float]]:
        n = self.V
        # fixed_position: length n, values either None or vertex index
        # derive free slots and free nodes
        free_slots = [i for i, v in enumerate(fixed_position) if v is None]
        fixed_nodes = [v for v in fixed_position if v is not None]
        free_nodes = [i for i in range(n) if i not in fixed_nodes]

        # quick sanity: ensure start/end if provided are compatible with fixed slots
        if start is not None:
            # if start is pinned to some other vertex -> impossible
            if start in fixed_nodes and fixed_position.index(start) != 0:
                return None, None
        if end is not None:
            if end in fixed_nodes and fixed_position.index(end) != n - 1:
                return None, None

        min_cost = math.inf
        min_path = None

        # iterate permutations of free_nodes placed into free_slots
        for perm in permutations(free_nodes):
            candidate = [None] * n
            # fill fixed
            for idx, node in enumerate(fixed_position):
                if node is not None:
                    candidate[idx] = node
            # fill free slots
            for slot_idx, node in zip(free_slots, perm):
                candidate[slot_idx] = node

            # validations
            if not _check_start_end(candidate, start, end):
                continue
            if not _check_precedence(candidate, precedence_constraints):
                continue
            # all nodes present exactly once?
            if len(set(candidate)) != n or any(x is None for x in candidate):
                continue

            cost = self._calculate_path_cost(candidate)
            if cost < min_cost:
                min_cost = cost
                min_path = candidate.copy()

        if min_path is None:
            return None, None
        return min_path, min_cost

    # ----------------------------- Public -----------------------------------
    def find_hamiltonian_path(self,
                              fixed_position: Optional[List[Optional[int]]] = None,
                              precedence_constraints: Optional[List[Tuple[int, int]]] = None,
                              start: Optional[int] = None,
                              end: Optional[int] = None) -> Tuple[Optional[List[int]], Optional[float]]:
        """Main entry. fixed_position length must equal V if provided. precedence list uses global vertex indices."""
        fixed_position = fixed_position or [None] * self.V
        precedence_constraints = precedence_constraints or []

        if len(fixed_position) != self.V:
            raise ValueError("fixed_position must have length equal to graph vertices")

        # quick cycle detection on full vertex set
        if _has_cycle(precedence_constraints, self.V):
            return None, None

        # If any fixed slot is present, use permutation fallback (safe)
        if any(x is not None for x in fixed_position):
            return self._permutation_search(fixed_position, precedence_constraints, start, end)

        # else use Held-Karp on nodes = all vertices except optional start/end
        nodes = [i for i in range(self.V) if i not in {start, end}]
        return self._held_karp_with_precedence(nodes, precedence_constraints, start, end)

def distance(origins, destinations):
    api_key = "w6YHI8MoaxcNVp4wPBP7WbanJ5KI6EEx3QtIFtV1Y0VBOWLMc85ZL399y9FUVqHk"
    url = "https://api-v2.distancematrix.ai/maps/api/distancematrix/json"

    params = {
        "origins": origins,
        "destinations": destinations,
        "key": api_key
    }

    response = requests.get(url, params=params)

    result = response.json()
    distance = result["rows"][0]["elements"][0]["distance"]["value"]
    duration = result["rows"][0]["elements"][0]["duration"]["value"]
    return distance, duration

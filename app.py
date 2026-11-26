"""
Project 4: Social Network Connection and Recommendation Analysis
Logic layer: graph + BFS + DFS + shortest path + mutuals + recommendations + benchmark.
Run:  pip install flask   ->   python app.py   ->   open http://127.0.0.1:5000
Keep app.py and index.html in the same folder.
"""
import os
import random
import time
from collections import deque

from flask import Flask, jsonify, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

NAMES = ["Aarav", "Ananya", "Arjun", "Diya", "Ishaan", "Kavya", "Rohan", "Meera", "Vivaan",
         "Saanvi", "Aditya", "Riya", "Kabir", "Navya", "Yash", "Tara", "Dev", "Anika", "Neil",
         "Prisha", "Rahul", "Sneha", "Karan", "Pooja", "Varun", "Isha", "Manav", "Simran",
         "Tanvi", "Harsh", "Nisha", "Dhruv", "Lakshya", "Mitali", "Parth", "Radhika", "Sahil",
         "Trisha", "Uday", "Zoya", "Ayaan", "Bhavya", "Chirag", "Divya", "Farhan"]


# --------------------------------------------------------------------------- #
#  Graph: users = vertices, connections = undirected edges (adjacency list)
# --------------------------------------------------------------------------- #
class Graph:
    def __init__(self):
        self.adj = {}

    def add_user(self, u):
        self.adj.setdefault(u, set())

    def connect(self, a, b):
        if a == b:
            return
        self.add_user(a)
        self.add_user(b)
        self.adj[a].add(b)
        self.adj[b].add(a)

    def edges(self):
        return sorted({tuple(sorted((a, b))) for a, nbrs in self.adj.items() for b in nbrs}, key=str)

    # ---- BFS: level by level, uses a queue -> gives shortest hops ---------- #
    def bfs(self, start):
        order, parent, depth = [], {start: None}, {start: 0}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in sorted(self.adj[u], key=str):
                if v not in parent:
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    queue.append(v)
        return order, parent, depth

    # ---- DFS: goes deep first, uses a stack (iterative = no recursion limit) #
    def dfs(self, start):
        order, parent, depth, seen = [], {}, {}, set()
        stack = [(start, None, 0)]
        while stack:
            u, p, d = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            parent[u], depth[u] = p, d
            order.append(u)
            for v in sorted(self.adj[u], key=str, reverse=True):  # reverse -> alphabetical visit
                if v not in seen:
                    stack.append((v, u, d + 1))
        return order, parent, depth

    def traverse(self, algo, start):
        return self.bfs(start) if algo == "bfs" else self.dfs(start)

    # ---- Path between two users (BFS = shortest, DFS = just "a" path) ------ #
    def find_path(self, algo, src, dst):
        t0 = time.perf_counter()
        order, parent, _ = self.traverse(algo, src)
        micros = round((time.perf_counter() - t0) * 1e6, 1)
        if dst not in parent:
            return {"path": [], "hops": None, "explored": len(order), "time_us": micros}
        path, node = [], dst
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return {"path": path, "hops": len(path) - 1,
                "explored": order.index(dst) + 1, "time_us": micros}

    # ---- Mutual connections = set intersection of both friend lists -------- #
    def mutual(self, a, b):
        return sorted(self.adj[a] & self.adj[b], key=str)

    # ---- Recommendations: friends-of-friends ranked by mutual count -------- #
    def recommend(self, user, k=5):
        scores = {}
        for friend in self.adj[user]:
            for fof in self.adj[friend]:
                if fof != user and fof not in self.adj[user]:
                    scores.setdefault(fof, set()).add(friend)
        ranked = sorted(scores.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))[:k]
        return [{"user": u, "mutual_count": len(m), "mutual": sorted(m, key=str)} for u, m in ranked]

    def components(self):
        seen, count = set(), 0
        for u in self.adj:
            if u not in seen:
                seen.update(self.bfs(u)[0])
                count += 1
        return count


def make_graph(labels, m=2, seed=1):
    """Preferential-attachment network: popular users attract more friends (like real life)."""
    rng, g, ends = random.Random(seed), Graph(), []
    for i, u in enumerate(labels):
        g.add_user(u)
        if i == 0:
            continue
        targets = set()
        while len(targets) < min(m, i):
            targets.add(rng.choice(ends) if ends and rng.random() < 0.8 else labels[rng.randrange(i)])
        for t in targets:
            g.connect(u, t)
            ends += [u, t]
    return g


def new_network(n):
    labels = NAMES[:n] if n <= len(NAMES) else NAMES + [f"User{i}" for i in range(n - len(NAMES))]
    return make_graph(labels, seed=random.randrange(10_000))


GRAPH = new_network(30)


# --------------------------------------------------------------------------- #
#  API
# --------------------------------------------------------------------------- #
def need(*names):
    for n in names:
        if n not in GRAPH.adj:
            raise ValueError(f"Unknown user: {n}")


@app.errorhandler(ValueError)
def bad_request(e):
    return jsonify(error=str(e)), 400


@app.after_request
def allow_origin(resp):  # lets index.html work even if it was opened straight from the folder
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/")
def home():
    return send_from_directory(BASE, "index.html")


@app.route("/api/graph")
def graph():
    nodes = [{"id": u, "degree": len(n)} for u, n in GRAPH.adj.items()]
    edges = GRAPH.edges()
    return jsonify(nodes=nodes, edges=edges, users=len(nodes), connections=len(edges),
                   components=GRAPH.components())


@app.route("/api/graph/new", methods=["POST"])
def regenerate():
    global GRAPH
    n = int((request.json or {}).get("n", 30))
    GRAPH = new_network(max(5, min(n, 60)))
    return jsonify(ok=True)


@app.route("/api/connect", methods=["POST"])
def connect():
    d = request.json or {}
    need(d.get("a"), d.get("b"))
    GRAPH.connect(d["a"], d["b"])
    return jsonify(ok=True)


@app.route("/api/traverse")
def traverse():
    algo, start = request.args.get("algo", "bfs"), request.args.get("start")
    need(start)
    t0 = time.perf_counter()
    order, parent, depth = GRAPH.traverse(algo, start)
    micros = round((time.perf_counter() - t0) * 1e6, 1)
    return jsonify(algo=algo, order=order, parent=parent, visited=len(order),
                   max_depth=max(depth.values()), time_us=micros)


@app.route("/api/path")
def path():
    src, dst = request.args.get("src"), request.args.get("dst")
    need(src, dst)
    return jsonify(bfs=GRAPH.find_path("bfs", src, dst), dfs=GRAPH.find_path("dfs", src, dst))


@app.route("/api/mutual")
def mutual():
    a, b = request.args.get("a"), request.args.get("b")
    need(a, b)
    return jsonify(mutual=GRAPH.mutual(a, b), a_friends=len(GRAPH.adj[a]), b_friends=len(GRAPH.adj[b]))


@app.route("/api/recommend")
def recommend():
    user = request.args.get("user")
    need(user)
    return jsonify(suggestions=GRAPH.recommend(user), friends=len(GRAPH.adj[user]))


@app.route("/api/benchmark")
def benchmark():
    """Time BFS vs DFS on networks of growing size (best of 5 runs each)."""
    rows = []
    for n in [100, 500, 1000, 5000, 10000, 20000]:
        g = make_graph(list(range(n)), seed=n)
        start, times = 0, {"bfs": [], "dfs": []}
        for algo in times:
            for _ in range(5):
                t0 = time.perf_counter()
                visited = len(g.traverse(algo, start)[0])
                times[algo].append((time.perf_counter() - t0) * 1000)
        rows.append({"nodes": n, "edges": len(g.edges()), "visited": visited,
                     "bfs_ms": round(min(times["bfs"]), 3), "dfs_ms": round(min(times["dfs"]), 3)})
    return jsonify(rows=rows)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

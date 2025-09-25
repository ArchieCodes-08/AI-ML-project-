import heapq, random, time, csv, math
import matplotlib.pyplot as plt
from collections import deque

# ==============================
# ENVIRONMENT
# ==============================
class GridWorld:
    def __init__(self, grid, start, goal, dyn_schedule=None):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.dyn_schedule = dyn_schedule or {}  # {t: [(x,y), ...]}

    def is_valid(self, x, y, t=0):
        if not (0 <= x < self.rows and 0 <= y < self.cols):
            return False
        if self.grid[x][y] == -1:  # static obstacle
            return False
        if t in self.dyn_schedule and (x, y) in self.dyn_schedule[t]:
            return False
        return True

    def cost(self, x, y):
        return self.grid[x][y] if self.grid[x][y] > 0 else 1

    def neighbors(self, node, t=0):
        x, y = node
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if self.is_valid(nx, ny, t):
                yield (nx, ny)

# ==============================
# PATH RECONSTRUCTION
# ==============================
def reconstruct(parent, goal):
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur, None)
    return list(reversed(path))

def path_cost(world, path):
    if not path: return float('inf')
    return sum(world.cost(x,y) for (x,y) in path)

# ==============================
# SEARCH ALGORITHMS
# ==============================
def bfs(world):
    q = deque([world.start])
    parent = {world.start: None}
    while q:
        node = q.popleft()
        if node == world.goal:
            break
        for neigh in world.neighbors(node):
            if neigh not in parent:
                parent[neigh] = node
                q.append(neigh)
    return reconstruct(parent, world.goal), len(parent)

def uniform_cost(world):
    pq = [(0, world.start)]
    parent = {world.start: None}
    g = {world.start: 0}
    expanded = 0
    while pq:
        cost, node = heapq.heappop(pq)
        expanded += 1
        if node == world.goal:
            break
        for neigh in world.neighbors(node):
            new_cost = g[node] + world.cost(*neigh)
            if neigh not in g or new_cost < g[neigh]:
                g[neigh] = new_cost
                parent[neigh] = node
                heapq.heappush(pq, (new_cost, neigh))
    return reconstruct(parent, world.goal), expanded

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(world):
    pq = [(0, world.start)]
    parent = {world.start: None}
    g = {world.start: 0}
    expanded = 0
    while pq:
        f, node = heapq.heappop(pq)
        expanded += 1
        if node == world.goal:
            break
        for neigh in world.neighbors(node):
            new_g = g[node] + world.cost(*neigh)
            if neigh not in g or new_g < g[neigh]:
                g[neigh] = new_g
                f = new_g + heuristic(neigh, world.goal)
                parent[neigh] = node
                heapq.heappush(pq, (f, neigh))
    return reconstruct(parent, world.goal), expanded

# ==============================
# LOCAL SEARCH (Hill Climb, Simulated Annealing)
# ==============================
def hill_climb(world, max_iters=500):
    cur = world.start
    path = [cur]
    expanded = 0
    for _ in range(max_iters):
        if cur == world.goal: break
        neighbors = list(world.neighbors(cur))
        if not neighbors: break
        expanded += 1
        cur = min(neighbors, key=lambda n: heuristic(n, world.goal))
        path.append(cur)
    return path if cur == world.goal else None, expanded

def simulated_annealing(world, max_iters=1000, T=10.0, alpha=0.99):
    cur = world.start
    path = [cur]
    expanded = 0
    for _ in range(max_iters):
        if cur == world.goal: break
        neighbors = list(world.neighbors(cur))
        if not neighbors: break
        expanded += 1
        nxt = random.choice(neighbors)
        dE = heuristic(cur, world.goal) - heuristic(nxt, world.goal)
        if dE > 0 or random.random() < math.exp(dE / T):
            cur = nxt
            path.append(cur)
        T *= alpha
    return path if cur == world.goal else None, expanded

# ==============================
# SIMULATION WITH DYNAMIC REPLANNING
# ==============================
class SimLog:
    def __init__(self): self.steps, self.replans, self.success, self.final_cost = 0, 0, False, float('inf')

def simulate(world, planner_func, planner_name="Planner", max_steps=200):
    log = SimLog()
    cur = world.start
    t = 0
    path, _ = planner_func(world)
    while path and t < max_steps:
        if cur == world.goal:
            log.success = True
            break
        nxt = path[1]
        if not world.is_valid(*nxt, t):  # obstacle appeared
            path, _ = planner_func(world)
            log.replans += 1
            if not path: break
            continue
        cur = nxt
        path = path[1:]
        t += 1
    log.steps = t
    log.final_cost = path_cost(world, reconstruct({path[i+1]: path[i] for i in range(len(path)-1)} | {world.start: None}, world.goal)) if path else float('inf')
    return log

# ==============================
# EXPERIMENT RUNNER
# ==============================
def run_experiments():
    maps = {
        "small": [
            [1,1,1,1],
            [1,-1,5,1],
            [1,1,1,1],
            [1,1,1,1]
        ],
        "medium": [[1]*10 for _ in range(10)],
        "large": [[1]*20 for _ in range(20)],
        "dynamic": [
            [1,1,1,1,1],
            [1,-1,1,-1,1],
            [1,1,1,1,1],
            [1,-1,1,-1,1],
            [1,1,1,1,1]
        ]
    }
    dyn_sched = {3:[(2,2)], 5:[(3,2)]}

    planners = {
        "BFS": bfs,
        "UCS": uniform_cost,
        "A*": astar,
        "HillClimb": hill_climb,
        "SimAnneal": simulated_annealing
    }

    results = []
    for mname, grid in maps.items():
        start, goal = (0,0), (len(grid)-1, len(grid[0])-1)
        world = GridWorld(grid, start, goal, dyn_sched if mname=="dynamic" else None)
        for pname, pfunc in planners.items():
            t0 = time.time()
            path, expanded = pfunc(world)
            t1 = time.time()
            sim_log = simulate(world, pfunc, planner_name=pname)
            results.append({
                "map": mname, "planner": pname,
                "nodes_expanded": expanded,
                "time": round(t1-t0,4),
                "path_cost": path_cost(world, path),
                "sim_success": sim_log.success,
                "replans": sim_log.replans
            })
            print(f"{mname}-{pname}: cost={path_cost(world,path)} expanded={expanded} success={sim_log.success}")

    with open("results.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)

    # Simple plot
    plt.figure(figsize=(10,5))
    for pname in planners.keys():
        xs = [r["map"] for r in results if r["planner"]==pname]
        ys = [r["path_cost"] for r in results if r["planner"]==pname]
        plt.plot(xs, ys, marker="o", label=pname)
    plt.legend(); plt.ylabel("Path Cost"); plt.title("Planner Comparison")
    plt.savefig("results_plot.png")
    plt.show()

# ==============================
# MAIN
# ==============================
print("Autonomous Delivery Agent - Running Experiments...")
run_experiments()
print("Done. Results saved to results.csv and results_plot.png")

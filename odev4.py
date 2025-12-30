import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from math import sqrt, cos, sin, pi
try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except:
    FOLIUM_AVAILABLE = False
    print("Folium not available. Map visualization will be skipped.")

# ---------------------------------------------------------
# OR-Tools check (optional)
# ---------------------------------------------------------
USE_ORTOOLS = False
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    USE_ORTOOLS = True
except:
    USE_ORTOOLS = False


# =========================================================
# Distance (Euclidean)
# =========================================================
def dist(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


# =========================================================
# Generate random topology
# =========================================================
def generate_topology(n_nodes, seed):
    rng = np.random.default_rng(seed)
    return rng.random((n_nodes, 2))


# =========================================================
# Neighborhood (Bölge) Tanımları
# =========================================================
class CircleNeighborhood:
    """Daire şeklinde bölge"""
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
    
    def contains(self, point):
        return dist(self.center, point) <= self.radius
    
    def sample_point(self):
        """Bölge içinden rastgele nokta"""
        angle = np.random.uniform(0, 2 * pi)
        r = self.radius * np.sqrt(np.random.uniform(0, 1))
        return self.center + r * np.array([cos(angle), sin(angle)])
    
    def get_boundary_points(self, n_points=20):
        """Görselleştirme için sınır noktaları"""
        angles = np.linspace(0, 2 * pi, n_points)
        return [self.center + self.radius * np.array([cos(a), sin(a)]) for a in angles]


class PolygonNeighborhood:
    """Poligon şeklinde bölge (basit dörtgen)"""
    def __init__(self, center, size):
        self.center = np.array(center)
        self.size = size
        # Merkez etrafında kare oluştur
        half = size / 2
        self.vertices = [
            center + np.array([-half, -half]),
            center + np.array([half, -half]),
            center + np.array([half, half]),
            center + np.array([-half, half])
        ]
    
    def contains(self, point):
        """Point-in-polygon test (basit kare için)"""
        x, y = point
        cx, cy = self.center
        half = self.size / 2
        return (cx - half <= x <= cx + half) and (cy - half <= y <= cy + half)
    
    def sample_point(self):
        """Bölge içinden rastgele nokta"""
        half = self.size / 2
        offset = np.random.uniform(-half, half, 2)
        return self.center + offset
    
    def get_boundary_points(self):
        """Görselleştirme için köşe noktaları"""
        return self.vertices + [self.vertices[0]]  # Kapalı poligon


def generate_neighborhoods(n_nodes, seed, neighborhood_type="circle", radius_range=(0.05, 0.15)):
    """Rastgele bölgeler oluştur"""
    rng = np.random.default_rng(seed)
    centers = rng.random((n_nodes, 2))
    neighborhoods = []
    
    for center in centers:
        if neighborhood_type == "circle":
            radius = rng.uniform(radius_range[0], radius_range[1])
            neighborhoods.append(CircleNeighborhood(center, radius))
        else:  # polygon
            size = rng.uniform(radius_range[0] * 2, radius_range[1] * 2)
            neighborhoods.append(PolygonNeighborhood(center, size))
    
    return neighborhoods


# =========================================================
# Tour length
# =========================================================
def tour_length(points, tour):
    total = 0.0
    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i + 1) % len(tour)]
        total += dist(points[a], points[b])
    return total


# =========================================================
# Optimal Point Selection in Neighborhoods
# =========================================================
def select_optimal_points(neighborhoods, tour, max_iterations=3):  # Daha hızlı için azaltıldı
    """
    Verilen bir tur sırası için her bölge içinden optimal noktaları seç.
    İteratif olarak tüm noktaları optimize eder.
    """
    n = len(neighborhoods)
    # Başlangıç: tüm noktalar merkezde
    selected_points = np.array([n.center.copy() for n in neighborhoods])
    
    # İteratif optimizasyon
    for iteration in range(max_iterations):
        improved = False
        for idx in range(n):
            i = tour[idx]
            prev_i = tour[(idx - 1) % n]
            next_i = tour[(idx + 1) % n]
            
            prev_point = selected_points[prev_i]
            next_point = selected_points[next_i]
            
            # Bu bölge için optimal nokta: prev ve next'e en yakın nokta
            def objective(p):
                return dist(prev_point, p) + dist(p, next_point)
            
            # Başlangıç noktası: mevcut seçilen nokta
            x0 = selected_points[i]
            
            # Sınırlar: bölge içinde kal
            if isinstance(neighborhoods[i], CircleNeighborhood):
                # Daire için: merkezden radius içinde
                center = neighborhoods[i].center
                radius = neighborhoods[i].radius
                
                # Gradient descent benzeri basit optimizasyon
                best_point = x0.copy()
                best_val = objective(best_point)
                
                # Merkeze doğru ve kenarlara doğru birkaç nokta dene
                for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    # Merkezden kenara doğru
                    direction = (x0 - center)
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        # Rastgele yön
                        angle = np.random.uniform(0, 2 * pi)
                        direction = np.array([cos(angle), sin(angle)])
                    
                    test_point = center + alpha * radius * direction
                    if neighborhoods[i].contains(test_point):
                        val = objective(test_point)
                        if val < best_val:
                            best_val = val
                            best_point = test_point
                            improved = True
                
                selected_points[i] = best_point
            else:
                # Poligon için: sınırlar içinde
                center = neighborhoods[i].center
                half = neighborhoods[i].size / 2
                
                best_point = x0.copy()
                best_val = objective(best_point)
                
                # Grid search benzeri
                for dx in [-half, -half/2, 0, half/2, half]:
                    for dy in [-half, -half/2, 0, half/2, half]:
                        test_point = center + np.array([dx, dy])
                        if neighborhoods[i].contains(test_point):
                            val = objective(test_point)
                            if val < best_val:
                                best_val = val
                                best_point = test_point
                                improved = True
                
                selected_points[i] = best_point
        
        # Eğer iyileşme yoksa dur
        if not improved and iteration > 0:
            break
    
    return selected_points


# =========================================================
# Nearest Neighbor
# =========================================================
def nearest_neighbor(points):
    n = len(points)
    visited = [False] * n
    tour = [0]
    visited[0] = True

    for _ in range(n - 1):
        last = tour[-1]
        best = None
        best_d = 1e18

        for i in range(n):
            if not visited[i]:
                d = dist(points[last], points[i])
                if d < best_d:
                    best_d = d
                    best = i

        tour.append(best)
        visited[best] = True

    return tour


# =========================================================
# 2-opt
# =========================================================
def two_opt(points, tour):
    improved = True
    best = tour[:]
    best_len = tour_length(points, best)
    n = len(tour)

    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_tour = best[:]
                new_tour[i:j] = reversed(new_tour[i:j])
                new_len = tour_length(points, new_tour)

                if new_len < best_len:
                    best = new_tour
                    best_len = new_len
                    improved = True

    return best


def heuristic_solver(points):
    tour = nearest_neighbor(points)
    tour = two_opt(points, tour)
    return tour


def heuristic_solver_with_neighborhoods(neighborhoods):
    """Neighborhoods için heuristic solver"""
    # Merkez noktaları kullanarak tur bul
    centers = np.array([n.center for n in neighborhoods])
    tour = nearest_neighbor(centers)
    tour = two_opt(centers, tour)
    
    # Optimal noktaları seç
    selected_points = select_optimal_points(neighborhoods, tour)
    
    # Seçilen noktalarla 2-opt iyileştirme
    tour = two_opt(selected_points, tour)
    
    # Tekrar optimal nokta seçimi
    selected_points = select_optimal_points(neighborhoods, tour)
    
    return tour, selected_points


# =========================================================
# Simulated Annealing
# =========================================================
def simulated_annealing(points, iterations=2000):
    n = len(points)
    tour = np.arange(n)
    np.random.shuffle(tour)

    best = tour.copy()
    best_len = tour_length(points, best)

    T = 1.0

    for _ in range(iterations):
        i, j = sorted(np.random.choice(n, 2, replace=False))
        new = tour.copy()
        new[i:j] = new[i:j][::-1]

        Lnew = tour_length(points, new)

        if Lnew < best_len or np.random.rand() < np.exp((best_len - Lnew) / T):
            tour = new
            if Lnew < best_len:
                best = new
                best_len = Lnew

        T *= 0.9995

    return best


def simulated_annealing_with_neighborhoods(neighborhoods, iterations=1000):  # Daha hızlı için azaltıldı
    """Neighborhoods için Simulated Annealing"""
    n = len(neighborhoods)
    centers = np.array([n.center for n in neighborhoods])
    
    tour = np.arange(n)
    np.random.shuffle(tour)
    
    # İlk optimal nokta seçimi
    selected_points = select_optimal_points(neighborhoods, tour)
    best = tour.copy()
    best_points = selected_points.copy()
    best_len = tour_length(selected_points, best)
    
    T = 1.0
    
    for _ in range(iterations):
        i, j = sorted(np.random.choice(n, 2, replace=False))
        new_tour = tour.copy()
        new_tour[i:j] = new_tour[i:j][::-1]
        
        # Yeni tur için optimal noktaları seç
        new_points = select_optimal_points(neighborhoods, new_tour)
        Lnew = tour_length(new_points, new_tour)
        
        if Lnew < best_len or np.random.rand() < np.exp((best_len - Lnew) / T):
            tour = new_tour
            selected_points = new_points
            if Lnew < best_len:
                best = new_tour
                best_points = new_points
                best_len = Lnew
        
        T *= 0.9995
    
    return best, best_points


# =========================================================
# Genetic Algorithm (AI SOLVER)
# =========================================================
def genetic_algorithm_solver(
    points,
    pop_size=80,
    generations=300,
    mutation_rate=0.15,
    elite_size=5
):
    n = len(points)

    def create_individual():
        ind = np.arange(n)
        np.random.shuffle(ind)
        return ind

    def fitness(ind):
        return 1.0 / tour_length(points, ind)

    def crossover(p1, p2):
        a, b = sorted(np.random.choice(n, 2, replace=False))
        child = [-1] * n
        child[a:b] = p1[a:b]

        ptr = 0
        for x in p2:
            if x not in child:
                while child[ptr] != -1:
                    ptr += 1
                child[ptr] = x

        return np.array(child)

    def mutate(ind):
        if np.random.rand() < mutation_rate:
            i, j = np.random.choice(n, 2, replace=False)
            ind[i], ind[j] = ind[j], ind[i]

    # Initial population
    population = [create_individual() for _ in range(pop_size)]

    for _ in range(generations):
        population = sorted(population, key=lambda x: tour_length(points, x))
        new_pop = population[:elite_size]

        fitness_vals = np.array([fitness(ind) for ind in population])
        probs = fitness_vals / fitness_vals.sum()

        while len(new_pop) < pop_size:
            idx1 = np.random.choice(len(population), p=probs)
            idx2 = np.random.choice(len(population), p=probs)
            p1, p2 = population[idx1], population[idx2]
            child = crossover(p1, p2)
            mutate(child)
            new_pop.append(child)

        population = new_pop

    best = min(population, key=lambda x: tour_length(points, x))
    return best


def genetic_algorithm_solver_with_neighborhoods(
    neighborhoods,
    pop_size=40,  # Daha hızlı için azaltıldı
    generations=100,  # Daha hızlı için azaltıldı
    mutation_rate=0.15,
    elite_size=5
):
    """Neighborhoods için Genetic Algorithm"""
    n = len(neighborhoods)
    
    def create_individual():
        ind = np.arange(n)
        np.random.shuffle(ind)
        return ind
    
    def get_tour_length(ind):
        # Optimal noktaları seç ve tur uzunluğunu hesapla
        selected_points = select_optimal_points(neighborhoods, ind)
        return tour_length(selected_points, ind)
    
    def crossover(p1, p2):
        a, b = sorted(np.random.choice(n, 2, replace=False))
        child = [-1] * n
        child[a:b] = p1[a:b]
        
        ptr = 0
        for x in p2:
            if x not in child:
                while child[ptr] != -1:
                    ptr += 1
                child[ptr] = x
        
        return np.array(child)
    
    def mutate(ind):
        if np.random.rand() < mutation_rate:
            i, j = np.random.choice(n, 2, replace=False)
            ind[i], ind[j] = ind[j], ind[i]
    
    # Initial population
    population = [create_individual() for _ in range(pop_size)]
    
    for gen in range(generations):
        # Fitness hesaplama (optimal nokta seçimi ile)
        tour_lengths = [get_tour_length(ind) for ind in population]
        population = sorted(population, key=lambda x: get_tour_length(x))
        new_pop = population[:elite_size]
        
        fitness_vals = np.array([1.0 / tl for tl in tour_lengths])
        probs = fitness_vals / fitness_vals.sum()
        
        while len(new_pop) < pop_size:
            idx1 = np.random.choice(len(population), p=probs)
            idx2 = np.random.choice(len(population), p=probs)
            p1, p2 = population[idx1], population[idx2]
            child = crossover(p1, p2)
            mutate(child)
            new_pop.append(child)
        
        population = new_pop
    
    # En iyi çözümü bul
    best = min(population, key=lambda x: get_tour_length(x))
    best_points = select_optimal_points(neighborhoods, best)
    
    return best, best_points


# =========================================================
# OR-Tools solver (optional baseline)
# =========================================================
def solve_ortools(points):
    n = len(points)
    dist_matrix = [[dist(points[i], points[j]) for j in range(n)] for i in range(n)]

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def cb(i, j):
        a = manager.IndexToNode(i)
        b = manager.IndexToNode(j)
        return int(dist_matrix[a][b] * 1000)

    transit = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 2

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None

    idx = routing.Start(0)
    tour = []
    while not routing.IsEnd(idx):
        tour.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))

    return tour


def solve_ortools_with_neighborhoods(neighborhoods):
    """Neighborhoods için OR-Tools (merkez noktaları kullan)"""
    centers = np.array([n.center for n in neighborhoods])
    tour = solve_ortools(centers)
    if tour is None:
        return None, None
    selected_points = select_optimal_points(neighborhoods, tour)
    return tour, selected_points


# =========================================================
# Visualization with Folium
# =========================================================
def visualize_tour_with_neighborhoods(neighborhoods, tour, selected_points, method_name, output_file):
    """Folium ile harita görselleştirmesi - Nilüfer / Bursa bölgesi"""
    if not FOLIUM_AVAILABLE:
        print(f"Folium not available. Skipping visualization for {method_name}")
        return
    
    # Bursa/Nilüfer koordinat aralıkları
    # Nilüfer ilçesi: Lat 40.15-40.30, Lon 28.9-29.1
    BURSA_LAT_MIN = 40.15
    BURSA_LAT_MAX = 40.30
    BURSA_LON_MIN = 28.9
    BURSA_LON_MAX = 29.1
    BURSA_LAT_CENTER = 40.225  # Nilüfer merkez
    BURSA_LON_CENTER = 29.0
    
    # [0,1] aralığındaki koordinatları Bursa/Nilüfer koordinatlarına dönüştür
    def to_bursa_coords(x, y):
        lat = BURSA_LAT_MIN + y * (BURSA_LAT_MAX - BURSA_LAT_MIN)
        lon = BURSA_LON_MIN + x * (BURSA_LON_MAX - BURSA_LON_MIN)
        return lat, lon
    
    # Merkez noktasını bul (tüm noktaların ortalaması)
    all_points = np.array([n.center for n in neighborhoods])
    center_lat, center_lon = to_bursa_coords(
        float(np.mean(all_points[:, 0])),
        float(np.mean(all_points[:, 1]))
    )
    
    # Harita oluştur - Nilüfer / Bursa
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Bölge bilgisi ekle
    folium.Marker(
        location=[BURSA_LAT_CENTER, BURSA_LON_CENTER],
        popup="Nilüfer / Bursa",
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(m)
    
    # Bölgeleri çiz
    for i, n in enumerate(neighborhoods):
        center = n.center
        lat, lon = to_bursa_coords(float(center[0]), float(center[1]))
        
        if isinstance(n, CircleNeighborhood):
            # Radius'u derece cinsinden yaklaşık mesafeye çevir (1 derece ≈ 111 km)
            # Küçük radius için daha küçük ölçekleme
            radius_meters = n.radius * (BURSA_LAT_MAX - BURSA_LAT_MIN) * 111000  # Metre cinsinden
            folium.Circle(
                location=[lat, lon],
                radius=radius_meters,
                popup=f"Bölge {i+1}",
                color='blue',
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.3
            ).add_to(m)
        else:
            # Poligon için köşe noktaları
            vertices = n.get_boundary_points()
            coords = [to_bursa_coords(float(v[0]), float(v[1])) for v in vertices]
            coords = [[lat, lon] for lat, lon in coords]  # Liste formatına çevir
            folium.Polygon(
                locations=coords,
                popup=f"Bölge {i+1}",
                color='blue',
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.3
            ).add_to(m)
    
    # Seçilen noktaları işaretle
    for i, point in enumerate(selected_points):
        lat, lon = to_bursa_coords(float(point[0]), float(point[1]))
        folium.Marker(
            location=[lat, lon],
            popup=f"Nokta {i+1}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    # Turu çiz
    tour_coords = []
    for idx in tour:
        point = selected_points[idx]
        lat, lon = to_bursa_coords(float(point[0]), float(point[1]))
        tour_coords.append([lat, lon])
    
    # Turu kapat
    tour_coords.append(tour_coords[0])
    
    folium.PolyLine(
        locations=tour_coords,
        color='red',
        weight=3,
        popup=f"{method_name} Tour"
    ).add_to(m)
    
    # Haritayı kaydet
    m.save(output_file)
    print(f"Map saved to {output_file}")


# =========================================================
# Experiment Runner (30 instances)
# =========================================================
def run_experiment(
    n_topologies=30,
    n_nodes=50,
    base_seed=12345,
    output_dir="tsp_assignment4_output"
):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for t in range(n_topologies):
        seed = base_seed + t
        points = generate_topology(n_nodes, seed)

        # Heuristic
        t0 = time.time()
        h_tour = heuristic_solver(points)
        h_len = tour_length(points, h_tour)
        h_rt = time.time() - t0
        rows.append(["Heuristic", t, h_len, h_rt, seed])

        # Simulated Annealing
        t0 = time.time()
        sa_tour = simulated_annealing(points)
        sa_tour = two_opt(points, sa_tour)
        sa_len = tour_length(points, sa_tour)
        sa_rt = time.time() - t0
        rows.append(["SA", t, sa_len, sa_rt, seed])

        # Genetic Algorithm (AI)
        t0 = time.time()
        ga_tour = genetic_algorithm_solver(points)
        ga_len = tour_length(points, ga_tour)
        ga_rt = time.time() - t0
        rows.append(["GeneticAlgorithm", t, ga_len, ga_rt, seed])

        # OR-Tools (optional)
        if USE_ORTOOLS:
            t0 = time.time()
            o_tour = solve_ortools(points)
            o_rt = time.time() - t0
            if o_tour is not None:
                o_len = tour_length(points, o_tour)
                rows.append(["OR-Tools", t, o_len, o_rt, seed])

    # Save results
    df = pd.DataFrame(rows, columns=[
        "method", "topology", "tour_length", "runtime", "seed"
    ])
    df.to_csv(f"{output_dir}/results.csv", index=False)

    summary = df.groupby("method")[["tour_length", "runtime"]].agg(["mean", "std"])
    summary.to_csv(f"{output_dir}/summary.csv")

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------
    plt.figure()
    df.boxplot(column="tour_length", by="method")
    plt.title("Tour Length Comparison")
    plt.suptitle("")
    plt.ylabel("Tour Length")
    plt.savefig(f"{output_dir}/tour_length_boxplot.png", dpi=300)
    plt.close()

    plt.figure()
    for m in df["method"].unique():
        d = df[df["method"] == m]
        plt.scatter(d["runtime"], d["tour_length"], label=m)
    plt.xlabel("Runtime (s)")
    plt.ylabel("Tour Length")
    plt.legend()
    plt.savefig(f"{output_dir}/runtime_vs_length.png", dpi=300)
    plt.close()

    # Combined plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df.boxplot(column="tour_length", by="method", ax=axes[0])
    axes[0].set_title("Tour Length")
    axes[0].set_xlabel("")

    for m in df["method"].unique():
        d = df[df["method"] == m]
        axes[1].scatter(d["runtime"], d["tour_length"], label=m)
    axes[1].set_xlabel("Runtime (s)")
    axes[1].set_ylabel("Tour Length")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_plots.png", dpi=300)
    plt.close()

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------
    with open(f"{output_dir}/comparison_report.txt", "w") as f:
        f.write("=== Assignment 4: AI Technique Integration ===\n")
        f.write("AI Method: Genetic Algorithm\n")
        f.write(f"OR-Tools available: {USE_ORTOOLS}\n\n")
        f.write(summary.to_string())

    print("Experiment finished.")
    print("Results saved in:", output_dir)


# =========================================================
# Assignment 5: TSP with Neighborhoods
# =========================================================
def run_experiment_with_neighborhoods(
    n_topologies=10,  # Daha hızlı test için azaltıldı
    n_nodes=15,  # Daha hızlı hesaplama için azaltıldı
    base_seed=12345,
    output_dir="tsp_assignment5_output",
    neighborhood_type="circle"
):
    """Assignment 5: Neighborhoods ile TSP deneyi"""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    
    # İlk birkaç topoloji için görselleştirme
    visualize_count = min(3, n_topologies)
    
    for t in range(n_topologies):
        seed = base_seed + t
        neighborhoods = generate_neighborhoods(n_nodes, seed, neighborhood_type)
        
        print(f"Processing topology {t+1}/{n_topologies}...")
        
        # Heuristic
        t0 = time.time()
        h_tour, h_points = heuristic_solver_with_neighborhoods(neighborhoods)
        h_len = tour_length(h_points, h_tour)
        h_rt = time.time() - t0
        rows.append(["Heuristic", t, h_len, h_rt, seed])
        
        # Simulated Annealing
        t0 = time.time()
        sa_tour, sa_points = simulated_annealing_with_neighborhoods(neighborhoods)
        sa_len = tour_length(sa_points, sa_tour)
        sa_rt = time.time() - t0
        rows.append(["SA", t, sa_len, sa_rt, seed])
        
        # Genetic Algorithm (AI)
        t0 = time.time()
        ga_tour, ga_points = genetic_algorithm_solver_with_neighborhoods(neighborhoods)
        ga_len = tour_length(ga_points, ga_tour)
        ga_rt = time.time() - t0
        rows.append(["GeneticAlgorithm", t, ga_len, ga_rt, seed])
        
        # OR-Tools (optional)
        if USE_ORTOOLS:
            t0 = time.time()
            o_tour, o_points = solve_ortools_with_neighborhoods(neighborhoods)
            o_rt = time.time() - t0
            if o_tour is not None:
                o_len = tour_length(o_points, o_tour)
                rows.append(["OR-Tools", t, o_len, o_rt, seed])
        
        # Görselleştirme (ilk birkaç topoloji için)
        if t < visualize_count:
            vis_dir = f"{output_dir}/visualizations"
            os.makedirs(vis_dir, exist_ok=True)
            
            visualize_tour_with_neighborhoods(
                neighborhoods, h_tour, h_points, 
                "Heuristic", f"{vis_dir}/topology_{t}_heuristic.html"
            )
            visualize_tour_with_neighborhoods(
                neighborhoods, sa_tour, sa_points,
                "Simulated_Annealing", f"{vis_dir}/topology_{t}_sa.html"
            )
            visualize_tour_with_neighborhoods(
                neighborhoods, ga_tour, ga_points,
                "Genetic_Algorithm", f"{vis_dir}/topology_{t}_ga.html"
            )
            if USE_ORTOOLS and o_tour is not None:
                visualize_tour_with_neighborhoods(
                    neighborhoods, o_tour, o_points,
                    "OR-Tools", f"{vis_dir}/topology_{t}_ortools.html"
                )
    
    # Sonuçları kaydet
    df = pd.DataFrame(rows, columns=[
        "method", "topology", "tour_length", "runtime", "seed"
    ])
    df.to_csv(f"{output_dir}/results.csv", index=False)
    
    summary = df.groupby("method")[["tour_length", "runtime"]].agg(["mean", "std"])
    summary.to_csv(f"{output_dir}/summary.csv")
    
    # Grafikler
    plt.figure(figsize=(10, 6))
    df.boxplot(column="tour_length", by="method")
    plt.title("Tour Length Comparison (TSP with Neighborhoods)")
    plt.suptitle("")
    plt.ylabel("Tour Length")
    plt.xlabel("Method")
    plt.savefig(f"{output_dir}/tour_length_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    for m in df["method"].unique():
        d = df[df["method"] == m]
        plt.scatter(d["runtime"], d["tour_length"], label=m, alpha=0.6)
    plt.xlabel("Runtime (s)")
    plt.ylabel("Tour Length")
    plt.legend()
    plt.title("Runtime vs Tour Length (TSP with Neighborhoods)")
    plt.savefig(f"{output_dir}/runtime_vs_length.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Kombine grafik
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    df.boxplot(column="tour_length", by="method", ax=axes[0])
    axes[0].set_title("Tour Length Comparison")
    axes[0].set_xlabel("")
    
    for m in df["method"].unique():
        d = df[df["method"] == m]
        axes[1].scatter(d["runtime"], d["tour_length"], label=m, alpha=0.6)
    axes[1].set_xlabel("Runtime (s)")
    axes[1].set_ylabel("Tour Length")
    axes[1].set_title("Runtime vs Tour Length")
    axes[1].legend()
    
    plt.suptitle("TSP with Neighborhoods - Assignment 5", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Rapor oluştur
    with open(f"{output_dir}/assignment5_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Assignment 5: TSP with Neighborhoods\n")
        f.write("=" * 60 + "\n\n")
        f.write("Bu çalışmada, TSP problemi neighborhoods (bölgeler) ile genişletilmiştir.\n")
        f.write("Her şehir artık sabit bir nokta değil, sürekli bir bölgedir (daire veya poligon).\n")
        f.write("Çözücüler, her bölge içinden optimal noktaları seçerek tur uzunluğunu minimize eder.\n\n")
        f.write(f"Neighborhood Type: {neighborhood_type}\n")
        f.write(f"Number of Nodes: {n_nodes}\n")
        f.write(f"Number of Topologies: {n_topologies}\n")
        f.write(f"OR-Tools available: {USE_ORTOOLS}\n")
        f.write(f"Folium available: {FOLIUM_AVAILABLE}\n\n")
        f.write("-" * 60 + "\n")
        f.write("Summary Statistics:\n")
        f.write("-" * 60 + "\n\n")
        f.write(summary.to_string())
        f.write("\n\n")
        f.write("-" * 60 + "\n")
        f.write("Methods:\n")
        f.write("-" * 60 + "\n")
        f.write("1. Heuristic: Nearest Neighbor + 2-opt + Optimal Point Selection\n")
        f.write("2. Simulated Annealing: SA with optimal point selection in neighborhoods\n")
        f.write("3. Genetic Algorithm: AI-based approach with population search\n")
        if USE_ORTOOLS:
            f.write("4. OR-Tools: Industrial-grade solver (baseline)\n")
        f.write("\n")
        f.write("Key Observations:\n")
        f.write("- Neighborhoods problemi daha karmaşık hale getirir\n")
        f.write("- Optimal nokta seçimi tur kalitesini önemli ölçüde etkiler\n")
        f.write("- Genetic Algorithm, popülasyon tabanlı arama sayesinde iyi sonuçlar üretir\n")
        f.write("- Runtime, optimal nokta seçimi nedeniyle artmıştır\n")
    
    print("\n" + "=" * 60)
    print("Assignment 5 Experiment Finished!")
    print("=" * 60)
    print(f"Results saved in: {output_dir}")
    print(f"- results.csv: Detailed results")
    print(f"- summary.csv: Statistical summary")
    print(f"- comparison_plots.png: Visualization")
    if visualize_count > 0:
        print(f"- visualizations/: Interactive maps for first {visualize_count} topologies")
    print(f"- assignment5_report.txt: Detailed report")
    print("=" * 60)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "assignment5":
        # Assignment 5: TSP with Neighborhoods
        run_experiment_with_neighborhoods()
    else:
        # Assignment 4: Original TSP
        run_experiment()
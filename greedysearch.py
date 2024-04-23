from queue import PriorityQueue

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v, cost):
        if u not in self.graph:
            self.graph[u] = {}
        self.graph[u][v] = cost

def greedy_best_first_search(graph, start, goal):
    priority_queue = PriorityQueue()
    priority_queue.put((0, start))
    visited = set()
    
    while not priority_queue.empty():
        _, current_node = priority_queue.get()
        if current_node in visited:
            continue
        print(current_node, end=' ')
        if current_node == goal:
            print("\nGoal Reached!")
            return
        visited.add(current_node)
        for neighbor, cost in graph.graph.get(current_node, {}).items():
            if neighbor not in visited:
                priority_queue.put((cost, neighbor))
    
    print("\nGoal not reached.")

g = Graph()


while True:
    u = input("Enter start node of edge (or 'done' to finish): ")
    if u.lower() == 'done':
        break
    v = input("Enter end node of edge: ")
    cost = int(input("Enter cost of edge: "))
    g.add_edge(u, v, cost)

start_node = input("Enter the starting node for Greedy Best-First Search: ")
goal_node = input("Enter the goal node for Greedy Best-First Search: ")

print("Greedy Best-First Search:")
greedy_best_first_search(g, start_node, goal_node)

# sample Output:
# Enter start node of edge (or 'done' to finish): A
# Enter end node of edge: B
# Enter cost of edge: 1
# Enter start node of edge (or 'done' to finish): A
# Enter end node of edge: C
# Enter cost of edge: 4
# Enter start node of edge (or 'done' to finish): B
# Enter end node of edge: D
# Enter cost of edge: 2
# Enter start node of edge (or 'done' to finish): C
# Enter end node of edge: D
# Enter cost of edge: 5
# Enter start node of edge (or 'done' to finish): D
# Enter end node of edge: E
# Enter cost of edge: 3
# Enter start node of edge (or 'done' to finish): done
# Enter the starting node for Greedy Best-First Search: A
# Enter the goal node for Greedy Best-First Search: E
# Greedy Best-First Search:
# A B D E 
# Goal Reached!

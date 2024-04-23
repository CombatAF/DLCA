from queue import PriorityQueue

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v, cost):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append((v, cost))

def a_star_search(graph, start, goal):
    priority_queue = PriorityQueue()
    priority_queue.put((0, start))
    visited = set()
    
    while not priority_queue.empty():
        cost, current_node = priority_queue.get()
        if current_node in visited:
            continue
        print(current_node, end=' ')
        if current_node == goal:
            print("\nGoal Reached!")
            return
        visited.add(current_node)
        for neighbor, edge_cost in graph.graph.get(current_node, []):
            if neighbor not in visited:
                priority_queue.put((edge_cost + heuristic(neighbor, goal), neighbor))
    
    print("\nGoal not reached.")

def heuristic(node, goal):
    return 0

g = Graph()


while True:
    u = input("Enter start node of edge (or 'done' to finish): ")
    if u.lower() == 'done':
        break
    v = input("Enter end node of edge: ")
    cost = int(input("Enter cost of edge: "))
    g.add_edge(u, v, cost)

start_node = input("Enter the starting node for A* search: ")
goal_node = input("Enter the goal node for A* search: ")

print("A* Search:")
a_star_search(g, start_node, goal_node)

#  sample Output:
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
# Enter the starting node for A* search: A
# Enter the goal node for A* search: E
# A* Search:
# A B D E 
# Goal Reached!


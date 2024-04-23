from collections import deque

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        print(node, end=" ")

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)

g = Graph()


while True:
    u = input("Enter start node of edge (or 'done' to finish): ")
    if u.lower() == 'done':
        break
    v = input("Enter end node of edge: ")
    g.add_edge(u, v)

start_node = input("Enter the starting node for BFS: ")

print("BFS Traversal:")
bfs(g.graph, start_node)

# sample Output:
# Enter start node of edge (or 'done' to finish):  A
# Enter end node of edge:  B
# Enter start node of edge (or 'done' to finish):  A
# Enter end node of edge:  C
# Enter start node of edge (or 'done' to finish):  B
# Enter end node of edge:  D
# Enter start node of edge (or 'done' to finish):  B
# Enter end node of edge:  E
# Enter start node of edge (or 'done' to finish):  C
# Enter end node of edge:  F
# Enter start node of edge (or 'done' to finish):  done
# Enter the starting node for BFS:  A
# BFS Traversal:
# A B C D E F 


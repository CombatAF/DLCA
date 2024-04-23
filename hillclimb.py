import random
import numpy as np

def generate_matrix(coordinate):
    matrix = []
    for i in range(len(coordinate)):
        for j in range(len(coordinate)):
            p = np.linalg.norm(coordinate[i] - coordinate[j])
            matrix.append(p)
    matrix = np.reshape(matrix, (len(coordinate), len(coordinate)))
    return matrix

def solution(matrix):
    points = list(range(len(matrix)))
    random.shuffle(points)
    return points

def path_length(matrix, solution):
    cycle_length = 0
    for i in range(len(solution)):
        cycle_length += matrix[solution[i]][solution[i - 1]]
    return cycle_length

def neighbors(matrix, solution):
    neighbors_list = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbor = solution.copy()
            neighbor[i] = solution[j]
            neighbor[j] = solution[i]
            neighbors_list.append(neighbor)

    best_neighbor = neighbors_list[0]
    best_path = path_length(matrix, best_neighbor)

    for neighbor in neighbors_list:
        current_path = path_length(matrix, neighbor)
        if current_path < best_path:
            best_path = current_path
            best_neighbor = neighbor
    return best_neighbor, best_path

def hill_climbing(coordinate):
    matrix = generate_matrix(coordinate)
    current_solution = solution(matrix)
    current_path = path_length(matrix, current_solution)
    neighbor = neighbors(matrix, current_solution)[0]
    best_neighbor, best_neighbor_path = neighbors(matrix, neighbor)

    while best_neighbor_path < current_path:
        current_solution = best_neighbor
        current_path = best_neighbor_path
        neighbor = neighbors(matrix, current_solution)[0]
        best_neighbor, best_neighbor_path = neighbors(matrix, neighbor)

    return current_path, current_solution


coordinate = []
num_cities = int(input("Enter the number of cities: "))
for i in range(num_cities):
    x, y = map(int, input(f"Enter the coordinates for city {i+1} (x y): ").split())
    coordinate.append([x, y])

final_solution = hill_climbing(np.array(coordinate))
print("The solution is:")
print(final_solution[1])
print("Total path length:", final_solution[0])


# sample Output:
# Enter the number of cities:  5
# Enter the coordinates for city 1 (x y):  1 2
# Enter the coordinates for city 2 (x y):  3 4
# Enter the coordinates for city 3 (x y):  5 6
# Enter the coordinates for city 4 (x y):  7 8
# Enter the coordinates for city 5 (x y):  9 10
# The solution is:
# [3, 1, 0, 2, 4]
# Total path length: 22.627416997969522


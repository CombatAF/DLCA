

class GoalStackPlanner:
    def __init__(self):
        self.stack = []

    def add_goal(self, goal):
        self.stack.append(goal)

    def execute(self):
        while self.stack:
            current_goal = self.stack.pop()
            if isinstance(current_goal, str):
                print(f"Executing action: {current_goal}")
            else:
                print(f"Decomposing goal: {current_goal[0]}")
                for subgoal in current_goal[1:]:
                    self.stack.append(subgoal)

if __name__ == "__main__":
    planner = GoalStackPlanner()

    
    while True:
        goal_input = input("Enter a goal (or 'done' to finish adding goals): ")
        if goal_input.lower() == 'done':
            break
        planner.add_goal(goal_input)

  
    print("Executing goals:")
    planner.execute()

# sample Output:
# Enter a goal (or 'done' to finish adding goals): Reach Destination
# Enter a goal (or 'done' to finish adding goals): Find Key
# Enter a goal (or 'done' to finish adding goals): Open Door
# Enter a goal (or 'done' to finish adding goals): Find Map
# Enter a goal (or 'done' to finish adding goals): Find Compass
# Enter a goal (or 'done' to finish adding goals): done
# Executing goals:
# Decomposing goal: Reach Destination
# Executing action: Find Compass
# Executing action: Find Map
# Executing action: Open Door
# Executing action: Find Key

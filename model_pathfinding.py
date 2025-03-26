# Implements pathfinding algorithms for agent movement.
# Currently not in use. Agent movement is currently handled by its neural network, allowing it to move somewhat randomly/freely.
""" 
class Pathfinding:
    def find_path(self, start, goal, grid):
        # A* pathfinding implementation
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            current = min(open_set, key=lambda x: x[0])[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            open_set = [x for x in open_set if x[1] != current]
            
            for neighbor in self._get_neighbors(current, grid):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, goal)
                    open_set.append((f_score[neighbor], neighbor))
        
        return None
    
    def _heuristic(self, a, b):
        # Manhattan distance heuristic
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    
    def _get_neighbors(self, pos, grid):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < len(grid) and 
                0 <= new_pos[1] < len(grid[0]) and 
                grid[new_pos[0]][new_pos[1]] == 0):
                neighbors.append(new_pos)
        return neighbors """
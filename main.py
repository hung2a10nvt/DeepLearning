import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import random, time
from collections import deque
import copy

class CityEnvironment:
        
    def __init__(self, map_file='citymap.txt', spawn_probability=0.02):
        self.map = self.load_map(map_file)
        self.height, self.width = self.map.shape
        self.spawn_probability = spawn_probability
        
        # Special positions
        self.start_positions = list(zip(*np.where(self.map == 2)))
        self.spawn_points = list(zip(*np.where(self.map == 3)))
        self.goal_positions = list(zip(*np.where(self.map == 4)))
        
        self.reset()
        
    def load_map(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        map_data = []
        for line in lines:
            row = [int(c) for c in line.strip()]
            map_data.append(row)
        
        return np.array(map_data)
    
    def reset(self, start_idx=0):
        """Reset environment to initial state"""
        self.agent_pos = list(self.start_positions[start_idx % len(self.start_positions)])
        self.vehicles = {}  
        self.vehicle_id_counter = 0
        self.steps = 0
        self.done = False
        self.crashed = False
        return self.get_state()
    
    def get_state(self):
        """Get current state representation"""
        state = np.copy(self.map)
        # Set agent position
        state[self.agent_pos[0], self.agent_pos[1]] = 5
        # Set vehicle positions
        for vehicle in self.vehicles.values():
            pos = vehicle['pos']
            state[pos[0], pos[1]] = 6
        return state
    
    def is_valid_position(self, pos):
        """Check if a position is valid (within bounds and not an obstacle)"""
        y, x = pos
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.map[y, x] != 1
        return False
    
    def get_valid_neighbors(self, pos):
        """Get valid neighboring positions from a given position"""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        neighbors = []
        
        for dy, dx in directions:
            new_pos = [pos[0] + dy, pos[1] + dx]
            if self.is_valid_position(new_pos):
                neighbors.append((new_pos, [dy, dx]))
        
        return neighbors
    
    def spawn_vehicle(self):
        """Spawn new vehicles"""
        for spawn_point in self.spawn_points:
            if random.random() < self.spawn_probability:
                # Check if spawn point is occupied
                occupied = any(v['pos'] == list(spawn_point) for v in self.vehicles.values())
                if not occupied:
                    # Create new vehicle
                    neighbors = self.get_valid_neighbors(spawn_point)
                    if neighbors:
                        # Pick a random direction to start
                        _, initial_direction = random.choice(neighbors)
                        steps_left = random.randint(15, 150)
                        
                        self.vehicles[self.vehicle_id_counter] = {
                            'pos': list(spawn_point),
                            'direction': initial_direction,
                            'steps_left': steps_left
                        }
                        self.vehicle_id_counter += 1
    
    def move_vehicle(self, vehicle_id):
        vehicle = self.vehicles[vehicle_id]
        current_pos = vehicle['pos']
        current_direction = vehicle['direction']
        
        # Get valid neighbors
        neighbors = self.get_valid_neighbors(current_pos)
        
        if not neighbors:
            return  # Cant move
        
        # Continue in the same direction with 70% probability
        if random.random() < 0.7:
            # Try to move in the same direction
            next_pos = [current_pos[0] + current_direction[0], 
                       current_pos[1] + current_direction[1]]
            
            if self.is_valid_position(next_pos):
                vehicle['pos'] = next_pos
                return
        
        # Choose a random valid neighbor
        new_pos, new_direction = random.choice(neighbors)
        vehicle['pos'] = new_pos
        vehicle['direction'] = new_direction
    
    def move_vehicles(self):
        """Move each vehicle and remove if out of steps"""
        vehicles_to_remove = []
        
        for vehicle_id, vehicle in self.vehicles.items():
            vehicle['steps_left'] -= 1
            
            if vehicle['steps_left'] <= 0:
                vehicles_to_remove.append(vehicle_id)
            else:
                self.move_vehicle(vehicle_id)
        
        # Remove vehicles that are out of steps
        for vehicle_id in vehicles_to_remove:
            del self.vehicles[vehicle_id]
    
    def check_collision(self):
        agent_tuple = tuple(self.agent_pos)
        for vehicle in self.vehicles.values():
            if tuple(vehicle['pos']) == agent_tuple:
                return True
        return False
    
    def step(self, action):
        """
        Take an action in the environment.
        action: 0: stay, 1: up, 2: down, 3: left, 4: right
        """
        self.steps += 1
        # Previous agent position (for ghost collision :( )
        old_agent_pos = tuple(self.agent_pos)
        
        # Agent moves
        directions = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]
        dy, dx = directions[action]
        new_agent_pos = [self.agent_pos[0] + dy, self.agent_pos[1] + dx]
        
        old_vehicle_positions = {vid: tuple(v['pos']) for vid, v in self.vehicles.items()}
        # Spawn and move vehicles
        self.spawn_vehicle()
        self.move_vehicles()
        
        for vid, vehicle in self.vehicles.items():
            new_vehicle_pos = tuple(vehicle['pos'])
            old_vehicle_pos = old_vehicle_positions.get(vid)
            
            if old_vehicle_pos is not None:
                # Check ghost collision
                if (tuple(new_agent_pos) == old_vehicle_pos and 
                    new_vehicle_pos == old_agent_pos):
                    self.done = True
                    self.crashed = True
                    return self.get_state(), -100, True, {'crashed': True, 'steps': self.steps, 'collision_type': 'ghost'}
        
        # Move agent
        if self.is_valid_position(new_agent_pos):
            # check if new position has vehicle
            if tuple(new_agent_pos) in [tuple(v['pos']) for v in self.vehicles.values()]:
                # Collided
                self.done = True
                self.crashed = True
                return self.get_state(), -100, True, {'crashed': True, 'steps': self.steps, 'collision_type': 'direct'}
            
            self.agent_pos = new_agent_pos
        
        # Check collision after all moves
        if self.check_collision():
            self.done = True
            self.crashed = True
            return self.get_state(), -100, True, {'crashed': True, 'steps': self.steps, 'collision_type': 'vehicle_into_agent'}
        
        # Check if reached goal
        if tuple(self.agent_pos) in [tuple(g) for g in self.goal_positions]:
            self.done = True
            return self.get_state(), 100, True, {'crashed': False, 'steps': self.steps}
        
        reward = -1
        
        return self.get_state(), reward, False, {'crashed': False, 'steps': self.steps}



class SimpleAgent:
    
    def __init__(self, env):
        self.env = env
        self.path = []
        self.replan_counter = 0  # Counter for replans
        
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_danger_map(self):
        """
        Create a danger map based on vehicle positions
        Cells occupied by vehicles are very dangerous
        """
        danger_map = np.zeros((self.env.height, self.env.width))
        
        for vehicle in self.env.vehicles.values():
            vy, vx = vehicle['pos']
            
            # Note vehicles' position (very dangerous)
            if 0 <= vy < self.env.height and 0 <= vx < self.env.width:
                danger_map[vy, vx] += 50
            
            # Note adj cells (moderate dangerous)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = vy + dy, vx + dx
                    if 0 <= ny < self.env.height and 0 <= nx < self.env.width:
                        danger_map[ny, nx] += 10
        
        return danger_map
    
    def find_path_astar(self):
        """
        Find path using A* algorithm considering danger map
        """
        import heapq
        
        start = tuple(self.env.agent_pos)
        goals = [tuple(g) for g in self.env.goal_positions]
        
        # Get danger map
        danger_map = self.get_danger_map()
        
        # Priority queue: (f_score, counter, current_pos, path)
        # counter to avoid tie-break (f_score) 
        counter = 0
        open_set = []
        heapq.heappush(open_set, (0, counter, start, []))
        counter += 1
        
        # g_score: cost from start to current
        g_score = {start: 0}
        
        # List of closed positions
        closed_set = set()
        
        while open_set:
            _, _, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Check if reached goal
            if current in goals:
                return path
            
            # Explore neighbors
            neighbors = self.env.get_valid_neighbors(list(current))
            for next_pos, direction in neighbors:
                next_tuple = tuple(next_pos)
                
                if next_tuple in closed_set:
                    continue
                
                # Cost to next_pos
                # Base cost = 1 + danger cost
                danger_cost = danger_map[next_tuple[0], next_tuple[1]]
                tentative_g = g_score[current] + 1 + danger_cost * 0.1
                
                # If this path to neighbor is better
                if next_tuple not in g_score or tentative_g < g_score[next_tuple]:
                    g_score[next_tuple] = tentative_g
                    
                    # heuristic: min distance to any goal
                    h_score = min(self.manhattan_distance(next_tuple, goal) for goal in goals)
                    f_score = tentative_g + h_score
                    
                    # Add to open set
                    action = self.direction_to_action(direction)
                    new_path = path + [action]
                    
                    heapq.heappush(open_set, (f_score, counter, next_tuple, new_path))
                    counter += 1
        
        return []  # No path found
    
    def direction_to_action(self, direction):
        """Direction -> action"""
        directions = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]
        return directions.index(direction)
    
    def should_replan(self):
        """
        Decide whether to replan based on nearby vehicles
        Replan if any vehicle is within 3 cells
        """
        agent_pos = tuple(self.env.agent_pos)
        
        for vehicle in self.env.vehicles.values():
            vehicle_pos = tuple(vehicle['pos'])
            distance = self.manhattan_distance(agent_pos, vehicle_pos)
            
            if distance <= 3:
                return True
        
        return False
    
    def get_action(self):
        """
        Get next action for the agent
        Replan path if necessary
        """
        # Calculate new path or if replan needed
        if not self.path or (len(self.path) > 0 and self.should_replan()):
            self.path = self.find_path_astar()
            self.replan_counter += 1
        
        if self.path:
            return self.path.pop(0)
        
        # If no path found
        self.path = self.find_path_astar()
        if self.path:
            return self.path.pop(0)
        
        return 0  # Stop if really no path :(


def run_experiment(map_file, spawn_probabilities, num_episodes=50):
    """
    Run exp with different spawn rates
        
    Returns:
        results: Dictionary contains of {spawn_prob: {'avg_steps': x, 'success_rate': y}}
    """
    results = {}
    
    for spawn_prob in spawn_probabilities:
        print(f"\nExperiencing with spawn rate = {spawn_prob}")
        env = CityEnvironment(map_file, spawn_probability=spawn_prob)
        
        steps_list = []
        success_count = 0
        total_replans = 0  # Replans counter
        
        for episode in range(num_episodes):
            env.reset(start_idx=episode)  # Change of starting node
            agent = SimpleAgent(env)
            
            done = False
            max_steps = 1000
            
            while not done and env.steps < max_steps:
                action = agent.get_action()
                state, reward, done, info = env.step(action)
                
                if done:
                    if not info['crashed']:
                        steps_list.append(info['steps'])
                        success_count += 1
                        total_replans += agent.replan_counter
                    break
        
        avg_steps = np.mean(steps_list) if steps_list else max_steps
        avg_replans = total_replans / success_count if success_count > 0 else 0
        success_rate = success_count / num_episodes
        
        results[spawn_prob] = {
            'avg_steps': avg_steps,
            'success_rate': success_rate,
            'total_success': success_count,
            'avg_replans': avg_replans
        }
        
        print(f"  Average steps: {avg_steps:.2f}")
        print(f"  Average replans: {avg_replans:.2f}")
        print(f"  Success rate: {success_rate:.2%}")
    
    return results


def visualize_results(results):
    spawn_probs = sorted(results.keys())
    avg_steps = [results[p]['avg_steps'] for p in spawn_probs]
    success_rates = [results[p]['success_rate'] for p in spawn_probs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Average steps
    ax1.plot(spawn_probs, avg_steps, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Spawn rate', fontsize=12)
    ax1.set_ylabel('Avarage steps', fontsize=12)
    ax1.set_title('Impact of traffic density on delivery time', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Success rate
    ax2.plot(spawn_probs, success_rates, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Spawn rate', fontsize=12)
    ax2.set_ylabel('Success rate', fontsize=12)
    ax2.set_title('Delivery success rate by traffic density', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_live_animation(map_file, spawn_probability=0.02, max_steps_per_episode=300, 
                         save_to_file=False, num_episodes_to_save=5):
    env = CityEnvironment(map_file, spawn_probability=spawn_probability)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    colors = ['#FFFFFF', '#2C3E50', '#2ECC71', '#F39C12', '#E74C3C', '#3498DB', '#E67E22']
    cmap = ListedColormap(colors)
    
    im = ax.imshow(env.get_state(), cmap=cmap, vmin=0, vmax=6, interpolation='nearest')
    
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc='#FFFFFF', ec='black', label='Path'),
        plt.Rectangle((0,0),1,1, fc='#2C3E50', label='Wall'),
        plt.Rectangle((0,0),1,1, fc='#2ECC71', label='Start point'),
        plt.Rectangle((0,0),1,1, fc='#F39C12', label='Cars spawn point'),
        plt.Rectangle((0,0),1,1, fc='#E74C3C', label='Goal'),
        plt.Rectangle((0,0),1,1, fc='#3498DB', label='Agent'),
        plt.Rectangle((0,0),1,1, fc='#E67E22', label='Cars'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    title_text = ax.set_title('Initializing...', fontsize=14, pad=20)
    ax.axis('off')
    
    episode_num = [0] 
    agent = [None]
    step_in_episode = [0]
    
    def init_episode():
        """Initialize a new episode"""
        episode_num[0] += 1
        start_idx = random.randint(0, len(env.start_positions) - 1)
        env.reset(start_idx=start_idx)
        agent[0] = SimpleAgent(env)
        step_in_episode[0] = 0
        print(f"\n{'='*50}")
        print(f"EPISODE #{episode_num[0]} START")
        print(f"{'='*50}")
    
    def update(frame):
        """Update function for each frame"""
        
        # If episode ended, start a new one
        if agent[0] is None or env.done or step_in_episode[0] >= max_steps_per_episode:
            if agent[0] is not None:
                # Old episode ended
                result = "COLLIDED ❌" if env.crashed else "SUCCESS ✓"
                print(f"Episode #{episode_num[0]} ends: {result} after {env.steps} steps")
                if frame % 10 == 0:  
                    init_episode()
            else:
                init_episode()
        
        # 1 step
        if not env.done and step_in_episode[0] < max_steps_per_episode:
            action = agent[0].get_action()
            state, reward, done, info = env.step(action)
            step_in_episode[0] += 1
        
        current_state = env.get_state()
        im.set_array(current_state)
        
        # Count vehicles
        num_vehicles = len(env.vehicles)
        
        # title
        status = ""
        if env.done:
            if env.crashed:
                status = " | ❌ Collided!"
            else:
                status = " | ✓ GOAL!"
        
        title_text.set_text(
            f'Episode #{episode_num[0]} | Step {env.steps} | '
            f'Vehicle: {num_vehicles} | Spawn prob: {spawn_probability}{status}'
        )
        
        return [im, title_text]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, 
        interval=100,  
        blit=True, 
        cache_frame_data=False  # No caching to create new episode each time
    )
    
    plt.tight_layout()
    plt.show()
    
    return anim



if __name__ == "__main__":
    map_file = 'citymap.txt'
    
    print("=" * 60)
    print("DELIVERY AGENT SIMULATION WITH A* ALGORITHM")
    print("=" * 60)
    
    # 1. Run experiments with different spawn rates
    print("\n[1] Run experiments with A* and avoid dynamic objects...")
    spawn_probabilities = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    results = run_experiment(map_file, spawn_probabilities, num_episodes=30)
    
    # 2. Plot
    print("\n[2] Graphing the results...")
    visualize_results(results)
    
    # 3. Create animation
    print("\n[4] Run live animation...")
    create_live_animation(map_file, spawn_probability=0.04, max_steps_per_episode=300)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
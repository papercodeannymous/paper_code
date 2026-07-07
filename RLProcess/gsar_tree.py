"""
GSAR-Tree: 
This module implements an adaptive R-Tree index structure where node selection
during insertion is optimized using a Proximal Policy Optimization (PPO) based
reinforcement learning agent. 

The system aims to minimize node access during
query operations by learning optimal insertion strategies.
"""

import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from collections import namedtuple
from myrtree import RTree
import argparse

# Set random seeds for reproducibility across different runs
random.seed(3)
# Directory for storing trained models
special_dir = "model/NORMAL/"

# Dataset size configurations
DATASIZE_TYPE = {
    "TW1": 100000, "TW2": 200000, "TW5": 500000,  # Synthetic datasets
    "HW1": 1000000,  # Real-world dataset
}

class dotdict(dict):
    """Dictionary with dot-notation access for configuration parameters"""
    def __getattr__(self, name):
        return self[name]

class PolicyNet(nn.Module):
    """
    Policy Network for the PPO agent.
    Maps state features to action probabilities using a softmax output.
    """
    def __init__(self, n_features, n_hidden, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)
    
    def forward(self, x):
        fc1_out = self.fc1(x)
        fc1_out_activation = F.relu(fc1_out)
        fc2_out = self.fc2(fc1_out_activation)
        out = F.softmax(fc2_out, dim=-1)  # Action probabilities
        return out

class ValueNet(nn.Module):
    """
    Value Network for the PPO agent.
    Estimates state value for advantage calculation in PPO updates.
    """
    def __init__(self, n_features, n_hidden, n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        fc1_out = self.fc1(x)
        fc1_out_activation = F.relu(fc1_out)
        fc2_out = self.fc2(fc1_out_activation)
        return fc2_out  # State value estimate

class ACPPO1:
    """
    Actor-Critic with Proximal Policy Optimization (PPO) agent.
    Combines policy and value networks with experience replay and PPO clipping.
    """
    def __init__(self, n_features, n_hidden, n_actions, lr_a, lr_c, lmbda, epochs, eps, gamma, device, buffer_size, epsilon):
        # Network architecture parameters
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.value_net_output = 1

        # Learning parameters
        self.lr_a = lr_a  # Actor learning rate
        self.lr_c = lr_c  # Critic learning rate
        self.gamma = gamma  # Discount factor
        self.lmbda = lmbda  # GAE parameter
        self.epochs = epochs  # PPO update epochs
        self.eps = eps  # PPO clipping parameter
        self.device = device

        # Experience replay and exploration
        self.buffer = []  # Transition buffer
        self.counter = 0  # Buffer counter
        self.buffer_size = buffer_size  # Maximum buffer size
        self.epsilon = epsilon  # Exploration rate
        self.step = 0  # Step counter
        self.exploit_thresh = 0  # Exploitation threshold

        # Temporary storage for current episode
        self.state_memory = []
        self.action_memory = []

    def build_net(self, optimizer_func):
        """Initialize policy and value networks with specified optimizer"""
        self.actor = PolicyNet(self.n_features, self.n_hidden, self.n_actions).to(self.device)
        self.critic = ValueNet(self.n_features, self.n_hidden, self.value_net_output).to(self.device)

        self.actor_optimizer = optimizer_func(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = optimizer_func(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, state, sp_flag=False):
        """
        Select action using current policy with epsilon-greedy exploration.
        
        Args:
            state: Current state representation from R-Tree
            sp_flag: Special flag for self-play mode action space
        
        Returns:
            action: Selected action index
            log_prob: Log probability of selected action for PPO updates
        """
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(self.device)
        try:
            action_prob = self.actor(state)
            if torch.any(torch.isnan(action_prob)) or torch.any(action_prob <= 0):
                raise ValueError("Invalid action probabilities")
            
            dist = torch.distributions.Categorical(action_prob)
            # Epsilon-greedy exploration strategy
            if random.random() < 1 - self.epsilon:
                action = dist.probs.argmax().item()  # Exploit: choose best action
            else:
                # Explore: random action with adjusted action space
                if sp_flag:
                    action = random.randint(0, 4)   
                else:
                    action = random.randint(0, self.n_actions - 1)   

            log_prob = dist.log_prob(torch.tensor(action, dtype=torch.int64).to(self.device))
            self.step += 1
        except (ValueError, RuntimeError) as e:
            print(f"Exception occurred in choose_action: {e}")
            print("EXCEPTION!!!!!!!")
            exit()
        return action, log_prob
    
    def store_transition(self, transiton):
        """Store transition in replay buffer with size limitation"""
        self.buffer.append(transiton)
        self.counter += 1
        # Remove oldest transitions if buffer exceeds capacity
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def ppo_learn(self):
        """
        Perform PPO update using stored transitions.
        Includes advantage calculation with GAE and clipped objective function.
        """
        # Extract batch data from buffer
        states = torch.tensor(np.array([trans.state for trans in self.buffer]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([trans.action for trans in self.buffer]), dtype=torch.long).to(self.device).view(-1, 1)
        rewards = torch.tensor(np.array([trans.reward for trans in self.buffer]), dtype=torch.float32).to(self.device).view(-1, 1)
        next_states = torch.tensor(np.array([trans.next_state for trans in self.buffer]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array([trans.dones for trans in self.buffer]), dtype=torch.float).to(self.device).view(-1, 1)

        # Calculate TD targets and advantages
        next_q_target = self.critic(next_states).to(self.device)  # V(s_{t+1})
        td_target = rewards + self.gamma * next_q_target * (1 - dones)  # R + γV(s_{t+1})
        td_value = self.critic(states).to(self.device)  # V(s_t)
        td_delta = td_target - td_value  # TD error: δ = target - prediction

        # Convert to numpy for advantage calculation
        td_delta = td_delta.cpu().detach().numpy()

        # Calculate advantages using Generalized Advantage Estimation (GAE)
        advantage = 0
        advantage_list = []
        for delta in td_delta[::-1]:  # Reverse for backward calculation
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage = torch.tensor(np.array(advantage_list), dtype=torch.float32).to(self.device)

        # Get old action probabilities for importance sampling
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # Multiple epochs of PPO updates for sample efficiency
        for _ in range(self.epochs):
            # Calculate new action probabilities
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)  # Importance weight: π_new/π_old
            
            # PPO clipped objective
            clip_item1 = ratio * advantage
            clip_item2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            actor_loss = -torch.mean(torch.min(clip_item1, clip_item2))  # Maximize expected advantage
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))  # Fit value function

            # Update actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update critic network  
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        # Clear buffer after learning (on-policy)
        if len(self.buffer) >= self.buffer_size:
            del self.buffer[:]

    def save_checkpoint(self, args):
        """
        Save model checkpoint with descriptive filename based on experiment parameters.
        
        Args:
            args: Configuration dictionary containing experiment parameters
        """
        filepath = special_dir + args.operation + "_" + args.train_volume + "_" + args.data_distribution + "_" + \
            args.reference_tree_type + "_" + str(args.max_entry) + "_" + str(args.action_space_size) + "_BestModel.pth"
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, args):
        """Load model checkpoint from file"""
        filepath = special_dir + args.operation + "_" + args.train_volume + "_" + args.data_distribution + "_" + \
            args.reference_tree_type + "_" + str(args.max_entry) + "_" + str(args.action_space_size) + "_BestModel.pth"
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(filepath, map_location=map_location)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print("Loaded checkpoint from {}".format(filepath))

def query_test(query_ratio, tree, query_type="RRQ"):
    """
    Test query performance for a given query area ratio.
    
    Args:
        query_ratio: Percentage of total area to query (0-100)
        tree: R-Tree instance to test
        query_type: Query type ("RRQ" for Random Range Query)
    
    Returns:
        [description, average_node_access]: Test results
        average_node_access: Average node accesses per query
    """
    x_min, x_max, y_min, y_max = 0, 100000, 0, 100000  # Fixed data space
    tree_acc_no = 0
    # Calculate query area based on ratio
    testing_query_area = query_ratio / 100 * ((x_max - x_min) * (y_max - y_min))    
    side = (testing_query_area**0.5) / 2  # Square query side length
    
    k = 0
    while k < 1000:  # 1000 queries for statistical significance
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if query_type == "RRQ":
            # Ensure query rectangle stays within bounds
            if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:
                tree_access = tree.Query((x - side, y - side, x + side, y + side))
                tree_acc_no += tree_access
                k = k + 1
    if query_type == "RRQ":
        return [f"{query_ratio}% query", tree_acc_no / 1000], tree_acc_no / 1000
    
def query_test_ALL(query_ratio, tree, reference_tree, query_type="RRQ"):
    """
    Comprehensive query test comparing GSAR-tree with reference R*-tree.
    Measures both node accesses and query execution time.
    
    Returns:
        dict: Query performance comparison results
    """
    x_min, x_max, y_min, y_max = 0, 100000, 0, 100000
    tree_acc_no = 0
    ref_tree_acc_no = 0

    tree_acc_time = 0
    ref_tree_acc_time = 0
    testing_query_area = query_ratio / 100 * ((x_max - x_min) * (y_max - y_min))    
    side = (testing_query_area**0.5) / 2
    k = 0
    while k < 1000:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if query_type == "RRQ":
            if x - side > x_min and y - side > y_min and x + side < x_max and y + side < y_max:
                # Test GSAR-tree
                tree_acc_start_time = datetime.now()
                tree_access = tree.Query((x - side, y - side, x + side, y + side))
                tree_acc_time += (datetime.now() - tree_acc_start_time).total_seconds()

                # Test reference R*-tree
                ref_acc_start_time = datetime.now()
                reference_tree_access = reference_tree.Query((x - side, y - side, x + side, y + side))
                ref_tree_acc_time += (datetime.now() - ref_acc_start_time).total_seconds()

                tree_acc_no += tree_access
                ref_tree_acc_no += reference_tree_access
                k += 1
                print(f"query in ratio {query_ratio}%: {k}", end='\r')

    print(f"query time in ratio {query_ratio}: GSAR:{tree_acc_time}, rstar:{ref_tree_acc_time}")

    return {
        "query_ratio": query_ratio,
        "tree_acc": tree_acc_no / 1000,  # Average node accesses for GSAR
        "ref_acc": ref_tree_acc_no / 1000  # Average node accesses for R*-tree
    }

def plot_query_results(test_result, title="Query Performance Comparison"):
    """
    Visualize query performance comparison between GSAR and R*-tree.
    
    Args:
        test_result: List of query test results from query_test_ALL
        title: Plot title
    """
    ratios = [res["query_ratio"] for res in test_result]
    tree_accs = [res["tree_acc"] for res in test_result]  # GSAR results
    ref_accs = [res["ref_acc"] for res in test_result]    # R*-tree results

    x = np.arange(len(ratios))
    width = 0.4  
    plt.figure(figsize=(10,6))    
    plt.bar(x, ref_accs, width=width/2, label='R*-Tree', color='orange', alpha=0.7)
    plt.bar(x, tree_accs, width=width/2, label='GSAR', color='skyblue')

    # Add trend lines
    plt.plot(x, ref_accs, 'o--', color='red', label='R*-Tree (line)')
    plt.plot(x, tree_accs, 'o--', color='blue', label='GSAR (line)')
    plt.xticks(x, ratios)
    plt.xlabel("Query ratio (%)")
    plt.ylabel("Average Node Access")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Main execution block for GSAR-tree training and evaluation.
    Supports three modes: training, self-play, and testing.
    """
    
    # Feature configuration for state representation
    FEATURES_TYPE = {125: 4}  # Feature type to dimension mapping
    
    # Set seeds for reproducibility
    np.random.seed(1)
    torch.manual_seed(1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Command line argument parsing
    train_or_test = {1: "training", 2:"testing"}
    parser = argparse.ArgumentParser(description="Select process mode for R-Tree experiment")
    parser.add_argument('--mode', type=int, choices=[1,2], default=2)
    parser.add_argument('--welltrained', type=bool, default=True)
    args = parser.parse_args()
    process_label = train_or_test[args.mode]
    directly_sp_args = False

    # Model loading configuration
    if args.welltrained:
        special_dir = "model/welltrained/"
        process_label = train_or_test[2]

    # ========== DATA CONFIGURATION ==========
    data_distribution = "NORMAL"  # Data distribution type
    train_volume = "TW1"          # Training dataset size (100K)
    testing_volume = "TW2"        # Testing dataset size (200K) 
    file_format = ".npy"          # Data file format

    # Spatial domain boundaries
    x_min, x_max, y_min, y_max = 0, 100000, 0, 100000
    data_edge_size = 1
    
    # File paths for training and testing data
    data_filename = "testing_data/" + train_volume + "_" + data_distribution + rf"{file_format}"
    testing_data_filename = "testing_data/testing_" + testing_volume + "_" + data_distribution + rf"{file_format}"
    training_set_size = DATASIZE_TYPE[train_volume]
    testing_set_size = DATASIZE_TYPE[testing_volume]

    # ========== QUERY CONFIGURATION ==========
    query_reward_freq = 10           # Frequency of reward calculation during training
    training_query_area_ratio = 0.05 # Query area ratio for training rewards

    # ========== MODEL SAVING CONFIGURATION ==========
    acppo2percent_q = 0              # Current performance metric
    min_acppo2percent_q = float('inf')  # Best performance metric
    load_bestModel_freq = 5          # Frequency of loading best model

    # ========== TREE CONFIGURATION ==========
    max_entry = 50                   # Maximum entries per node
    min_entry_factor = 0.4           # Minimum entries factor for node underflow
    
    # State and action space configuration
    feature_type = 125
    num_features = FEATURES_TYPE[feature_type]  # State feature dimension
    action_space_size = int(20)                 # Number of possible actions  
    state_space_size = action_space_size * (num_features)  # Total state dimension

    # Reference tree type selection
    REFTREE_TYPE = {1: "rtree", 2: "rstar-tree"}
    reference_tree_type = REFTREE_TYPE[2]  # Use R*-tree as baseline

    # ========== EXPERIMENT CONFIGURATION ==========
    # Training configuration
    if max_entry == 50:
        training_args = dotdict({
            'operation': "train",
            'train_volume': train_volume,
            'data_distribution': data_distribution,
            'reference_tree_type': reference_tree_type,
            'max_entry': max_entry,
            'action_space_size': action_space_size
        })
    else:
        training_args = dotdict({
            'operation': f"3Layer-{max_entry}-train",
            'train_volume': train_volume,
            'data_distribution': data_distribution,
            'reference_tree_type': reference_tree_type,
            'max_entry': max_entry,
            'action_space_size': action_space_size
        })

    # Self-play configurations
    sp_args = dotdict({
        'operation': "self-play",
        'train_volume': train_volume,
        'data_distribution': data_distribution,
        'reference_tree_type': "self",
        'max_entry': max_entry,
        'action_space_size': action_space_size
    })
    dir_sp_args = dotdict({
        'operation': "dir-SP", 
        'train_volume': train_volume,
        'data_distribution': data_distribution,
        'reference_tree_type': "self",
        'max_entry': max_entry,
        'action_space_size': action_space_size
    })
    
    testing_args = training_args  # Use training config for testing

    # ========== HYPERPARAMETERS ==========
    num_episodes = 20       # Training episodes
    sp_num_episodes = 1     # Self-play episodes  
    gamma = 0.98            # Discount factor
    lr_a = 1e-4             # Actor learning rate
    lr_c = 1e-4             # Critic learning rate
    n_hidden = 64           # Hidden layer size
    lmbda=0.98              # GAE parameter
    epochs=10               # PPO epochs
    eps=0.2                 # PPO clipping parameter
    buffer_size = 20        # Replay buffer size
    epsilon = 0.1           # Exploration rate

    # Load datasets
    model_dataset = np.load(data_filename)          # Training data
    testing_dataset = np.load(testing_data_filename)  # Testing data

    # ========== TREE STRATEGY SETUP ==========
    # Initialize GSAR-tree with RL
    acppo_tree = RTree(max_entry, min_entry_factor)
    acppo_tree.SetDefaultInsertStrategy("INS_AREA")
    acppo_tree.SetDefaultSplitStrategy("SPL_MIN_MARGIN")

    # Initialize reference R*-tree for comparison
    refer_tree = RTree(max_entry, min_entry_factor)
    if reference_tree_type == REFTREE_TYPE[1]:
        refer_tree.SetDefaultInsertStrategy("INS_AREA")
        refer_tree.SetDefaultSplitStrategy("SPL_MIN_AREA")
    elif reference_tree_type == REFTREE_TYPE[2]:
        refer_tree.SetDefaultInsertStrategy("INS_RSTAR") 
        refer_tree.SetDefaultSplitStrategy("SPL_MIN_OVERLAP")

    # ========== AGENT INITIALIZATION ==========
    agent = ACPPO1(
        n_features=state_space_size,
        n_hidden=n_hidden,
        n_actions=action_space_size,
        lr_a=lr_a,
        lr_c=lr_c,
        lmbda=lmbda,
        epochs=epochs,
        eps=eps,
        gamma=gamma,
        device=device,
        buffer_size=buffer_size,
        epsilon=epsilon
    )
    
    optim_func = torch.optim.Adam
    agent.build_net(optim_func)

    # Transition storage for experience replay
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'dones'])
    
    # Query ratios for performance evaluation
    queries = [2.0, 1.0, 0.5, 0.05]
    training_query_area = training_query_area_ratio / 100 * ( (x_max - x_min) * (y_max - y_min))      

    return_rewards_list = []
    test_result = []

    # ========== BUILD REFERENCE TREE ==========
    # Construct a fixed R*-tree for baseline comparison
    fixed_normal_rtree = RTree(max_entry, min_entry_factor)

    if reference_tree_type == REFTREE_TYPE[1]:
        # Guttman R-tree configuration
        fixed_normal_rtree.SetDefaultInsertStrategy("INS_AREA")
        fixed_normal_rtree.SetDefaultSplitStrategy("SPL_MIN_AREA")
    else:
        # R*-tree configuration  
        fixed_normal_rtree.SetDefaultInsertStrategy("INS_RSTAR")
        fixed_normal_rtree.SetDefaultSplitStrategy("SPL_MIN_OVERLAP")

    # Build the reference tree with training data
    for i in range(len(model_dataset)):
        insert_obj = model_dataset[i]
        if reference_tree_type == REFTREE_TYPE[1]:
            fixed_normal_rtree.DefaultInsert(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])       
        elif reference_tree_type == REFTREE_TYPE[2]:
            fixed_normal_rtree.DirectInsert(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])           
            fixed_normal_rtree.DirectSplitWithReinsert()

    # ========== MODEL TRAINING ==========
    if process_label == "training":
        print("Starting GSAR-tree training...")
        for i_episode in range(num_episodes):
            # Reset trees for new episode
            acppo_tree.Clear()
            refer_tree.Clear()
            
            rl_cnt = 0      # Count of RL-based decisions
            no_rl_cnt = 0   # Count of heuristic decisions
            episode_rewards = 0
            test_result = []

            # Process each data point in training set
            for i in range(training_set_size):
                insert_obj = model_dataset[i]
                
                # Insert into reference tree (baseline)
                if reference_tree_type == REFTREE_TYPE[1]:
                    refer_tree.DefaultInsert(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])
                elif reference_tree_type == REFTREE_TYPE[2]:
                    refer_tree.DirectInsert(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])
                    refer_tree.DirectSplitWithReinsert()
                
                # Prepare GSAR-tree for insertion
                acppo_tree.PrepareRectangle(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])

                # Traverse tree to find insertion location
                while not acppo_tree.IsLeaf(acppo_tree.node_ptr):
                    if acppo_tree.GetMinAreaContainingChild() is None:
                        # Use RL agent for node selection
                        rl_cnt += 1
                        states = acppo_tree.RetrieveEvaluatedInsertStatesByType(action_space_size, num_features, feature_type)
                        action, log_prob = agent.choose_action(states)
                        agent.state_memory.append(states)
                        agent.action_memory.append(action)
                        acppo_tree.InsertWithEvaluatedLoc(action)
                    else:
                        # Use heuristic for node selection
                        no_rl_cnt += 1
                        insert_loc = acppo_tree.GetMinAreaContainingChild()
                        acppo_tree.InsertWithLoc(insert_loc)
                # Insert into leaf node
                acppo_tree.InsertWithLoc(0)

                # Apply appropriate split strategy
                if reference_tree_type == REFTREE_TYPE[1]:
                    acppo_tree.DefaultSplit()     # Guttman R-tree split
                elif reference_tree_type == REFTREE_TYPE[2]:
                    acppo_tree.DirectSplitWithReinsert()    # R*-tree split

                # Calculate rewards periodically based on query performance
                if (len(agent.state_memory) % query_reward_freq == 0) and len(agent.state_memory) >= query_reward_freq:
                    avg_access_rate = 0
                    # Sample multiple queries for reward calculation
                    for k in range(query_reward_freq):
                        # Generate random query rectangle
                        y_x_ratio = random.uniform(0.1, 1)
                        y_length = (training_query_area * y_x_ratio) ** 0.5
                        x_length = training_query_area / y_length

                        x_center = (model_dataset[i - k][0] + model_dataset[i - k][2]) / 2
                        y_center = (model_dataset[i - k][1] + model_dataset[i - k][3]) / 2
                        query_rec = [x_center - x_length / 2, y_center - y_length / 2, x_center + x_length / 2, y_center + y_length / 2]

                        # Calculate node access rates for comparison
                        refer_tree_query_rate = refer_tree.AccessRate(query_rec)
                        acppo_tree_query_rate = acppo_tree.AccessRate(query_rec)
                        # Reward: reduction in node accesses compared to R*-tree
                        avg_access_rate += (refer_tree_query_rate - acppo_tree_query_rate)
                    
                    # Store transitions for PPO update
                    idx = 0
                    records_num = len(agent.action_memory)
                    for idx in range(records_num):
                        _state = agent.state_memory[idx]
                        _action = agent.action_memory[idx]
                        _reward = avg_access_rate  # Shared reward for all actions in sequence
                        if idx == records_num - 1:
                            _next_state = agent.state_memory[idx]
                            _done = True
                        else:
                            _next_state = agent.state_memory[idx + 1]
                            _done = False
                        trans = Transition(_state, _action, _reward, _next_state, _done)
                        agent.store_transition(trans)
                    
                    # Perform PPO update
                    agent.ppo_learn()
                    # Reset memory for next sequence
                    agent.state_memory = []
                    agent.action_memory = []
                    episode_rewards += records_num * avg_access_rate
                    # Synchronize trees
                    refer_tree.CopyTree(acppo_tree.tree_ptr)
                    print(f"Training Episode: {i_episode}\ttrained_data: {i}", end='\r')

            # Evaluate current model performance
            for qidx, ratio in enumerate(queries):
                if qidx == 0:
                    temp_res, acppo2percent_q = query_test(ratio, acppo_tree)
                else:
                    temp_res, _ = query_test(ratio, acppo_tree)
                test_result.append(temp_res)

            return_rewards_list.append(episode_rewards)
            print(f"qcompare: {test_result}")
            
            # Save best model based on query performance
            if acppo2percent_q < min_acppo2percent_q:
                min_acppo2percent_q = acppo2percent_q
                print("save the best model", acppo2percent_q)
                agent.save_checkpoint(training_args)     
            
            # Periodically reload best model for training stability
            if (i_episode + 1) % 5 == 0:
                agent.load_checkpoint(training_args)
                
        print("\n\nFinished Training\nCan start self-play or testing")
        process_label  = train_or_test[2]

    # ========== SELF-PLAY TRAINING ==========
    if process_label == "self-play":
        print("Starting self-play training...")
        # Initialize competitor agent
        competitor = ACPPO1(
            n_features=state_space_size,
            n_hidden=n_hidden,
            n_actions=action_space_size,
            lr_a=lr_a,
            lr_c=lr_c,
            lmbda=lmbda,
            epochs=epochs,
            eps=eps,
            gamma=gamma,
            device=device,
            buffer_size=buffer_size,
            epsilon=epsilon
        )
        optim_func = torch.optim.Adam
        competitor.build_net(optim_func)

        # Load and initialize models for self-play
        if directly_sp_args is not True:
            agent.load_checkpoint(training_args)
            agent.save_checkpoint(sp_args)
            competitor.load_checkpoint(sp_args)
        else:
            sp_args.operation = "dir-SP"

        # Self-play training loop
        for i_episode in range(sp_num_episodes):
            acppo_tree.Clear()
            refer_tree.Clear()

            episode_rewards = 0
            test_result = []

            for i in range(training_set_size):
                insert_obj = model_dataset[i]
                # Both trees prepare for insertion
                acppo_tree.PrepareRectangle(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])
                refer_tree.PrepareRectangle(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])
                states = acppo_tree.RetrieveEvaluatedInsertStatesByType(action_space_size, num_features, feature_type)
                ref_states = refer_tree.RetrieveEvaluatedInsertStatesByType(action_space_size, num_features, feature_type)

                # GSAR-tree insertion with RL agent
                while states is not None:
                    if acppo_tree.GetMinAreaContainingChild() is None:
                        action, log_prob = agent.choose_action(states)
                        agent.state_memory.append(states)
                        agent.action_memory.append(action)
                        acppo_tree.InsertWithEvaluatedLoc(action)
                    else:
                        insert_loc = acppo_tree.GetMinAreaContainingChild()
                        acppo_tree.InsertWithLoc(insert_loc)
                    states = acppo_tree.RetrieveEvaluatedInsertStatesByType(action_space_size, num_features, feature_type)
                acppo_tree.InsertWithLoc(0)

                # Competitor tree insertion
                while ref_states is not None:
                    if refer_tree.GetMinAreaContainingChild() is None:
                        ref_action, _ = competitor.choose_action(ref_states)
                        competitor.state_memory.append(ref_states)
                        competitor.action_memory.append(ref_action)
                        refer_tree.InsertWithEvaluatedLoc(ref_action)
                    else:
                        ref_insert_loc = refer_tree.GetMinAreaContainingChild()
                        refer_tree.InsertWithLoc(ref_insert_loc)
                    ref_states = refer_tree.RetrieveEvaluatedInsertStatesByType(action_space_size, num_features, feature_type)
                refer_tree.InsertWithLoc(0)

                # Apply splits
                if reference_tree_type == REFTREE_TYPE[1]:
                    acppo_tree.DefaultSplit()     
                    refer_tree.DefaultSplit()
                elif reference_tree_type == REFTREE_TYPE[2]:
                    acppo_tree.DirectSplitWithReinsert()    
                    refer_tree.DirectSplitWithReinsert()
                
                # Reward calculation and learning (similar to training)
                if (len(agent.state_memory) % query_reward_freq == 0) and len(agent.state_memory) >= query_reward_freq:
                    avg_access_rate = 0
                    for k in range(query_reward_freq):
                        y_x_ratio = random.uniform(0.1, 1)
                        y_length = (training_query_area * y_x_ratio) ** 0.5
                        x_length = training_query_area / y_length

                        x_center = (model_dataset[i - k][0] + model_dataset[i - k][2]) / 2
                        y_center = (model_dataset[i - k][1] + model_dataset[i - k][3]) / 2
                        query_rec = [x_center - x_length / 2, y_center - y_length / 2, x_center + x_length / 2, y_center + y_length / 2]

                        refer_tree_query_rate = refer_tree.AccessRate(query_rec)
                        acppo_tree_query_rate = acppo_tree.AccessRate(query_rec)
                        avg_access_rate += (refer_tree_query_rate - acppo_tree_query_rate)
                    
                    idx = 0
                    records_num = len(agent.action_memory)
                    for idx in range(records_num):
                        _state = agent.state_memory[idx]
                        _action = agent.action_memory[idx]
                        _reward = avg_access_rate
                        if idx == records_num - 1:
                            _next_state = agent.state_memory[idx]
                            _done = True
                        else:
                            _next_state = agent.state_memory[idx + 1]
                            _done = False
                        trans = Transition(_state, _action, _reward, _next_state, _done)
                        agent.store_transition(trans)
                    
                    agent.ppo_learn()
                    agent.state_memory = []
                    agent.action_memory = []
                    episode_rewards += records_num * avg_access_rate
                    refer_tree.CopyTree(acppo_tree.tree_ptr)

                    print(f"Self-Play Episode: {i_episode}\ttrained_data: {i}", end='\r')

            # Performance evaluation
            for qidx, ratio in enumerate(queries):
                if qidx == 0:
                    temp_res, acppo2percent_q = query_test(ratio, acppo_tree)
                else:
                    temp_res, _ = query_test(ratio, acppo_tree)
                test_result.append(temp_res)

            print(f"qcompare: {test_result}")
            
            # Model management
            if acppo2percent_q < min_acppo2percent_q:
                min_acppo2percent_q = acppo2percent_q
                print("save the best model", acppo2percent_q)
                agent.save_checkpoint(sp_args)  
                competitor.load_checkpoint(sp_args) 
            else:  
                if (i_episode + 1) % 5 == 0:
                    agent.load_checkpoint(sp_args)
                    
        print("\n\nFinished Self-Playing\nCan start testing")

    # ========== MODEL TESTING ==========
    if process_label == "testing":
        print("Starting GSAR-tree testing...")
        
        # Load trained model
        agent.load_checkpoint(testing_args)
        acppo_tree.Clear()
        refer_tree.Clear()
        rl_cnt = 0
        no_rl_cnt = 0
        test_result = []

        # Measure construction time
        start_time = datetime.now()

        # Build trees with testing data
        for i in range(testing_set_size):
            insert_obj = testing_dataset[i]

            # Build reference R*-tree
            refer_tree.DirectInsert(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])
            refer_tree.DirectSplitWithReinsert()

            # Build GSAR-tree with RL agent
            acppo_tree.PrepareRectangle(insert_obj[0], insert_obj[1], insert_obj[2], insert_obj[3])           
            while not acppo_tree.IsLeaf(acppo_tree.node_ptr):
                if acppo_tree.GetMinAreaContainingChild() is None:
                    rl_cnt += 1
                    states = acppo_tree.RetrieveEvaluatedInsertStatesByType(action_space_size, num_features, feature_type)
                    action, log_prob = agent.choose_action(states)
                    agent.state_memory.append(states)
                    agent.action_memory.append(action)
                    acppo_tree.InsertWithEvaluatedLoc(action)
                else:
                    no_rl_cnt += 1
                    insert_loc = acppo_tree.GetMinAreaContainingChild()
                    acppo_tree.InsertWithLoc(insert_loc)
            acppo_tree.InsertWithLoc(0)

            # Apply appropriate split strategy
            if reference_tree_type == REFTREE_TYPE[1]:
                acppo_tree.DefaultSplit()     # Guttman R-tree split
            elif reference_tree_type == REFTREE_TYPE[2]:
                acppo_tree.DirectSplitWithReinsert()    # R*-tree split

            print(f"Reconstructing the tree: inserting data {i}", end='\r')

        # Performance metrics
        constructing_time = datetime.now() - start_time
        print("Reconsructing time is: ", constructing_time)
        print("Tree height: ", acppo_tree.GetTreeHeight())

        # Comprehensive query performance evaluation
        print("Query comparison")
        for qidx, ratio in enumerate(queries):
            temp_res = query_test_ALL(ratio, acppo_tree, refer_tree)
            test_result.append(temp_res)

        # Visualize results
        plot_query_results(test_result)

        # Final results summary
        print("\nconstruct tree and get final test result:")
        print(f"qcompare: {test_result}")
        
    print(f"==================================================\nFINISH\n==================================================")
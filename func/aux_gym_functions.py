from base64 import b64encode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder


def get_mean_last_5_percent(array):
    last_5_percent = array[-round(len(array) * 0.05) :]
    return np.mean(last_5_percent)


def plot_image(plot, render, size):
    plot.figure(figsize=(14, 5))
    plot.xticks(range(size))
    plot.yticks(range(size))
    plot.imshow(render, extent=[0, size, size, 0])
    plot.show()


def render_mp4(videopath: str) -> str:
    """
    Gets a string containing a b4-encoded version of the MP4 video at the specified path.
    """
    mp4 = open(videopath, "rb").read()
    base64_encoded_mp4 = b64encode(mp4).decode()
    return (
        f'<video width=400 controls><source src="data:video/mp4;'
        f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'
    )


class ReplayBuffer:
    """
    Experience Replay Buffer

    We will use Experience Replay in order to store the agent's experiences in the form (s_t, a_t, r_t, s_{t+1}):
    - state s_t in which the agent was in
    - action a_t the agent selected in that state
    - reward r_t received for taking the action in the state
    - next state s_{t+1} that the agent arrived to after action

    After collecting a number N of these tuples (s_t, a_t, r_t, s_{t+1}), we randomly pass the data to train network.
    """

    def __init__(self, max_length, observation_space_n):
        self.index, self.size, self.max_length = 0, 0, max_length

        self.states = np.zeros((max_length, observation_space_n), dtype=np.float32)
        self.actions = np.zeros((max_length), dtype=np.uint8)
        self.rewards = np.zeros((max_length), dtype=np.float32)
        self.next_states = np.zeros((max_length, observation_space_n), dtype=np.float32)
        self.dones = np.zeros((max_length), dtype=np.uint8)

    def __len__(self):
        return self.size

    def update(self, state, action, reward, next_state, is_terminal):
        # Update the Replay Buffer
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = is_terminal

        self.index = (self.index + 1) % self.max_length
        if self.size < self.max_length:
            self.size += 1

    def sample(self, batch_size):
        # Pick indexes randomly from the Replay Buffer
        idxs = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )


class LinearNetwork(nn.Module):
    """
    Linear neural network for Deep Q-learning
    """

    def __init__(self, observation_space_n, action_space_n):
        super(LinearNetwork, self).__init__()

        self.layers = nn.Sequential(
        nn.Linear(observation_space_n, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, action_space_n),
        )
    """  
        self.layers = nn.Sequential(
                nn.Linear(observation_space_n, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, action_space_n),
            )
    """

    # Called with either one element to determine next action, or a batch during optimization
    # Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state):
        return self.layers(state)


class DeepQLearningAgent:

    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        batch_size: int,
        max_memory: int,
        tau: float,
        observation_space: np.ndarray,
        action_space: int,
        is_double_network: bool
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
            batch_size: The sample size for batch
            max_memory: The max number of iterations to be stored in memory buffer
            tau: The influence of local network on the target network
            observation_space: The observation space
            action_space: The action space
            is_double_network: To use Deep Q-learning or Double Deep Q-learning algorithm
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.tau = tau
        self.observation_space = observation_space
        self.action_space = action_space
        self.is_double_network = is_double_network
        self.training_error = []

        # Check that there is a GPU avaiable
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize experience replay buffer
        self.memory = ReplayBuffer(self.max_memory, self.observation_space.shape[0])

        # Input and output size based on the Env
        # These lines establish the feed-forward part of the netwotk used to choose actions
        self.policy_dqn = LinearNetwork(self.observation_space.shape[0], self.action_space.n).to(self.device)

        if(self.is_double_network):
            self.target_dqn = LinearNetwork(self.observation_space.shape[0], self.action_space.n).to(self.device)
            self.target_dqn.eval()
            # Copy policy model parameters to target model parameters
            for target_param, policy_param in zip(self.policy_dqn.parameters(),self.target_dqn.parameters()):
                target_param.data.copy_(policy_param)
            print("Double Deep Q-learning agent started with PyTorch")
        else:
            print("Deep Q-learning agent started with PyTorch")

        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate)

    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = self.policy_dqn.forward(state).argmax(dim=-1)
            action = action.cpu().numpy()
            return action

    def choose_action(self, state):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return self.act(state)

    def remember(self, state, action, reward, new_state, is_terminal):
        self.memory.update(state, action, reward, new_state, is_terminal)

    def update(self):
        # If we have less experience than batch size, we don't start training
        if self.batch_size * 10 > self.memory.size:
            return

        # A sample of the training experiences
        (states, actions, rewards, next_states, dones) = self.memory.sample(self.batch_size)

        # Turning the experiences into tensors
        states = torch.as_tensor(states).to(self.device)
        actions = torch.as_tensor(actions).to(self.device).unsqueeze(-1)
        rewards = torch.as_tensor(rewards).to(self.device).unsqueeze(-1)
        next_states = torch.as_tensor(next_states).to(self.device)
        dones = torch.as_tensor(dones).to(self.device).unsqueeze(-1)

        # Eval states
        Q1 = self.policy_dqn.forward(states).gather(-1, actions.long())

        with torch.no_grad():
            # Eval next states
            if(self.is_double_network):
                Q2 = self.target_dqn.forward(next_states).max(dim=-1, keepdim=True)[0]
            else:
                Q2 = self.policy_dqn.forward(next_states).max(dim=-1, keepdim=True)[0]

            target = (rewards + (1 - dones) * self.discount_factor * Q2).to(self.device)

        temporal_difference_loss = F.mse_loss(Q1, target)

        self.training_error.append(temporal_difference_loss.item())

        # Train the network using target and the predicted q_network values
        self.optimizer.zero_grad()
        temporal_difference_loss.backward()
        self.optimizer.step()

        if(self.is_double_network):
            # Update the target network
            for target_param, policy_param in zip(self.target_dqn.parameters(), self.policy_dqn.parameters()):
                target_param.data.copy_(self.tau * policy_param + (1 - self.tau) * target_param)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def test_accurracy(env, agent, num_steps, num_episodes=100):
    counter = 0
    nb_success = 0.0

    while counter < num_episodes:
        is_terminal = False
        state, info = env.reset()

        for i in range(num_steps):
            action = agent.act(state)

            state, reward, terminated, truncated, info = env.step(action)
            is_terminal = terminated or truncated
            nb_success += reward

            if is_terminal:
                break

        counter += 1

    return nb_success / num_episodes


def record_trained_video(env, agent, video_file, num_steps):
    video = VideoRecorder(env, video_file)
    # returns an initial observation
    state, info = env.reset()
    env.render()
    video.capture_frame()

    for i in range(num_steps):
        action = agent.act(state)

        state, reward, terminated, truncated, info = env.step(action)
        is_terminal = terminated or truncated

        env.render()
        video.capture_frame()

        if is_terminal:
            print(
                "step",
                i + 1,
                ":",
                action,
                ",",
                state,
                ",",
                reward,
                ",",
                terminated,
                ",",
                truncated,
                ",",
                info,
            )
            break
    video.close()

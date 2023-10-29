__author__ = "John Statheros (1828326)"
__maintainer__ = "Timothy Reeder (455840)"
__version__ = '0.1.0'
__status__ = 'Development'

import numpy as np


class ReplayBuffer:
    """
    A replay buffer stores the agent's experience and samples transitions from
    it for training.

    Args:
        max_replay_buffer_len: The maximum length of the replay buffer.
        priority_default: The starting priority for each transition.
        alpha: A hyperparameter that controls the importance sampling factor.
        beta: A hyperparameter that controls the prioritization exponent.
        phi: A hyperparameter that controls the probability of selecting a
            transition from the replay buffer based on its priority.
        c: A hyperparameter used to avoid zeros in the denominator when
            calculating the importance sampling weights.
    """
    def __init__(self,
                 max_replay_buffer_len: int,
                 priority_default: float,
                 alpha: float,
                 beta: float,
                 phi: float,
                 c: float) -> None:
        self.max_replay_buffer_len = max_replay_buffer_len + 1
        self.priority_default = priority_default

        # The buffer of transitions.
        self.buffer = []
        # The number of visits to each transition.
        self.n_visits = []
        # The loss computed on the dynamics model for this transition.
        self.model_loss = []
        # The priority for retraining on this transition.
        self.priority = []

        # Curious-adversarial parameters
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.c = c
        
    def add_to_buffer(self,
                      state: np.ndarray,
                      action: int,
                      reward: float,
                      state_next:
                      np.ndarray,
                      done: float) -> None:
        """
        Adds a transition to the replay buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            state_next: The next state.
            done: Whether the episode is done.
        """
        transition = (state, action, reward, state_next, done)
        
        self.buffer.append(transition)
        self.n_visits.append(0)
        self.model_loss.append(0.)
        self.priority.append(self.priority_default)
        
        if len(self.buffer) > self.max_replay_buffer_len:
            self.buffer.pop(0)
            self.n_visits.pop(0)
            self.model_loss.pop(0)
            self.priority.pop(0)
            
    def __encode(self, idxs: np.ndarray) -> tuple[np.ndarray]:
        """
        Function to encode environment
        Args:
            idxs: indexes

        Returns:
            environment in a tuple of arrays
        """
        states, actions, rewards, state_nexts, dones = [], [], [], [], []
        for idx in idxs:
            state, action, reward, state_next, done = self.buffer[idx]
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state_nexts.append(state_next)
            dones.append(done)
            
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.uint8),
            np.array(rewards, dtype=np.float32),
            np.array(state_nexts, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )
        
    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Samples a batch of transitions from the replay buffer.

        Args:
            batch_size: The number of transitions to sample.

        Returns:
            A tuple of NumPy arrays containing the sampled transitions.
        """
        # Normalize the priorities of the transitions.
        norm_priority = np.array(self.priority, dtype=np.float32)
        norm_priority /= np.sum(norm_priority)

        # Sample a batch of indexes from the replay buffer according to the
        # normalized priorities.
        all_idxs = np.arange(0, len(self.buffer))
        idxs = np.random.choice(all_idxs,
                                size=batch_size,
                                p=norm_priority)

        # Encode the sampled transitions.
        transitions = self.__encode(idxs)

        return idxs, transitions
    
    def __calculate_priority(self, idxs: np.ndarray) -> np.ndarray:
        """
        Calculates the priority of the given transitions.

        Args:
            idxs: A NumPy array containing the indexes of the transitions.

        Returns:
            A NumPy array containing the priorities of the transitions.
        """
        # Get the number of visits and model loss for the given transitions.
        idxs = list(idxs)
        n_visits = np.array(self.n_visits, dtype=np.float32)[idxs]
        model_loss = np.abs(np.array(self.model_loss, dtype=np.float32)[idxs])

        # Calculate the priority of the transitions using the
        # curious-adversarial parameters.
        return self.c * np.power(self.beta, n_visits) + \
            np.power(model_loss + self.phi, self.alpha)
    
    def update_priority(self, idxs: np.ndarray, model_losses) -> None:
        """
        Updates the priority of the given transitions.

        Args:
            idxs: A NumPy array containing the indexes of the transitions.
            model_losses: A NumPy array containing the model losses for the
                given transitions.
        """
        for i, idx in enumerate(idxs):
            # Increment the number of visits for the transition.
            self.n_visits[idx] += 1
            # Update the model loss for the transition.
            self.model_loss[idx] = model_losses[i].to("cpu").item()

        # Calculate the new priorities for the transitions.
        new_priorities = self.__calculate_priority(idxs)

        # Update the priorities of the transitions.
        for i, idx in enumerate(idxs):
            self.priority[idx] = new_priorities[i]
        
    def reset_buffer(self) -> None:
        """
        Resets the replay buffer.
        """
        self.buffer = []
        self.n_visits = []
        self.model_loss = []
        self.priority = []

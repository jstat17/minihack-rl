import numpy as np
from typing import Iterable
import matplotlib.pyplot as plt


class ReplayBuffer():
    max_replay_buffer_len: int # maximum replay buffer length
    priority_default: float # the starting priority for each transition
    
    buffer: list[tuple[np.ndarray, int, float, np.ndarray, float]] # the buffer of transitions
    n_visits: list[int] # the number of visits to each transition
    model_loss: list[float] # the loss computed on the dynamics model for this transition
    priority: list[float] # the priority for retraining on this transition
    
    # curious-adversarial parameters
    alpha: float
    beta: float
    phi: float
    c: float
    
    
    def __init__(self, max_replay_buffer_len: int, priority_default: float, alpha: float, beta: float, phi: float, c: float) -> None:
        self.max_replay_buffer_len = max_replay_buffer_len + 1
        self.priority_default = priority_default
        
        self.buffer = []
        self.n_visits = []
        self.model_loss = []
        self.priority = []
        
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.c = c
        
    def add_to_buffer(self, state: np.ndarray, action: int, reward: float, state_next: np.ndarray, done: float) -> None:
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
        norm_priority = np.array(self.priority, dtype=np.float32)
        norm_priority /= np.sum(norm_priority)
        
        all_idxs = np.arange(0, len(self.buffer))
        idxs = np.random.choice(
            all_idxs,
            size = batch_size,
            p = norm_priority
        )
        
        return idxs, self.__encode(idxs)
    
    def __calculate_priority(self, idxs: np.ndarray) -> np.ndarray:
        idxs = list(idxs)
        n_vists = np.array(self.n_visits, dtype=np.float32)[idxs]
        model_loss = np.abs(np.array(self.model_loss, dtype=np.float32)[idxs])
        
        return self.c * np.power(self.beta, n_vists) + np.power(model_loss + self.phi, self.alpha)
    
    def update_priority(self, idxs: np.ndarray, model_losses: Iterable) -> None:
        for i, idx in enumerate(idxs):
            self.n_visits[idx] += 1
            self.model_loss[idx] = model_losses[i].to("cpu").item()
            
        new_priorities = self.__calculate_priority(idxs)
        for i, idx in enumerate(idxs):
            self.priority[idx] = new_priorities[i]
            
        # plt.hist(self.priority)
        # plt.show()
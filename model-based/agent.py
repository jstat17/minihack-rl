import numpy as np
from replay_buffer import ReplayBuffer
from networks import DQN, DeepModel
import torch as th
from torch.optim import Adam
from torch import nn

device = "cuda" if th.cuda.is_available() else "cpu"


class Agent():
    obs_shape: tuple[int] # observation shape
    obs_keys: list[str] # list of keys to index minihack observations
    obs_dtype: np.dtype # dtype of observations
    act_shape: int # number of actions the agent can choose from
    
    batch_size: int # number of samples to draw from replay buffer
    replay_buffer: ReplayBuffer # replay buffer
    
    Q_target: DQN # DQN used for target values, periodically updated
    Q_learning: DQN # DQN that is directly trained
    M: DeepModel # dynamics models
    
    gamma: float # discount factor
    lr_Q: float # learning rate for Q
    lr_M: float # learning rate for M
    lamb: float # trade-off between M's state and reward losses
    
    def __init__(self, obs_shape: tuple[int], obs_keys: list[str], obs_dtype: np.dtype, act_shape: int, batch_size: int,\
                 max_replay_buffer_len: int, priority_default: float, alpha: float, beta: float, phi: float, c: float,\
                 gamma: float, lr_Q: float, lr_M: float, lamb: float) -> None:
        
        self.obs_shape = obs_shape
        self.obs_keys = obs_keys
        self.obs_dtype = obs_dtype
        self.act_shape = act_shape
        
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(
            max_replay_buffer_len = max_replay_buffer_len,
            priority_default = priority_default,
            alpha = alpha,
            beta = beta,
            phi = phi,
            c = c
        )
        
        self.gamma = gamma
        self.lr_Q = lr_Q
        self.lr_M = lr_M
        self.lamb = lamb
        
        # DQNs
        self.Q_target = DQN(
            obs_shape = self.obs_shape,
            act_shape = self.act_shape
        )
        self.Q_target.to(device)
        
        self.Q_learning = DQN(
            obs_shape = self.obs_shape,
            act_shape = self.act_shape
        )
        self.Q_learning.to(device)
        self.update_target_network()
        
        self.Q_criterion = lambda y, Q_state_action: th.mean(th.pow(y - Q_state_action, 2))
        self.Q_optimizer = Adam(self.Q_learning.parameters(), lr=self.lr_Q)
        
        # Dynamics Model
        self.M = DeepModel(
            obs_shape = self.obs_shape,
            act_shape = self.act_shape
        )
        self.M.to(device)
        
        self.M_component_criterion = nn.MSELoss(reduction='none')
        self.M_criterion = lambda L1, L2, lamb=self.lamb: L1 + lamb*L2
        self.M_optimizer = Adam(self.M.parameters(), lr=self.lr_M)
        
    def reshape_state(self, channels: list[np.ndarray]) -> np.ndarray:
        obs = np.zeros(self.obs_shape, dtype=self.obs_dtype)
        assert len(channels) == self.obs_shape[0]
        
        for i, chn in enumerate(channels):
             obs[i] = chn
             
        return obs
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        max_val = 64 #np.iinfo(self.obs_dtype).max
        min_val = np.iinfo(self.obs_dtype).min
        
        normed = np.zeros_like(state.shape, dtype=np.float32)
        normed = state
        
        return (normed - min_val) / (max_val - min_val)
        
    def update_target_network(self) -> None:
        """Update DQN target network from the weights in the learning network
        """
        self.Q_target.load_state_dict(
            self.Q_learning.state_dict().copy()
        )
        
    def optimise_Q_loss(self, state: th.Tensor, action: th.Tensor, reward: th.Tensor, state_next: th.Tensor, done: th.Tensor) -> th.Tensor:
        """Uses minibatch sample to optimise TD-error of Q_learning

        Returns:
            th.tensor: loss for minibatch (scalar)
        """
        
        self.Q_learning.train()
        self.Q_target.eval()
        self.Q_optimizer.zero_grad()
        
        # state = th.tensor(
        #     self.normalize_state(state),
        #     dtype = th.float32
        # ).to(device)
        # reward = th.tensor(reward, dtype=th.float32).to(device)
        # state_next = th.tensor(
        #     self.normalize_state(state_next),
        #     dtype = th.float32
        # ).to(device)
        # done = th.tensor(done, dtype=th.float32).to(device)
        
        Qs_state = self.Q_learning(state)
        with th.no_grad():
            Qs_state_next = self.Q_target(state_next)
        Q_max_state_next = th.max(Qs_state_next, dim=1)
        
        y = (
            reward + (self.gamma * th.multiply(Q_max_state_next.values, (1 - done)))
        ).to(device)
        Q_state_action = (
            Qs_state[np.arange(Qs_state.shape[0]), action.tolist()]
        ).to(device)
        
        loss = (self.Q_criterion(y, Q_state_action)).to(device)
        loss.backward()
        
        self.Q_optimizer.step()
        
        return loss
    
    def optimise_M_loss(self, state: th.Tensor, action: th.Tensor, reward: th.Tensor, state_next: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Uses minibatch sample to optimise M dynamics model

        Returns:
            th.tensor: loss for minibatch (scalar)
        """
        
        self.M.train()
        self.M_optimizer.zero_grad()
        
        
        # state = th.tensor(
        #     self.normalize_state(state),
        #     dtype = th.float32
        # ).to(device)
        # action = th.tensor(action, dtype=th.uint8).to(device)
        # reward = th.tensor(reward, dtype=th.float32).to(device)
        # state_next = th.tensor(
        #     self.normalize_state(state_next),
        #     dtype = th.float32
        # )
        # frame_next = (state_next[:, 0:2, :, :]).to(device)
        
        pred_frame_next, pred_reward = self.M(state, action)
        loss_frame = th.mean(
            self.M_component_criterion(state_next, pred_frame_next),
            dim = (1, 2, 3)
        )
        loss_reward = self.M_component_criterion(reward, pred_reward)
        
        losses = self.M_criterion(loss_frame, loss_reward)
        loss = th.mean(losses)
        loss.backward()
        self.M_optimizer.step()
        
        return th.reshape(losses, (-1,)), loss
    
    def act(self, state: th.Tensor) -> th.Tensor:
        """Gets Q_target's state-action values

        Args:
            state (th.tensor): current state

        Returns:
            Tuple[int, float]: action, action's Q value
        """
        self.Q_target.eval()
        with th.no_grad():
            Q_values = self.Q_target(state)
        
        return Q_values
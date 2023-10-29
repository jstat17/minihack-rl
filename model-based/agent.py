__author__ = "John Statheros (1828326)"
__maintainer__ = "Timothy Reeder (455840)"
__version__ = '0.1.0'
__status__ = 'Development'

import numpy as np
from replay_buffer import ReplayBuffer
from networks import DQN, DeepModel
import torch as th
from torch.optim import Adam
from torch import nn
from nle.nethack import actions

device = "cuda" if th.cuda.is_available() else "cpu"

#
get_inv_code = {
    "potion": 0.1*64,
    "ring": 0.1*64,
    "boots": 0.1*64,
    "horn": 0.25*64,
    "wand": 0.075*64
}
inv_objects_possible = ["potion", "ring", "boots", "horn", "wand"]
pickup_meta_actions = {
    "potion": (actions.Command.PICKUP, actions.Command.QUAFF, -1),
    "ring": (actions.Command.PICKUP, actions.Command.PUTON, -1, ord("r")),
    "boots": (actions.Command.PICKUP, actions.Command.PUTON, -1),
    "horn": (actions.Command.PICKUP,),
    "wand": (actions.Command.PICKUP,)
}
tool_meta_actions = {
    "horn": (actions.Command.APPLY, -1, ord("y"),
             actions.CompassCardinalDirection.E),
    "wand": (actions.Command.ZAP, -1, actions.CompassCardinalDirection.E)
}


class Agent:
    # observation shape
    obs_shape: tuple[int]
    # list of keys to index minihack observations
    obs_keys: list[str]
    # dtype of observations
    obs_dtype: np.dtype
    # number of actions the agent can choose from
    act_shape: int

    # number of samples to draw from replay buffer
    batch_size: int
    # replay buffer
    replay_buffer: ReplayBuffer

    # DQN used for target values, periodically updated
    Q_target: DQN
    # DQN that is directly trained
    Q_learning: DQN
    # dynamics models
    M: DeepModel

    # discount factor
    gamma: float
    # learning rate for Q
    lr_Q: float
    # learning rate for M
    lr_M: float
    # trade-off between M's state and reward losses
    lamb: float

    # agent inventory encoding
    inv: float
    # agent's list of objects in inventory
    inv_objs: list[str]
    # inventory object's activation hotkey
    tool_hotkeys: dict[list, tuple[int]]
    
    def __init__(self,
                 obs_shape: tuple[int],
                 obs_keys: list[str],
                 obs_dtype: np.dtype,
                 act_shape: int,
                 batch_size: int,
                 max_replay_buffer_len: int,
                 priority_default: float,
                 alpha: float,
                 beta: float,
                 phi: float,
                 c: float,
                 gamma: float,
                 lr_Q: float,
                 lr_M: float,
                 lamb: float) -> None:
        """

        Args:
            obs_shape: A tuple of integers representing the shape of the
                observation space.
            obs_keys: A list of strings representing the keys of the
                observations.
            obs_dtype: A NumPy data type representing the data type of the
                observations.
            act_shape: An integer representing the number of actions available
                to the agent.
            batch_size: The batch size to use for training the agent.
            max_replay_buffer_len: The maximum length of the replay buffer.
            priority_default: The default priority for new transitions added to
                the replay buffer.
            alpha: The importance sampling factor.
            beta: The prioritization exponent.
            phi: The probability of selecting a transition from the replay
                buffer based on its priority.
            c: A constant used to avoid zeros in the denominator when
                calculating the importance sampling weights.
            gamma: The discount factor.
            lr_Q: The learning rate for the Q-network.
            lr_M: The learning rate for the dynamics model.
            lamb: The trade-off parameter between the reconstruction loss and
                the prediction loss for the dynamics model.
        """
        
        self.obs_shape = obs_shape
        self.obs_keys = obs_keys
        self.obs_dtype = obs_dtype
        self.act_shape = act_shape
        
        self.batch_size = batch_size

        # Replay buffer. Stores the agent's experience in the form of (state,
        # action, next_state, reward) tuples.
        self.replay_buffer = ReplayBuffer(
            max_replay_buffer_len=max_replay_buffer_len,
            priority_default=priority_default,
            alpha=alpha,
            beta=beta,
            phi=phi,
            c=c
        )

        # Hyperparameters
        self.gamma = gamma
        self.lr_Q = lr_Q
        self.lr_M = lr_M
        self.lamb = lamb
        
        self.inv = 0.
        self.inv_objs = []
        self.tool_hotkeys = dict()

        # Q-network
        # Estimates the Q-values for state-action pairs.
        self.Q_learning = DQN(
            obs_shape=self.obs_shape,
            act_shape=self.act_shape
        )

        # Target Q-network
        # Used to calculate the target Q-values during Q-learning.
        self.Q_target = DQN(
            obs_shape=self.obs_shape,
            act_shape=self.act_shape
        )

        self.Q_target.to(device)
        self.Q_learning.to(device)
        self.update_target_network()
        
        self.Q_criterion = lambda y, Q_state_action: th.mean(th.pow(y - Q_state_action, 2))
        self.Q_optimizer = Adam(self.Q_learning.parameters(), lr=self.lr_Q)

        # Dynamics model
        # Predicts the next state and reward for a given state and action.
        self.M = DeepModel(
            obs_shape=self.obs_shape,
            act_shape=self.act_shape
        )
        self.M.to(device)

        # Dynamics model loss function
        # The loss function is a weighted sum of the reconstruction loss and
        # the prediction loss.
        self.M_component_criterion = nn.MSELoss(reduction='none')
        self.M_criterion = lambda L1, L2, lamb=self.lamb: L1 + lamb*L2

        # Dynamics model optimizer
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
        
        # print(f"{Qs_state.shape = }")
        # print(f"{Qs_state = }")
        # print(action)
        y = (
            reward + (self.gamma * th.multiply(Q_max_state_next.values, (1 - done)))
        ).to(device)
        Q_state_action = (
            Qs_state[np.arange(Qs_state.shape[0]), action.tolist()]
        ).to(device)
        
        # print(f"{y.shape = }")
        # print(f"{y = }")
        # print(f"{Q_state_action.shape = }")
        # print(f"{Q_state_action = }")
        # print(f"{(y - Q_state_action) = }")
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
        
        pred_state_next, pred_reward = self.M(state, action)
        loss_state = th.mean(
            self.M_component_criterion(state_next, pred_state_next),
            dim = (1, 2, 3)
        )
        loss_reward = self.M_component_criterion(reward, pred_reward)
        
        losses = self.M_criterion(loss_state, loss_reward)
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
    
    def update_inv(self, obj: str) -> None:
        self.inv += get_inv_code[obj]
        if obj and obj not in self.inv_objs:
            self.inv_objs.append(obj)
        
    def reset_inv(self) -> None:
        self.inv = 0.
        self.inv_objs = []
        self.tool_hotkeys = dict()
        
    @staticmethod
    def get_text(chars: np.ndarray) -> str:
        msg = ""
        for char in chars:
            if char != 0:
                msg += chr(char)
                
        return msg
    
    @staticmethod
    def get_msg_object(msg: str) -> str:
        for obj in inv_objects_possible:
            if obj in msg and "break" not in msg:
                return obj
            
        return ""
    
    @staticmethod
    def chars_to_obj(chars: np.ndarray) -> str:
        return Agent.get_msg_object(
            Agent.get_text(chars)
        )
        
    @staticmethod
    def get_pickup_meta_action(obj: str) -> tuple[int]:
        return pickup_meta_actions[obj]
    
    def get_tool_meta_action(self, obj: str) -> tuple[int]:
        if obj in self.inv_objs:
            return tool_meta_actions[obj]
        else:
            return (tool_meta_actions[obj][0],)
        
    def add_tool_hotkey(self, obj: str, hotkey: int) -> None:
        self.tool_hotkeys[obj] = hotkey

import numpy as np
from replay_buffer import ReplayBuffer

class Agent():
    obs_shape: tuple[int] # observation shape
    obs_keys: list[str] # list of keys to index minihack observations
    obs_dtype: np.dtype # dtype of observations
    act_shape: int # number of actions the agent can choose from
    
    batch_size: int # number of samples to draw from replay buffer
    replay_buffer: ReplayBuffer # replay buffer
    
    def __init__(self, obs_shape: tuple[int], obs_keys: list[str], obs_dtype: np.dtype, act_shape: int, batch_size: int,\
                 max_replay_buffer_len: int, priority_default: float, alpha: float, beta: float, phi: float, c: float) -> None:
        
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
        
    def reshape_state(self, channels: list[np.ndarray]) -> np.ndarray:
        obs = np.zeros(self.obs_shape, dtype=self.obs_dtype)
        assert len(channels) == self.obs_shape[0]
        
        for i, chn in enumerate(channels):
             obs[i] = chn
             
        return obs
    
    def normalize_states(self, state: np.ndarray) -> np.ndarray:
        max_val = np.iinfo(self.obs_dtype).max
        min_val = np.iinfo(self.obs_dtype).min
        
        normed = np.zeros_like(state.shape, dtype=np.float32)
        normed = state
        
        return (normed - min_val) / (max_val - min_val)
        
__author__ = "John Statheros (1828326)"
__maintainer__ = "Timothy Reeder (455840)"
__version__ = '0.1.0'
__status__ = 'Development'

import torch as th
from torch import nn


class DQN(nn.Module):
    """Deep Q-Network (DQN) model.

    Args:
        obs_shape: A tuple of integers representing the shape of the observation space.
        act_shape: An integer representing the number of actions available to the agent.
    """
    def __init__(self, obs_shape: tuple[int], act_shape: int) -> None:
        super().__init__()

        # Calculate the number of input channels based on the observation shape
        in_chn, _, _ = obs_shape

        # Define the layers of the DQN network
        layers = [nn.Flatten(),
                  nn.Linear(in_features=64*4*4, out_features=512),
                  nn.ReLU(inplace=True),
                  nn.Linear(in_features=512, out_features=512),
                  nn.ReLU(inplace=True),
                  nn.Linear(in_features=512, out_features=act_shape)]

        # Define the encoder layers
        self.enconder = nn.ModuleList()
        # 32x6x6
        self.enconder.append(self.__conv_block(in_channels=in_chn,
                                               out_channels=32,
                                               kernel_size=4,
                                               stride=1,
                                               padding=0,
                                               activation=True))
        # 64x4x4
        self.enconder.append(self.__conv_block(in_channels=32,
                                               out_channels=64,
                                               kernel_size=3,
                                               stride=1,
                                               padding=0,
                                               activation=True))
        # 64x4x4
        self.enconder.append(self.__conv_block(in_channels=64,
                                               out_channels=64,
                                               kernel_size=3,
                                               stride=1,
                                               padding='same',
                                               activation=True))
        
        self.sequential = nn.Sequential(*layers)
        
    def __conv_block(self,
                     in_channels: int,
                     out_channels: int,
                     kernel_size: int,
                     stride: int,
                     padding: int,
                     activation: bool) -> nn.Sequential:
        """Defines a convolutional block.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            kernel_size: The size of the kernel.
            stride: The stride of the convolution.
            padding: The padding of the convolution.
            activation: Whether to apply a ReLU activation function after the
                convolution.

        Returns:
            A sequential model containing the convolutional block.
        """
        layers = [nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)]
        
        if activation:
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
        
    def forward(self, state: th.tensor) -> th.tensor:
        """Computes the Q-values for the given state.

        Args:
            state: A PyTorch tensor representing the state.

        Returns:
            A PyTorch tensor representing the Q-values for the given state.
        """
        x = state

        # Pass the state through the encoder layers.
        for conv_block in self.enconder:
            x = conv_block(x)
        
        return self.sequential(x)
    
    
class DeepModel(nn.Module):
    """Deep Q-Network (DQN) model.

    Args:
        obs_shape: A tuple of integers representing the shape of the observation space.
        act_shape: An integer representing the number of actions available to the agent.
    """
    def __init__(self, obs_shape: tuple[int], act_shape: int) -> None:
        super().__init__()
        # Calculate the number of input channels based on the observation shape.
        in_chn, _, _ = obs_shape

        # Define the action embedding layer.
        self.action_embedding = nn.Embedding(act_shape, 3*3, _freeze=True)
        
        # Define the encoder layers.
        self.encoder = nn.ModuleList()
        # 8x7x7
        self.encoder.append(self.__conv_block(in_channels=in_chn,
                                              out_channels=8,
                                              kernel_size=3,
                                              stride=1,
                                              padding=0,
                                              activation=True))
        # 16x5x5
        self.encoder.append(self.__conv_block(in_channels=8,
                                              out_channels=16,
                                              kernel_size=3,
                                              stride=1,
                                              padding=0,
                                              activation=True))
        # 32x3x3
        self.encoder.append(self.__conv_block(in_channels=16,
                                              out_channels=32,
                                              kernel_size=3,
                                              stride=1,
                                              padding=0,
                                              activation=True))

        # Define the decoder layers.
        self.decoder = nn.ModuleList()
        self.decoder.append(self.__upscale_conv_block(in_channels=33,
                                                      out_channels=16,
                                                      kernel_size=3,
                                                      stride=1,
                                                      output_size=(5,5),
                                                      mode='bilinear',
                                                      activation=True))
        # concatenate 16 encoder channels
        self.decoder.append(self.__upscale_conv_block(in_channels=32,
                                                      out_channels=8,
                                                      kernel_size=3,
                                                      stride=1,
                                                      output_size=(7,7),
                                                      mode='bilinear',
                                                      activation=True))
        # concatenate 8 encoder channels
        self.decoder.append(self.__upscale_conv_block(in_channels=16,
                                                      out_channels=4,
                                                      kernel_size=3,
                                                      stride=1,
                                                      output_size=(9,9),
                                                      mode='bilinear',
                                                      activation=True))
        # concatenate input frames:
        # 4 output channels, 0-1 are future frame, 2 is inventory, 3 is reward
        self.decoder.append(self.__conv_block(in_channels=4,
                                              out_channels=in_chn+1,
                                              kernel_size=3,
                                              stride=1,
                                              padding='same',
                                              activation=False))
        
    @staticmethod
    def __conv_block(in_channels: int,
                     out_channels: int,
                     kernel_size: int,
                     stride: int,
                     padding: int,
                     activation: bool) -> nn.Sequential:
        """Defines a convolutional block.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            kernel_size: The size of the kernel.
            stride: The stride of the convolution.
            padding: The padding of the convolution.
            activation: Whether to apply a ReLU activation function after the
                convolution.

        Returns:
            A sequential model containing the convolutional block.
        """
        layers = [nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)]
        
        if activation:
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    @staticmethod
    def __upscale_conv_block(in_channels: int,
                             out_channels: int,
                             kernel_size: int,
                             stride: int,
                             output_size: tuple[int],
                             mode: str,
                             activation: bool) -> nn.Sequential:
        """
        Defines an upscale convolutional block.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            kernel_size: The size of the kernel.
            stride: The stride of the convolution.
            output_size: The output size of the up-scaled feature map.
            mode: The interpolation mode to use.
            activation: Whether to apply a ReLU activation function after the
                convolution.

        Returns:
            A sequential model containing the upscale convolutional block.
        """
        layers = [
            Interpolation(output_size=output_size,
                          mode=mode,
                          align_corners=False),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding='same')
        ]
        
        if activation:
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, state: th.Tensor, action: th.Tensor) -> tuple[th.Tensor]:
        """
        Computes the next state and reward for the given state and action.

        Args:
            state: A PyTorch tensor representing the state.
            action: A PyTorch tensor representing the action.

        Returns:
            A tuple of PyTorch tensors representing the next state and reward.
        """
        x = state
        a = action[:, None]
        
        # encoder
        encoder_outputs = []
        for i, conv_block in enumerate(self.encoder):
            x = conv_block(x)
            
            if i < len(self.encoder) - 1:
                encoder_outputs.append(x)
            
        # bottleneck
        action_channel = self.action_embedding(a)
        action_channel = action_channel.view(action_channel.size(0), 1, 3, 3)
        x = th.concat([x, action_channel], dim=1)
        
        # decoder
        for i, upscale_conv_block in enumerate(self.decoder):
            x = upscale_conv_block(x)
            
            if i < len(self.decoder) - 2:
                x = th.concat([x, encoder_outputs.pop()], dim=1)

        # Compute the next state and reward
        state_next = x[:, 0:3, :, :]
        reward = x[:, 3:, :, :]
        reward = nn.functional.adaptive_avg_pool2d(reward, 1)
        
        return state_next, reward.reshape((-1,))
    

class Interpolation(nn.Module):
    """
    Module for performing interpolation.

    Args:
        output_size: A tuple of integers representing the output size of the interpolated feature map.
        mode: The interpolation mode to use.
        align_corners: Whether to align the corners of the input and output feature maps.
    """
    def __init__(self,
                 output_size: tuple[int],
                 mode: str,
                 align_corners: bool) -> None:
        super().__init__()
        
        self.output_size = output_size
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Performs interpolation on the input tensor.

        Args:
            x: A PyTorch tensor representing the input feature map.

        Returns:
            A PyTorch tensor representing the interpolated feature map.
        """
        return nn.functional.interpolate(x,
                                         size=self.output_size,
                                         mode=self.mode,
                                         align_corners=self.align_corners)

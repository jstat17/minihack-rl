import torch as th
from torch import nn


class DQN(nn.Module):
    def __init__(self, obs_shape: tuple[int], act_shape: int) -> None:
        super().__init__()
        
        in_chn, _, _ = obs_shape
        
        # layers = [
        #     nn.Conv2d(in_channels=in_chn, out_channels=8, kernel_size=3, stride=1, padding='same'), # 8x9x9
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='same'), # 16x9x9
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same'), # 32x9x9
        #     nn.ReLU(inplace=True),
        #     nn.Flatten(),
        #     nn.Linear(in_features=32*9*9, out_features=256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=256, out_features=act_shape)
        # ]
        layers = [
            nn.Flatten(),
            nn.Linear(in_features=64*4*4, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=act_shape)
        ]
        
        self.enconder = nn.ModuleList()
        self.enconder.append(self.__conv_block(in_channels=in_chn, out_channels=32, kernel_size=4, stride=1, padding=0, activation=True)) # 16x6x6
        self.enconder.append(self.__conv_block(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, activation=True)) # 32x4x4
        self.enconder.append(self.__conv_block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same', activation=True)) # 64x4x4
        
        self.sequential = nn.Sequential(*layers)
        
    def __conv_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int | str, activation: bool) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        
        if activation:
            layers.append(
                nn.ReLU(inplace=True)
            )
            
        return nn.Sequential(*layers)
        
    def forward(self, state: th.tensor) -> th.tensor:
        x = state
        
        for conv_block in self.enconder:
            x = conv_block(x)
        
        # x = th.concat([x, state], dim=1)
        
        return self.sequential(x)
    
    
class DeepModel(nn.Module):
    def __init__(self, obs_shape: tuple[int], act_shape: int) -> None:
        super().__init__()
        
        in_chn, _, _ = obs_shape
        self.action_embedding = nn.Embedding(act_shape, 3*3, _freeze=True)
        
        # encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(self.__conv_block(in_channels=in_chn, out_channels=8, kernel_size=3, stride=1, padding=0, activation=True)) # 8x7x7
        self.encoder.append(self.__conv_block(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0, activation=True)) # 16x5x5
        self.encoder.append(self.__conv_block(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, activation=True)) # 32x3x3
        
        # bottleneck adds an additional layer in action embedding
        
        # decoder
        self.decoder = nn.ModuleList()
        self.decoder.append(self.__upscale_conv_block(in_channels=33, out_channels=16, kernel_size=3, stride=1, output_size=(5,5), mode='bilinear', activation=True))
        # concatenate 16 encoder channels
        self.decoder.append(self.__upscale_conv_block(in_channels=32, out_channels=8, kernel_size=3, stride=1, output_size=(7,7), mode='bilinear', activation=True))
        # concatenate 8 encoder channels
        self.decoder.append(self.__upscale_conv_block(in_channels=16, out_channels=4, kernel_size=3, stride=1, output_size=(9,9), mode='bilinear', activation=True))
        # concatenate input frames --- 3 output channels, 0-1 are future frame, 2 is reward
        self.decoder.append(self.__conv_block(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding='same', activation=False))
        
    def __conv_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int | str, activation: bool) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        
        if activation:
            layers.append(
                nn.ReLU(inplace=True)
            )
            
        return nn.Sequential(*layers)
    
    def __upscale_conv_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, output_size: tuple[int], mode: str,\
                             activation: bool) -> nn.Sequential:
        layers = [
            Interpolation(output_size=output_size, mode=mode, align_corners=False),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding='same')
        ]
        
        if activation:
            layers.append(
                nn.ReLU(inplace=True)
            )
            
        return nn.Sequential(*layers)
    
    def forward(self, state: th.Tensor, action: th.Tensor) -> tuple[th.Tensor]:
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
            # elif i == 2:
            #     x = th.concat([x, state[:, 0:2, :, :]], dim=1)
            
                
        state_next = x[:, 0:2, :, :]
        reward = x[:, 2:, :, :]
        reward = nn.functional.adaptive_avg_pool2d(reward, 1)
        
        return state_next, reward.reshape((-1,))
    

class Interpolation(nn.Module):
    def __init__(self, output_size: tuple[int], mode: str, align_corners: bool) -> None:
        super().__init__()
        
        self.output_size = output_size
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        return nn.functional.interpolate(x, size=self.output_size, mode=self.mode, align_corners=self.align_corners)
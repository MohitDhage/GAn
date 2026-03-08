import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    3D CNN Discriminator to classify (64x64x64) volumes as real or fake.
    """
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # (B, 1, 64, 64, 64) -> (B, 32, 32, 32, 32)
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 32, 32, 32, 32) -> (B, 64, 16, 16, 16)
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 64, 16, 16, 16) -> (B, 128, 8, 8, 8)
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 128, 8, 8, 8) -> (B, 256, 4, 4, 4)
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 256, 4, 4, 4) -> (B, 1, 1, 1, 1)
            nn.Conv3d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)

if __name__ == "__main__":
    # Test discriminator
    d = Discriminator()
    dummy_input = torch.randn(2, 1, 64, 64, 64)
    output = d(dummy_input)
    print(f"Discriminator output shape: {output.shape}") # Should be (2, 1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)    # B x HW x C'
        key = self.key(x).view(B, -1, H * W)                         # B x C' x HW
        attention = torch.bmm(query, key)                           # B x HW x HW
        attention = F.softmax(attention, dim=-1)

        value = self.value(x).view(B, -1, H * W)                    # B x C x HW
        out = torch.bmm(value, attention.permute(0, 2, 1))          # B x C x HW
        out = out.view(B, C, H, W)

        return self.gamma * out + x

class AttentionGenerator(nn.Module):
    def __init__(self, feature_dim=128):
        super(AttentionGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),  # 3 channels + mask
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SelfAttention(128),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SelfAttention(256),
        )

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, 256 * 32 * 32),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, image, mask, feature):
        x = torch.cat([image, mask], dim=1)  # Shape: (B, 4, 256, 256)
        enc = self.encoder(x)  # Shape: (B, 256, 32, 32)
        feat = self.feature_proj(feature).view(-1, 256, 32, 32)
        combined = torch.cat([enc, feat], dim=1)
        out = self.decoder(combined)
        return out
    

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),  # 3 channels + mask
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, image, mask):
        # 4-channel input
        x = torch.cat([image, mask], dim=1)
        return self.model(x)


class StableContextualAttention(nn.Module):
    def __init__(self, in_channels, rate=2):
        super(StableContextualAttention, self).__init__()
        self.rate = rate
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
       
    def forward(self, x, mask):
        # Simplified implementation to avoid numerical instability
        B, C, H, W = x.shape
       
        # Split into known and unknown regions
        x_known = x * (1 - mask)
       
        # Simple attention mechanism
        # Apply convolution to get query and key features
        query = F.conv2d(x * mask, self.conv.weight, self.conv.bias, padding=0)
        key = F.conv2d(x_known, self.conv.weight, self.conv.bias, padding=0)
       
        # Normalize features for stable cosine similarity
        query_norm = F.normalize(query.view(B, C, -1), dim=1)
        key_norm = F.normalize(key.view(B, C, -1), dim=1)
       
        # Calculate attention with numerical stability
        attention = torch.bmm(query_norm.permute(0, 2, 1), key_norm)
        attention = F.softmax(attention / 0.1, dim=-1)  # Temperature scaling for stability
       
        # Weighted sum of values
        value = x_known.view(B, C, -1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
       
        # Combine with original features
        return x_known + out * mask
    

def stable_edge_aware_loss(generated, real, mask, eps=1e-8):
    # Use pre-defined Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3).cuda()
   
    # Apply to each channel with gradient clipping
    def get_edges(image):
        edges = []
        for c in range(image.shape[1]):
            channel = image[:, c:c+1]
            grad_x = F.conv2d(channel, sobel_x, padding=1)
            grad_y = F.conv2d(channel, sobel_y, padding=1)
            # Add epsilon for numerical stability
            edge = torch.sqrt(torch.clamp(grad_x**2 + grad_y**2, min=eps))
            edges.append(edge)
        return torch.cat(edges, dim=1)
   
    # Get edges with gradient clipping
    gen_edge = torch.clamp(get_edges(generated), max=10.0)
    real_edge = torch.clamp(get_edges(real), max=10.0)
   
    # Calculate loss only in mask region with gradient clipping
    edge_loss = F.l1_loss(gen_edge * mask, real_edge * mask)
    return torch.clamp(edge_loss, max=10.0)  # Prevent extreme values


class StableGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(StableGatedConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
       
    def forward(self, x):
        feature = self.conv2d(x)
        # Use tanh instead of sigmoid for better gradient stability
        mask = torch.tanh(self.mask_conv2d(x)) * 0.5 + 0.5
        return feature * mask
    

class StableGenerator(nn.Module):
    def __init__(self, feature_dim=128):
        super(StableGenerator, self).__init__()
       
        # Encoder with stable gated convolutions
        self.encoder = nn.Sequential(
            StableGatedConv2d(4, 64, 4, 2, 1),  # 3 channels + mask
            nn.LeakyReLU(0.2, inplace=True),
            StableGatedConv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),  # Use InstanceNorm instead of BatchNorm for stability
            nn.LeakyReLU(0.2, inplace=True),
            StableGatedConv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
       
        # Feature projection with weight normalization
        self.feature_proj = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feature_dim, 256 * 32 * 32)),
            nn.LeakyReLU(0.2, inplace=True)
        )
       
        # Simplified attention module
        self.attention = StableContextualAttention(512)
       
        # Decoder with skip connections for stability
        self.decoder1 = nn.Sequential(
            StableGatedConv2d(512, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
       
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
       
        self.decoder3 = nn.Sequential(
            StableGatedConv2d(128, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
       
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
       
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()  # Constrain output to [-1, 1]
        )

    def forward(self, image, mask, feature):
        # Input concatenation
        x = torch.cat([image, mask], dim=1)
       
        # Encoding
        enc = self.encoder(x)
       
        # Feature projection with clamping for stability
        feat = self.feature_proj(feature).view(-1, 256, 32, 32)
        feat = torch.clamp(feat, -10, 10)  # Prevent extreme values
       
        # Combine encoder output and feature projection
        combined = torch.cat([enc, feat], dim=1)
       
        # Apply attention with downsampled mask for stability
        mask_downsampled = F.interpolate(mask, size=combined.shape[2:], mode='nearest')
        attended = self.attention(combined, mask_downsampled)
       
        # Decode with gradient clamping between stages
        x = self.decoder1(attended)
        x = torch.clamp(x, -10, 10)
       
        x = self.decoder2(x)
        x = torch.clamp(x, -10, 10)
       
        x = self.decoder3(x)
        x = torch.clamp(x, -10, 10)
       
        x = self.decoder4(x)
        x = torch.clamp(x, -10, 10)
       
        x = self.decoder5(x)
       
        # Apply mask for final output
        return image * (1 - mask) + x * mask
    


def load_generator():
    generator = StableGenerator()
    checkpoint_path = 'checkpoint_epoch_50.pth'  
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    generator.load_state_dict(checkpoint['generator_state_dict'])
    return generator
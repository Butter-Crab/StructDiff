import torch
import torch.nn.functional as F
from torch import nn


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.sobel_x = nn.Parameter(sobel_x.repeat(3, 1, 1, 1), requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_y.repeat(3, 1, 1, 1), requires_grad=False)

    def forward(self, recon_image, mix_image):
        device = recon_image.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        original_image = mix_image[:, :3, :, :]
        mask = mix_image[:, 3:6, :, :]

        recon_edge_x = F.conv2d(recon_image, sobel_x, padding=1, groups=3)
        recon_edge_y = F.conv2d(recon_image, sobel_y, padding=1, groups=3)
        original_edge_x = F.conv2d(original_image, sobel_x, padding=1, groups=3)
        original_edge_y = F.conv2d(original_image, sobel_y, padding=1, groups=3)

        recon_edges = torch.sqrt(recon_edge_x ** 2 + recon_edge_y ** 2)
        original_edges = torch.sqrt(original_edge_x ** 2 + original_edge_y ** 2)

        recon_edges_masked = recon_edges * mask
        original_edges_masked = original_edges * mask
        return F.l1_loss(recon_edges_masked, original_edges_masked)

import math
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
import pdb

def params(N):
    return {
        # Ensure requires_grad=True for any parameter you want to optimize
        "means3D": torch.rand(N, 3, device='cuda:0', requires_grad=True),
        # "means2D": torch.zeros((N, 3), dtype=torch.float32, device='cuda:0'),  # Not optimizing this (by default)
        # "shs": torch.rand(N, 3, 3, device='cuda:0'),  # Not optimizing this (by default)
        "sh_objs": torch.rand(N, 1, 3, device='cuda:0', requires_grad=True),
        "colors_precomp": torch.rand(N, 3, device='cuda:0', requires_grad=True),
        "opacities": torch.rand(N, 1, device='cuda:0', requires_grad=True),
        "scales": torch.rand(N, 3, device='cuda:0', requires_grad=True),
        "rotations": torch.randn(N, 4, device='cuda:0', requires_grad=True),
        # "cov3D_precomp": None,
        "theta": torch.zeros(3, requires_grad=True, device='cuda:0'),
        "rho": torch.zeros(3, requires_grad=True, device='cuda:0'),
    }
def render(params):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    
    raster_settings = GaussianRasterizationSettings(
        image_height=680, 
        image_width=1200, 
        tanfovx=1.0, 
        tanfovy=0.5666666666666667, 
        bg=torch.tensor([0., 0., 0.], device='cuda:0'), 
        scale_modifier=1.0, 
        viewmatrix=torch.tensor([[[ 1.0000e+00,  6.2655e-17,  6.1216e-17,  0.0000e+00],
         [-4.0461e-08,  1.0000e+00,  5.9605e-08,  0.0000e+00],
         [ 1.9389e-08,  1.1921e-07,  1.0000e+00,  0.0000e+00],
         [-1.8626e-08, -1.1921e-07, -1.1921e-07,  1.0000e+00]]], device='cuda:0'), 
        projmatrix=torch.tensor([[[ 1.0000e+00,  1.1048e-16,  6.1222e-17,  6.1216e-17],
         [-4.0510e-08,  1.7647e+00,  5.9611e-08,  5.9605e-08],
         [-8.3331e-04, -1.4704e-03,  1.0001e+00,  1.0000e+00],
         [-1.8527e-08, -2.1019e-07, -1.0001e-02, -1.1921e-07]]], device='cuda:0'), 
        projmatrix_raw=torch.tensor([[[ 1.0000e+00,  1.1048e-16,  6.1222e-17,  6.1216e-17],
         [-4.0510e-08,  1.7647e+00,  5.9611e-08,  5.9605e-08],
         [-8.3331e-04, -1.4704e-03,  1.0001e+00,  1.0000e+00],
         [-1.8527e-08, -2.1019e-07, -1.0001e-02, -1.1921e-07]]], device='cuda:0'),
        sh_degree=0, 
        campos=torch.tensor([1.8626e-08, 1.1921e-07, 1.1921e-07], device='cuda:0'), 
        prefiltered=False, 
        debug=True
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = params["means3D"]
    screenspace_points = (
        torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    means2D = screenspace_points
    colors_precomp = params["colors_precomp"]
    opacity = params["opacities"]
    scales = params["scales"]
    rotations = params["rotations"]
    # cov3D_precomp = params["cov3D_precomp"]
    cam_rot_delta = params["theta"]
    cam_trans_delta = params["rho"]
    shs_objs = params["sh_objs"]    


    rendered_image, rendered_obj, radii, depth, opacity, n_touched = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        sh_objs=shs_objs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        theta=cam_rot_delta,
        rho=cam_trans_delta,
    )
    # print(rendered_image.shape, rendered_obj.shape, radii.shape, depth.shape, opacity.shape, n_touched.shape)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
        "sh_objs": rendered_obj,
    }

if __name__ == "__main__":
    # Read sample img, depth and semantic
    import cv2
    import numpy as np
    img = cv2.imread('test/frame000000.jpg')
    depth = cv2.imread('test/depth000000.png')
    seg_map = np.load('test/frame000000_s.npy')
    fet_map = np.load('test/frame000000_f.npy')
    img = torch.tensor(img, device='cuda:0').float().permute(2, 0, 1) / 255.0
    depth = torch.tensor(depth, device='cuda:0').float()
    seg_map = torch.tensor(seg_map, device='cuda:0').float()
    fet_map = torch.tensor(fet_map, device='cuda:0').float()
    
    
    y, x = torch.meshgrid(torch.arange(0, 680, device='cuda:0'), torch.arange(0, 1200, device='cuda:0'))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    seg = seg_map[:, y, x].squeeze(-1).long()
    mask = seg != -1
    point_feature = fet_map[seg[0:1]].squeeze(0)
    mask = mask[0:1].reshape(1, 680, 1200)
    point_feature = point_feature.reshape(680, 1200, -1).permute(2, 0, 1)
    print("img", img.shape, "depth", depth.shape, "point_feature", point_feature.shape, "mask", mask.shape, img.max(), img.min())
    
    init_params = params(3000)
    trainable_params = [
        init_params["means3D"],
        init_params["colors_precomp"],
        init_params["opacities"],
        init_params["scales"],
        init_params["rotations"],
        init_params["theta"],
        init_params["rho"],
        init_params["sh_objs"],
    ]
    # start lr = 0.01, decay by 0.1 every 1000 steps
    optimizer = torch.optim.Adam(trainable_params, lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3000, 6000], gamma=0.1)
    losses = []
    mse_loss = torch.nn.MSELoss()
    for i in range(10000):
        optimizer.zero_grad()
        
        
        rendered_pkg = render(init_params)
        rendered_img = rendered_pkg['render']
        rendered_obj = rendered_pkg['sh_objs']
        loss = mse_loss(rendered_img, img) + mse_loss(rendered_obj, point_feature) + init_params["scales"].abs().mean() * 0.01
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i % 500 == 0:
            # save loss fig and current rendered image
            print(f"Step {i}, loss {loss.item()}")
            cv2.imwrite(f"test/rendered_{i}.png", (rendered_img.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8))
            cv2.imwrite(f"test/point_feature_{i}.png", (rendered_obj.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8))
            # Save loss fig
            import matplotlib.pyplot as plt
            plt.plot(losses)
            plt.savefig("test/loss.png")
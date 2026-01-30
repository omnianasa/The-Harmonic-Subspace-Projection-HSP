import torch
import cv2
import numpy as np

class FoveatedTransform:
    def __init__(self, center=(112, 112), radius=50, blur_kernel=25):
        self.center = center
        self.radius = radius
        self.blur_kernel = blur_kernel

    def __call__(self, tensor):
        # Input tensor is (C, H, W). Convert to (H, W, C) for OpenCV
        img = tensor.permute(1, 2, 0).numpy()
        h, w, c = img.shape

        # Create the peripheral blur
        low_res = cv2.GaussianBlur(img, (self.blur_kernel, self.blur_kernel), 0)

        # Create circular mask for the fovea (central focus)
        y, x = np.ogrid[:h, :w]
        mask = (x - self.center[0])**2 + (y - self.center[1])**2 <= self.radius**2
        
        # Merge: Keep center sharp, blur everything else
        foveated = low_res.copy()
        foveated[mask] = img[mask]

        return torch.from_numpy(foveated).permute(2, 0, 1)
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Hooks-based Grad-CAM for CustomResNet.

    Supports one or more target layers. When multiple layers are given,
    the per-layer CAMs are upsampled to input spatial size and averaged,
    which improves localization for deep ResNets with small final feature maps.

    Usage:
        cam_extractor = GradCAM(model, target_layers=[
            model.feature_extractor.layer3,
            model.feature_extractor.layer4,
        ])
        cam = cam_extractor(tile, class_idx=1)   # [H, W] numpy array
        cam_extractor.remove_hooks()
    """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = (
            target_layers if isinstance(target_layers, (list, tuple)) else [target_layers]
        )
        self._activations = {}
        self._gradients = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for i, layer in enumerate(self.target_layers):
            self._hooks.append(
                layer.register_forward_hook(self._make_fwd_hook(i))
            )
            self._hooks.append(
                layer.register_full_backward_hook(self._make_bwd_hook(i))
            )

    def _make_fwd_hook(self, idx):
        def hook(module, input, output):
            self._activations[idx] = output.detach()
        return hook

    def _make_bwd_hook(self, idx):
        def hook(module, grad_input, grad_output):
            self._gradients[idx] = grad_output[0].detach()
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, tile, class_idx="pred"):
        """
        Compute the Grad-CAM for a single tile.

        Args:
            tile: [C, H, W] tensor (un-batched) — as returned by BleachDataset
            class_idx: int class to explain, or "pred" to use the model's prediction

        Returns [H, W] float32 numpy array, values in [0, 1].
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        x = tile.unsqueeze(0).to(device)   # [1, C, H, W]
        x.requires_grad_(False)

        self._activations.clear()
        self._gradients.clear()

        logits = self.model(x)             # [1, num_classes]

        if class_idx == "pred":
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        H_in, W_in = tile.shape[1], tile.shape[2]
        cam_sum = np.zeros((H_in, W_in), dtype=np.float32)

        for i in range(len(self.target_layers)):
            acts = self._activations[i]    # [1, C_l, h_l, w_l]
            grads = self._gradients[i]     # [1, C_l, h_l, w_l]

            weights = grads.mean(dim=(2, 3), keepdim=True)  # global avg pool
            cam_layer = (weights * acts).sum(dim=1, keepdim=True)  # [1,1,h_l,w_l]
            cam_layer = F.relu(cam_layer)

            cam_layer = F.interpolate(
                cam_layer, size=(H_in, W_in), mode="bilinear", align_corners=False
            )
            cam_sum += cam_layer.squeeze().cpu().numpy()

        cam_avg = cam_sum / len(self.target_layers)
        cam_min, cam_max = cam_avg.min(), cam_avg.max()
        if cam_max - cam_min > 1e-8:
            cam_avg = (cam_avg - cam_min) / (cam_max - cam_min)

        return cam_avg


def make_occlusion_map(model, tile, patch_size=16, stride=8,
                       baseline="mean", target_class="pred", device="cpu"):
    """
    Sliding-window occlusion sensitivity map for a single tile.

    Occludes a square patch at every position and measures the drop in the
    target class score. High values = region matters to the prediction.

    Args:
        model: CustomResNet, in eval mode
        tile: [C, H, W] tensor (un-batched)
        patch_size: side length of the occlusion patch in pixels
        stride: step size between patch positions
        baseline: "mean" fills the patch with the per-channel mean of the tile;
                  "zero" fills with zeros
        target_class: int or "pred" (uses the unoccluded prediction)
        device: torch device string

    Returns [H, W] float32 numpy array of score drops (higher = more important).
    """
    model.eval()
    model.to(device)

    x = tile.unsqueeze(0).to(device)      # [1, C, H, W]

    with torch.no_grad():
        base_logits = model(x)
        base_scores = torch.softmax(base_logits, dim=1)

        if target_class == "pred":
            target_class = int(base_logits.argmax(dim=1).item())

        base_score = base_scores[0, target_class].item()

    _, C, H, W = x.shape
    occ_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    if baseline == "mean":
        fill = x.mean(dim=(2, 3), keepdim=True).expand_as(x)
    else:
        fill = torch.zeros_like(x)

    with torch.no_grad():
        for top in range(0, H - patch_size + 1, stride):
            for left in range(0, W - patch_size + 1, stride):
                x_occ = x.clone()
                x_occ[:, :, top:top + patch_size, left:left + patch_size] = \
                    fill[:, :, top:top + patch_size, left:left + patch_size]

                occ_score = torch.softmax(model(x_occ), dim=1)[0, target_class].item()
                drop = base_score - occ_score

                occ_map[top:top + patch_size, left:left + patch_size] += drop
                count_map[top:top + patch_size, left:left + patch_size] += 1

    count_map = np.where(count_map == 0, 1, count_map)
    occ_map /= count_map

    occ_min, occ_max = occ_map.min(), occ_map.max()
    if occ_max - occ_min > 1e-8:
        occ_map = (occ_map - occ_min) / (occ_max - occ_min)

    return occ_map

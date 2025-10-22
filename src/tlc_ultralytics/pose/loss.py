from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import KeypointLoss, v8PoseLoss
from ultralytics.utils.ops import xyxy2xywh

from tlc_ultralytics.detect.loss import UnreducedBboxLoss


class v8UnreducedPoseLoss(v8PoseLoss):
    """v8PoseLoss that returns unreduced per-anchor losses for per-sample computation."""

    def __init__(self, model, training: bool = False):  # model must be de-paralleled
        super().__init__(model)
        # Prefer dataset-provided OKS sigmas if available on the model instance
        oks_sigmas = getattr(model, "oks_sigmas", None)
        if oks_sigmas is not None:
            sigmas_tensor = torch.as_tensor(oks_sigmas, device=self.device, dtype=torch.float32)
            self.keypoint_loss = KeypointLoss(sigmas=sigmas_tensor)
        # Replace bbox loss with unreduced variant and store training flag
        self.bbox_loss = UnreducedBboxLoss(self.reg_max)
        self.training = training

    def __call__(self, preds, batch) -> dict[str, torch.Tensor]:
        """Calculate unreduced losses for box, cls, dfl, pose (kpts), and kobj.

        Returns a dict of tensors shaped (batch, num_anchors) for each component.
        """
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (B, A, C)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (B, A, 4*reg_max or 4)
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()  # (B, A, K*kdim)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = self.make_anchors(feats)

        # Targets
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes and keypoints
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # (B, A, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (B, A, K, D)

        # Assignment
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # Classification loss (per-anchor)
        cls_loss = self.bce(pred_scores, target_scores.to(dtype)).sum(dim=2)

        # Box and DFL losses (per-anchor, zeros where not fg)
        box_loss_full = torch.zeros_like(cls_loss)
        dfl_loss_full = torch.zeros_like(cls_loss)
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            box_loss, dfl_loss = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                max(target_scores.sum(), 1),
                fg_mask,
            )
            box_loss_full[fg_mask] = box_loss.to(cls_loss.dtype).squeeze()
            dfl_loss_full[fg_mask] = dfl_loss.to(cls_loss.dtype).squeeze()

        # Keypoints losses (pose and kobj), per-anchor on fg positions
        pose_loss_full = torch.zeros_like(cls_loss)
        kobj_loss_full = torch.zeros_like(cls_loss)
        if fg_mask.any():
            # Prepare selected GT keypoints batched per image as in v8PoseLoss.calculate_keypoints_loss
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            bidx = batch_idx.flatten()
            bs = len(fg_mask)
            max_kpts = torch.unique(bidx, return_counts=True)[1].max()
            batched_keypoints = torch.zeros(
                (bs, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
            )
            for i in range(bs):
                keypoints_i = keypoints[bidx == i]
                batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

            target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
            selected_keypoints = batched_keypoints.gather(
                1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
            )

            # Divide coords by stride to match pred scale
            selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

            gt_kpt = selected_keypoints[fg_mask]
            area = xyxy2xywh(target_bboxes[fg_mask])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[fg_mask]
            if gt_kpt.shape[-1] == 3:
                kpt_mask = gt_kpt[..., 2] != 0
            else:
                kpt_mask = torch.full_like(gt_kpt[..., 0], True, dtype=torch.bool)

            # Pose location loss per-anchor, replicate KeypointLoss but keep per-anchor reduction
            d = (pred_kpt[..., 0] - gt_kpt[..., 0]).pow(2) + (pred_kpt[..., 1] - gt_kpt[..., 1]).pow(2)
            # e = d / ((2 * self.keypoint_loss.sigmas) ** 2 * (area + 1e-9) * 2)
            e = d / ((2 * self.keypoint_loss.sigmas).pow(2) * (area + 1e-9) * 2)
            kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
            pose_loss_vec = (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean(dim=1)

            # Keypoint objectness loss per-anchor
            if pred_kpt.shape[-1] == 3:
                kobj_loss_vec = F.binary_cross_entropy_with_logits(
                    pred_kpt[..., 2], kpt_mask.float(), reduction="none"
                ).mean(dim=1)
            else:
                kobj_loss_vec = torch.zeros_like(pose_loss_vec)

            # Scatter back into full tensors
            pose_loss_full[fg_mask] = pose_loss_vec.to(cls_loss.dtype)
            kobj_loss_full[fg_mask] = kobj_loss_vec.to(cls_loss.dtype)

        losses: dict[str, torch.Tensor] = {
            "cls_loss": cls_loss,
            "box_loss": box_loss_full,
            "dfl_loss": dfl_loss_full,
            "pose_loss": pose_loss_full,
            "kobj_loss": kobj_loss_full,
        }

        if self.training:
            cls_weight = self.hyp.cls if hasattr(self.hyp, "cls") else self.hyp["cls"]
            box_weight = self.hyp.box if hasattr(self.hyp, "box") else self.hyp["box"]
            dfl_weight = self.hyp.dfl if hasattr(self.hyp, "dfl") else self.hyp["dfl"]
            pose_weight = self.hyp.pose if hasattr(self.hyp, "pose") else self.hyp["pose"]
            kobj_weight = self.hyp.kobj if hasattr(self.hyp, "kobj") else self.hyp["kobj"]
            losses["loss"] = (
                cls_weight * cls_loss
                + box_weight * box_loss_full
                + dfl_weight * dfl_loss_full
                + pose_weight * pose_loss_full
                + kobj_weight * kobj_loss_full
            )

        return losses

    def make_anchors(self, feats):
        # Helper to avoid importing at top-level to keep parity with ultralytics.utils.loss
        from ultralytics.utils.tal import make_anchors

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        return anchor_points, stride_tensor


class TLCv8PoseLoss(v8PoseLoss):
    """Pose loss that prefers dataset-provided OKS sigmas when available.

    If the attached model has an attribute `oks_sigmas` (list/ndarray/tensor), use it to
    configure the `KeypointLoss`. Falls back to the default behavior otherwise.
    """

    def __init__(self, model):
        super().__init__(model)
        oks_sigmas = getattr(model, "oks_sigmas", None)
        if oks_sigmas is not None:
            sigmas_tensor = torch.as_tensor(oks_sigmas, device=self.device, dtype=torch.float32)
            self.keypoint_loss = KeypointLoss(sigmas=sigmas_tensor)

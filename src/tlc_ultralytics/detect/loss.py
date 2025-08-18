from __future__ import annotations

import torch
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist, make_anchors


class UnreducedBboxLoss(BboxLoss):
    """BboxLoss that returns unreduced losses for per-sample computation."""

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: float,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing IoU and DFL losses.

        :param pred_dist: Predicted distribution tensor
        :param pred_bboxes: Predicted bounding boxes
        :param anchor_points: Anchor points for the predictions
        :param target_bboxes: Target bounding boxes
        :param target_scores: Target scores
        :param target_scores_sum: Sum of target scores
        :param fg_mask: Foreground mask
        :return: Tuple of (iou_loss, dfl_loss) tensors
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = (1.0 - iou) * weight

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
        else:
            loss_dfl = torch.zeros_like(loss_iou)

        assert loss_iou.shape == loss_dfl.shape, f"IoU Loss shape {loss_iou.shape} != DFL Loss shape {loss_dfl.shape}"

        return loss_iou, loss_dfl


class v8UnreducedDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss that returns unreduced losses for per-sample computation."""

    def __init__(self, model, tal_topk: int = 10, training: bool = False):
        """Initialize the unreduced detection loss.

        :param model: The YOLO model
        :param tal_topk: Top-k for TAL assignment
        :param training: Whether in training mode
        """
        super().__init__(model, tal_topk=tal_topk)

        m = model.model[-1]  # Detect() module
        self.bbox_loss = UnreducedBboxLoss(m.reg_max)
        self.training = training

    def __call__(self, preds, batch) -> dict[str, torch.Tensor]:
        """Calculate unreduced losses for box, cls and dfl.

        :param preds: Model predictions
        :param batch: Batch data
        :return: Dictionary containing unreduced losses
        """
        feats = preds[1] if isinstance(preds, (list, tuple)) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        cls_loss = self.bce(pred_scores, target_scores.to(dtype)).sum(dim=2)

        # Bbox loss
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
                target_scores_sum,
                fg_mask,
            )
            box_loss_full[fg_mask] = box_loss.to(cls_loss.dtype).squeeze()
            dfl_loss_full[fg_mask] = dfl_loss.to(cls_loss.dtype).squeeze()

        losses = {
            "cls_loss": cls_loss,
            "box_loss": box_loss_full,
            "dfl_loss": dfl_loss_full,
        }

        if self.training:
            cls_weight = self.hyp.cls if hasattr(self.hyp, "cls") else self.hyp["cls"]
            box_weight = self.hyp.box if hasattr(self.hyp, "box") else self.hyp["box"]
            dfl_weight = self.hyp.dfl if hasattr(self.hyp, "dfl") else self.hyp["dfl"]
            losses["loss"] = cls_loss * cls_weight + box_loss_full * box_weight + dfl_loss_full * dfl_weight

        return losses

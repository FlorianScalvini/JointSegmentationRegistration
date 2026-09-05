"""
PyTorch Dataset classes for spatio-temporal longitudinal brain MRI sequences.

Each dataset wraps a list of per-subject session lists, where every session
entry is a ``(image_path, seg_path, age)`` triplet.  At index time the
datasets load all sessions for a given subject, optionally apply spatial
transforms, sort by acquisition age, and return stacked tensors ready for
the registration model.

Two variants are provided:

* :class:`SpatioTemporalDataset` — training dataset.  Filters out subjects
  with fewer than two sessions (single time-points cannot be used for
  longitudinal registration), loads SDF maps alongside images and
  segmentations, and sorts sessions by age before stacking.
* :class:`SpatioTemporalDatasetValidation` — validation / test dataset.
  Simpler pipeline without SDF loading; exposes :meth:`get_subject` so the
  test loop can retrieve the original TorchIO subject (with affine) for
  NIfTI export.

Author : Florian Scalvini
"""

# --- Third-party ---
import torch
import torchio as tio
from torchio import transforms
import random
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Training dataset
# ──────────────────────────────────────────────────────────────────────────────

class SpatioTemporalDataset(torch.utils.data.Dataset):
    """Longitudinal MRI dataset for training.

    Loads image volumes, segmentation label maps, signed-distance-function
    (SDF) maps, and acquisition ages for each subject.  Subjects with fewer
    than two sessions are silently discarded because at least two time-points
    are required for longitudinal registration.  Sessions are sorted by age
    before being stacked into tensors.

    Parameters
    ----------
    data : list
        Outer list — one entry per subject.  Inner list — one entry per
        session, each being a ``[image_path, seg_path, age]`` triplet.
        *seg_path* may be ``None`` if no segmentation is available.
    transform : transforms.Transform or None
        Spatial transform applied to each image volume independently.
    transform_seg : transforms.Transform or None
        Spatial transform applied to segmentation and SDF maps independently.
    """

    def __init__(
        self,
        data: list,
        transform: transforms.Transform | None = None,
        transform_aug: transforms.Transform | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.transform_aug = transform_aug
        self.data: list = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                self.data.append(data[i][j])
        
    def __len__(self) -> int:
        """Return the number of subjects in the dataset."""
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all sessions for subject *idx* sorted by age.

        Parameters
        ----------
        idx : int
            Subject index.

        Returns
        -------
        mri_stack_out : torch.Tensor
            Stacked MRI volumes of shape ``(T, 1, X, Y, Z)``.
        seg_stack_out : torch.Tensor
            Stacked segmentation label maps of shape ``(T, 1, X, Y, Z)``.
        time_stack_out : torch.Tensor
            Acquisition ages of shape ``(T,)``.
        """
        mri_stack = []
        seg_stack = []
        time_stack = []
        data = self.data[idx]
     
        session = tio.Subject(
            image=tio.ScalarImage(data[0]),
            label=tio.LabelMap(data[1]) if data[1] is not None else None,
        )
        if self.transform is not None:
            session = self.transform(session) # type: ignore
        if self.transform_aug is not None:
            session = self.transform_aug(session)  # type: ignore

        mri_stack.append(session.image.data)
        if session.label is not None:
            labels = session.label.data
            labels = labels - 1
            labels[labels < 0] = 0
            seg_stack.append(labels)
        time_stack.append(data[2])
        del session

        # ── 5. stack ──────────────────────────────────────────────────
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)
        seg_stack_out = torch.stack(seg_stack, dim=0)  # (T_total, 1, X, Y, Z)
        time_stack_out = torch.tensor(time_stack, dtype=torch.float)  # (T_total,)
        return mri_stack_out, seg_stack_out, time_stack_out


# ──────────────────────────────────────────────────────────────────────────────
#  Validation / test dataset
# ──────────────────────────────────────────────────────────────────────────────

class SpatioTemporalDatasetValidation(torch.utils.data.Dataset):
    """Longitudinal MRI dataset for validation and testing.

    Lighter variant of :class:`SpatioTemporalDataset` that omits SDF loading
    and keeps all subjects regardless of session count.  Exposes
    :meth:`get_subject` so the test loop can retrieve the full TorchIO
    subject (with affine matrix) for NIfTI-format saving.

    Parameters
    ----------
    data : list
        Outer list — one entry per subject.  Inner list — one entry per
        session, each being a ``[image_path, seg_path, age]`` triplet.
    transform : transforms.Transform or None
        Spatial transform applied to each full subject at load time.
    transform_seg : transforms.Transform or None
        Spatial transform applied to segmentation maps (reserved for API
        consistency; not applied inside ``__getitem__``).
    reverse_transform : transforms.Transform or None
        Inverse spatial transform used to map predictions back to the
        original subject space (e.g. :class:`tio.CropOrPad`).
    """

    def __init__(
        self,
        data: list,
        transform: transforms.Transform | None = None,
        reverse_transform: transforms.Transform | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.reverse_transform = reverse_transform
        self.data = data

    def __len__(self) -> int:
        """Return the number of subjects in the dataset."""
        return len(self.data)

    def get_reverse_transform(self) -> transforms.Transform | None:
        """Return the inverse spatial transform, or ``None`` if not set."""
        return self.reverse_transform

    def get_subject(self, idx: int, time_idx: int = 0) -> tio.Subject:
        """Return the raw TorchIO subject at *idx* without applying transforms.

        Parameters
        ----------
        idx : int
            Subject index.

        Returns
        -------
        tio.Subject
            Subject loaded from disk with ``image`` and (optionally) ``label``
            fields, preserving the original affine for NIfTI export.
        """
        data = self.data[idx]
        session = tio.Subject(
            image=tio.ScalarImage(data[time_idx][0]),
            label=tio.LabelMap(data[time_idx][1]) if data[time_idx][1] is not None else None
        )
        return session

    def restore_prediction(
        self, prediction: torch.Tensor, subject_idx: int, time_idx: int
    ) -> tio.LabelMap:
        """Restore a model-grid label map to the original NIfTI geometry."""
        original = self.get_subject(subject_idx, time_idx)
        original_shape = original.image.spatial_shape
        original_affine = original.image.affine.copy()
        processed = self.transform(original) if self.transform is not None else original
        prediction_image = tio.LabelMap(
            tensor=prediction.cpu(), affine=processed.image.affine
        )

        if self.transform is not None:
            crop_shape = self.transform.transforms[0].target_shape
            inverse_spatial = tio.transforms.Compose([
                tio.transforms.Resize(crop_shape),
                tio.transforms.CropOrPad(original_shape),
            ])
            prediction_image = inverse_spatial(prediction_image)

        # Copy the source affine explicitly to avoid small numerical changes
        # introduced by the resize/crop operations.
        return tio.LabelMap(
            tensor=prediction_image.data,
            affine=original_affine,
        )

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all sessions for subject *idx* sorted by age.

        Parameters
        ----------
        idx : int
            Subject index.

        Returns
        -------
        mri_stack_out : torch.Tensor
            Stacked MRI volumes of shape ``(T, 1, X, Y, Z)``.
        seg_stack_out : torch.Tensor
            Stacked segmentation label maps of shape ``(T, 1, X, Y, Z)``.
        time_stack_out : torch.Tensor
            Acquisition ages of shape ``(T,)``.
        """
        mri_stack = []
        seg_stack = []
        time_stack = []
        data = self.data[idx]
        for i in range(len(data)):
            session = tio.Subject(
                image=tio.ScalarImage(data[i][0]),
                label=tio.LabelMap(data[i][1]) if data[i][1] is not None else None,
            )
            if self.transform is not None:
                session = self.transform(session) # type: ignore
            mri_stack.append(session.image.data)
            if session.label is not None:
                labels = session.label.data
                labels = labels - 1
                labels[labels < 0] = 0
                seg_stack.append(labels)
            time_stack.append(data[i][2])
            del session

        # ── 5. stack ──────────────────────────────────────────────────
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)
        seg_stack_out = torch.stack(seg_stack, dim=0)  # (T_total, 1, X, Y, Z)
        time_stack_out = torch.tensor(time_stack, dtype=torch.float)  # (T_total,)

        return mri_stack_out, seg_stack_out, time_stack_out

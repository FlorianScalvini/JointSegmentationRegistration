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
  longitudinal registration), loads images and segmentations, and sorts
  sessions by age before stacking.
* :class:`SpatioTemporalDatasetValidation` — validation / test dataset.
  Exposes :meth:`get_subject` so the
  test loop can retrieve the original TorchIO subject (with affine) for
  NIfTI export.

Author : Florian Scalvini
"""

# --- Third-party ---
import numpy as np
import torch
import torchio as tio
from torchio import transforms


def merge_background_labels(labels: torch.Tensor) -> torch.Tensor:
    """Map raw labels {0, 1, 2, 3, 4} to training labels {0, 1, 2, 3}."""
    labels = labels.long()
    return torch.where(labels <= 1, torch.zeros_like(labels), labels - 1)


# ──────────────────────────────────────────────────────────────────────────────
#  Training dataset
# ──────────────────────────────────────────────────────────────────────────────

class SpatioTemporalDataset(torch.utils.data.Dataset):
    """Longitudinal MRI dataset for training.

    Loads image volumes, segmentation label maps, and acquisition ages for
    each subject.  Subjects with fewer
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
        Spatial transform applied to segmentation maps independently.
    """

    def __init__(
        self,
        data: list,
        data_pretrain: list
    ) -> None:
        super().__init__()
        self.data: list = []
        self.data_pretrain: list = data_pretrain
        for i in range(len(data)):
            if len(data[i]) >= 2:
                self.data.append(data[i])

    def __len__(self) -> int:
        """Return the number of subjects in the dataset."""
        return len(self.data)
    
    def get_random_pretrain_data_session(self) -> tuple[torch.Tensor, torch.Tensor]:
        random_subject_idx = int(torch.randint(0, len(self.data_pretrain), (1,)).item())
        random_session_idx = int(torch.randint(0, len(self.data_pretrain[random_subject_idx]), (1,)).item())
        data = self.data_pretrain[random_subject_idx][random_session_idx]
        session = tio.Subject(
                image=tio.ScalarImage(data[0]),
                label=tio.LabelMap(data[1])
        )
        return session.image.data, merge_background_labels(session.label.data)

    def __getitem__(
        self, idx: int
    ) -> tuple:
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
        anchor_mri : torch.Tensor
            Random image from the independent labeled pretraining database.
        anchor_seg : torch.Tensor
            Ground-truth label map paired with ``anchor_mri``.
        """
        mri_stack = []
        time_stack = []
        seg_stack = []
        data = self.data[idx]
        for i in range(len(data)):
            session = tio.Subject(
                image=tio.ScalarImage(data[i][0]),
                label=tio.LabelMap(data[i][1])
            )
            mri_stack.append(session.image.data)
            seg_stack.append(merge_background_labels(session.label.data))
            time_stack.append(data[i][2])
            del session

        # ── 5. stack ──────────────────────────────────────────────────
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)
        seg_stack_out = torch.stack(seg_stack, dim=0)  # (T_total, 1, X, Y, Z)
        time_stack_out = torch.tensor(time_stack, dtype=torch.float)  # (T_total,)
        anchor_mri, anchor_seg = self.get_random_pretrain_data_session()

        return mri_stack_out, seg_stack_out, time_stack_out, anchor_mri, anchor_seg

# ──────────────────────────────────────────────────────────────────────────────
#  Validation / test dataset
# ──────────────────────────────────────────────────────────────────────────────

class SpatioTemporalDatasetValidation(torch.utils.data.Dataset):
    """Longitudinal MRI dataset for validation and testing.

    Validation variant that keeps all subjects regardless of session count. Exposes
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
        data: list
    ) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        """Return the number of subjects in the dataset."""
        return len(self.data)
    

    def get_subject(self, subject_idx: int, time_idx: int) -> tio.Subject:
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
        data = self.data[subject_idx]
        session = tio.Subject(
            image=tio.ScalarImage(data[time_idx][0]),
            label=tio.LabelMap(data[time_idx][1]) if data[time_idx][1] is not None else None
        )
        return session

    def __getitem__(
        self, idx: int
    ) -> tuple:
        """Return all sessions for subject *idx* as stacked tensors.

        Parameters
        ----------
        idx : int
            Subject index.

        Returns
        -------
        mri_stack_out : torch.Tensor
            Stacked MRI volumes of shape ``(T, 1, X, Y, Z)``.
        seg_stack_out : torch.Tensor
            Stacked segmentation label maps of shape ``(T, 1, X, Y, Z)``,
            or an empty tensor if no labels are available.
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
            mri_stack.append(session.image.data)
            if session.label is not None:
                seg_stack.append(merge_background_labels(session.label.data))
            time_stack.append(data[i][2])
            del session
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)

        if len(seg_stack) > 0:
            seg_stack_out = torch.stack(seg_stack, dim=0)  # (T_total, 1, X, Y, Z)
        else:
            seg_stack_out = torch.empty(0)

        time_stack_out = torch.tensor(time_stack, dtype=torch.float)  # (T_total,)
        return mri_stack_out, seg_stack_out, time_stack_out

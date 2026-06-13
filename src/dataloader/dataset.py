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
        augmentation: bool = False,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.augmentation = augmentation
        self.data: list = []
        for i in range(len(data)):
            if len(data[i]) >= 2:
                self.data.append(data[i])

    def __len__(self) -> int:
        """Return the number of subjects in the dataset."""
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        time_stack = []
        data = self.data[idx]
        for i in range(len(data)):
            session = tio.Subject(
                image=tio.ScalarImage(data[i][0]),
            )
            if self.transform is not None:
                session.image = self.transform(session.image) # type: ignore

            mri_stack.append(session.image.data)
            time_stack.append(data[i][2])
            del session

        # ── 5. stack ──────────────────────────────────────────────────
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)
        time_stack_out = torch.tensor(time_stack, dtype=torch.float)  # (T_total,)

        return mri_stack_out, time_stack_out


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
        transform_seg: transforms.Transform | None = None,
        reverse_transform: transforms.Transform | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.transform_seg = transform_seg
        self.reverse_transform = reverse_transform
        self.data = data

    def __len__(self) -> int:
        """Return the number of subjects in the dataset."""
        return len(self.data)

    def get_reverse_transform(self) -> transforms.Transform | None:
        """Return the inverse spatial transform, or ``None`` if not set."""
        return self.reverse_transform

    def get_subject(self, idx: int) -> tio.Subject:
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
            image=tio.ScalarImage(data[0][0]),
            label=tio.LabelMap(data[0][1]) if data[0][1] is not None else None
        )
        return session

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            if self.transform is not None:
                session = self.transform(session)

            mri_stack.append(session.image.data)
            if session.label is not None:
                seg_stack.append(session.label.data)
            time_stack.append(data[i][2])
            del session
        mri_stack_out = torch.stack(mri_stack, dim=0)  # (T_total, 1, X, Y, Z)

        if len(seg_stack) > 0:
            seg_stack_out = torch.stack(seg_stack, dim=0)  # (T_total, 1, X, Y, Z)
        else:
            seg_stack_out = torch.empty(0)

        time_stack_out = torch.tensor(time_stack, dtype=torch.float)  # (T_total,)
        return mri_stack_out, seg_stack_out, time_stack_out

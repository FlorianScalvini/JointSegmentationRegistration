"""
NIfTI segmentation → VTK surface-mesh conversion utility.

Reads a labelled NIfTI volume (.nii / .nii.gz) and extracts a smooth,
world-space surface mesh for one or more label values, then writes the
result as a VTK unstructured grid (.vtk).

Three main modes are available:

* **Single label** – extract the iso-surface for one integer label value
  (``convert_nifti_label_to_vtk``).
* **Multi-label union** – fuse several labels into a single binary mask and
  extract the combined surface (``convert_nifti_labels_union_to_vtk``).
* **All labels** – batch-export every non-zero label present in the volume
  to separate VTK files (``export_all_labels_to_vtk``).

Surface extraction can use either *Flying Edges* (fast, per-label) or
*Surface Nets* (smoother shared boundaries).  Optional Windowed-Sinc
smoothing and Pro-decimation are applied as post-processing steps.

Usage::

    python nifti_to_vtk.py --nifti /path/to/parcellation.nii.gz \\
                         --labels 3 4 --method surface_nets --smoothing 0.6

Author : Florian Scalvini
"""

# --- Standard library ---
import math
import glob
import argparse

# --- Third-party ---
import vtk


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clone_identity(img: vtk.vtkImageData) -> vtk.vtkImageData:
    """Return a shallow copy of *img* with identity origin and unit spacing."""
    out = vtk.vtkImageData()
    out.ShallowCopy(img)
    out.SetOrigin(0,0,0)
    out.SetSpacing(1,1,1)
    return out

def _wsinc_params(s: float) -> tuple[float, int]:
    """Derive Windowed-Sinc passband and iteration count from smoothing factor *s*."""
    # passband = 10^(-4*s), iterations = 20 + 40*s
    return 10.0**(-4.0*s), int(round(20 + 40*s))

def _surf_nets_iters(s: float) -> int:
    """Return the SurfaceNets internal smoothing iteration count for factor *s*."""
    return int(math.floor(15.0*s*s + 9.0*s))

def _gen_surface_ijk(
    img: vtk.vtkImageData,
    labels: list[int],
    method: str = "flying_edges",
    smoothing_factor: float = 0.5,
    surf_nets_internal: bool = False,
    decimation: float = 0.0,
) -> vtk.vtkPolyData:
    """Extract an iso-surface in IJK space, then optionally decimate and smooth it."""
    ijk = _clone_identity(img)

    if method == "flying_edges":
        fe = vtk.vtkDiscreteFlyingEdges3D()
        fe.SetInputData(ijk)
        fe.ComputeGradientsOff()
        fe.ComputeNormalsOff()
        for i,v in enumerate(labels):
            fe.SetValue(i, int(v))
        fe.Update()
        surf = fe.GetOutput()
    elif method == "surface_nets":
        sn = vtk.vtkSurfaceNets3D()
        sn.SetInputData(ijk)
        sn.SmoothingOff()
        if surf_nets_internal:
            sn.SmoothingOn()
            sn.SetNumberOfIterations(_surf_nets_iters(smoothing_factor))
        for i,v in enumerate(labels):
            sn.SetValue(i, int(v))
        sn.Update()
        surf = sn.GetOutput()
    else:
        raise ValueError("method must be 'flying_edges' or 'surface_nets'")

    if surf.GetNumberOfPolys()==0:
        empty = vtk.vtkPolyData(); empty.Initialize(); return empty

    if decimation and decimation>0.0:
        dec = vtk.vtkDecimatePro()
        dec.SetInputData(surf)
        dec.SetFeatureAngle(60); dec.SplittingOff(); dec.PreserveTopologyOn(); dec.SetMaximumError(1)
        dec.SetTargetReduction(float(decimation))
        dec.Update()
        surf = dec.GetOutput()

    if smoothing_factor>0.0 and not surf_nets_internal:
        pb, iters = _wsinc_params(smoothing_factor)
        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputData(surf)
        sm.SetNumberOfIterations(iters)
        sm.SetPassBand(pb)
        sm.BoundarySmoothingOff(); sm.FeatureEdgeSmoothingOff()
        sm.NonManifoldSmoothingOn(); sm.NormalizeCoordinatesOn()
        sm.Update()
        surf = sm.GetOutput()

    return surf

def _ijk_to_world(
    surface_ijk: vtk.vtkPolyData,
    img_from_reader: vtk.vtkImageData,
) -> vtk.vtkPolyData:
    """Apply the image affine (direction × spacing + origin) to map IJK → world coords."""
    # Try direction matrix (VTK 9), else build from origin/spacing
    m = vtk.vtkMatrix4x4(); m.Identity()
    if hasattr(img_from_reader, "GetDirectionMatrix"):
        dm = img_from_reader.GetDirectionMatrix()
        sx,sy,sz = img_from_reader.GetSpacing()
        for r in range(3):
            for c in range(3):
                m.SetElement(r, c, dm.GetElement(r,c) * (sx if c==0 else sy if c==1 else sz))
        ox,oy,oz = img_from_reader.GetOrigin()
        m.SetElement(0,3,ox); m.SetElement(1,3,oy); m.SetElement(2,3,oz)
    else:
        sx,sy,sz = img_from_reader.GetSpacing()
        ox,oy,oz = img_from_reader.GetOrigin()
        m.SetElement(0,0,sx); m.SetElement(1,1,sy); m.SetElement(2,2,sz)
        m.SetElement(0,3,ox); m.SetElement(1,3,oy); m.SetElement(2,3,oz)

    xf = vtk.vtkTransform(); xf.SetMatrix(m)
    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(surface_ijk); tf.SetTransform(xf); tf.Update()
    return tf.GetOutput()

def _finish(
    surface_world: vtk.vtkPolyData,
    compute_normals: bool,
    method: str,
) -> vtk.vtkPolyData:
    """Optionally compute consistent normals and strip internal scalar arrays."""
    out = vtk.vtkPolyData()
    if compute_normals and method=="flying_edges":
        n = vtk.vtkPolyDataNormals()
        n.SetInputData(surface_world)
        n.ConsistencyOn(); n.SplittingOff()
        n.Update()
        out.ShallowCopy(n.GetOutput())
    else:
        out.ShallowCopy(surface_world)
    pd = out.GetPointData()
    if pd is not None:
        pd.RemoveArray("ImageScalars")
    return out

def _fuse_labels_to_binary(
    img: vtk.vtkImageData,
    labels: list[int],
) -> vtk.vtkImageData:
    """OR-fuse *labels* into a single unsigned-char binary mask (value = 1)."""
    labels = [int(v) for v in labels]
    labels = sorted(set(labels))
    # start with first mask
    th = vtk.vtkImageThreshold()
    th.SetInputData(img)
    th.ThresholdBetween(labels[0], labels[0])
    th.SetInValue(1); th.SetOutValue(0)
    th.SetOutputScalarTypeToUnsignedChar()
    th.Update()
    merged = th.GetOutput()
    # OR the rest
    for val in labels[1:]:
        t2 = vtk.vtkImageThreshold()
        t2.SetInputData(img)
        t2.ThresholdBetween(val, val)
        t2.SetInValue(1); t2.SetOutValue(0)
        t2.SetOutputScalarTypeToUnsignedChar()
        t2.Update()
        logic = vtk.vtkImageLogic()
        logic.SetInput1Data(merged); logic.SetInput2Data(t2.GetOutput())
        logic.SetOperationToOr(); logic.SetOutputTrueValue(1)
        logic.Update()
        merged = logic.GetOutput()
    return merged

def _write_binary_mask_nifti(
    mask_uc: vtk.vtkImageData,
    ref_img: vtk.vtkImageData,
    out_path: str,
) -> None:
    """Save the fused binary mask (unsigned char) as NIfTI, copying geometry."""
    # Ensure geometry matches reference
    mask_uc.SetOrigin(ref_img.GetOrigin())
    mask_uc.SetSpacing(ref_img.GetSpacing())
    if hasattr(ref_img, "GetDirectionMatrix") and hasattr(mask_uc, "SetDirectionMatrix"):
        mask_uc.SetDirectionMatrix(ref_img.GetDirectionMatrix())

    w = vtk.vtkNIFTIImageWriter()
    w.SetFileName(out_path)
    w.SetInputData(mask_uc)

    # Build sform/qform from direction+spacing+origin (good enough for most pipelines)
    m = vtk.vtkMatrix4x4(); m.Identity()
    if hasattr(ref_img, "GetDirectionMatrix"):
        dm = ref_img.GetDirectionMatrix()
        sx,sy,sz = ref_img.GetSpacing()
        for r in range(3):
            for c in range(3):
                m.SetElement(r, c, dm.GetElement(r,c) * (sx if c==0 else sy if c==1 else sz))
        ox,oy,oz = ref_img.GetOrigin()
        m.SetElement(0,3,ox); m.SetElement(1,3,oy); m.SetElement(2,3,oz)
    w.SetSFormMatrix(m)
    w.SetQFormMatrix(m)

    w.Write()
    print(f"Saved fused mask NIfTI: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def _polydata_to_unstructured_grid(poly: vtk.vtkPolyData) -> vtk.vtkUnstructuredGrid:
    """Triangulate *poly* and wrap it in a vtkUnstructuredGrid."""
    # Triangulate (safer for conversion)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()

    # AppendFilter converts any vtkDataSet to vtkUnstructuredGrid
    app = vtk.vtkAppendFilter()
    app.AddInputData(tri.GetOutput())
    app.Update()

    ug = vtk.vtkUnstructuredGrid()
    ug.ShallowCopy(app.GetOutput())
    return ug

def cast_all_arrays_to_float32(ug: vtk.vtkUnstructuredGrid) -> None:
    """Cast every point- and cell-data array in *ug* to float32 in-place."""
    pd = ug.GetPointData()
    cd = ug.GetCellData()
    for data in (pd, cd):
        names = [data.GetArrayName(i) for i in range(data.GetNumberOfArrays())]
        arrays = [data.GetArray(i) for i in range(data.GetNumberOfArrays())]
        # remove all first to avoid index shifts
        for name in names:
            data.RemoveArray(name)
        # re-add cast copies
        for name, arr in zip(names, arrays):
            fa = vtk.vtkFloatArray()
            fa.SetName(name if name is not None else "")
            fa.SetNumberOfComponents(arr.GetNumberOfComponents())
            fa.SetNumberOfTuples(arr.GetNumberOfTuples())
            for i in range(arr.GetNumberOfTuples()):
                fa.SetTuple(i, arr.GetTuple(i))
            data.AddArray(fa)

def convert_nifti_label_to_vtk(
    nii_path: str,
    label_value: int,
    out_vtk: str,
    decimation_factor: float = 0.0,
    smoothing_factor: float = 0.5,
    compute_surface_normals: bool = True,
    method: str = "flying_edges",
    surf_nets_internal_smoothing: bool = False,
) -> None:
    """Extract the surface for a single label and write it to a VTK file."""
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nii_path); reader.Update()
    img = reader.GetOutput()

    surf_ijk = _gen_surface_ijk(
        img, [int(label_value)],
        method=method,
        smoothing_factor=smoothing_factor,
        surf_nets_internal=surf_nets_internal_smoothing,
        decimation=decimation_factor
    )
    surf_world = _ijk_to_world(surf_ijk, img)
    final = _finish(surf_world, compute_surface_normals, method)


    ug = _polydata_to_unstructured_grid(final)
    cast_all_arrays_to_float32(ug)
    w = vtk.vtkUnstructuredGridWriter()
    w.SetFileName(out_vtk)
    w.SetInputData(ug)
    w.Write()

def convert_nifti_labels_union_to_vtk(
    nii_path: str,
    labels: list[int],
    out_vtk: str,
    out_mask_nifti: str | None = None,
    new_label_value: int = 1,
    decimation_factor: float = 0.0,
    smoothing_factor: float = 0.5,
    compute_surface_normals: bool = True,
    method: str = "flying_edges",
    surf_nets_internal_smoothing: bool = False,
) -> None:
    """Fuse *labels* into a binary mask, extract the union surface, and write VTK."""
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nii_path); reader.Update()
    img = reader.GetOutput()

    merged_mask_uc = _fuse_labels_to_binary(img, labels)

    # optionally write the fused binary mask to disk
    if out_mask_nifti:
        _write_binary_mask_nifti(merged_mask_uc, img, out_mask_nifti)

    # map 1 -> new_label_value for surface extraction
    cast = vtk.vtkImageShiftScale()
    cast.SetInputData(merged_mask_uc)
    cast.SetShift(0.0); cast.SetScale(float(new_label_value))
    cast.SetOutputScalarTypeToUnsignedShort()
    cast.Update()
    fused_labelmap = cast.GetOutput()

    # preserve geometry (again, for safety)
    fused_labelmap.SetOrigin(img.GetOrigin())
    fused_labelmap.SetSpacing(img.GetSpacing())
    if hasattr(img, "GetDirectionMatrix") and hasattr(fused_labelmap, "SetDirectionMatrix"):
        fused_labelmap.SetDirectionMatrix(img.GetDirectionMatrix())

    # extract single union surface
    surf_ijk = _gen_surface_ijk(
        fused_labelmap, [new_label_value],
        method=method,
        smoothing_factor=smoothing_factor,
        surf_nets_internal=surf_nets_internal_smoothing,
        decimation=decimation_factor
    )
    surf_world = _ijk_to_world(surf_ijk, fused_labelmap)
    final = _finish(surf_world, compute_surface_normals, method)

    ug = _polydata_to_unstructured_grid(final)
    cast_all_arrays_to_float32(ug)
    w = vtk.vtkUnstructuredGridWriter()
    w.SetFileName(out_vtk)              # keep .vtk
    w.SetInputData(ug)
    w.Write()

def export_all_labels_to_vtk(nii_path: str, out_prefix: str, **kwargs) -> None:
    """Detect all non-zero labels in *nii_path* and export each to a separate VTK file."""
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nii_path); reader.Update()
    img = reader.GetOutput()

    acc = vtk.vtkImageAccumulate()
    acc.SetInputData(img)
    acc.IgnoreZeroOn()
    rng = img.GetScalarRange()
    low, high = int(math.floor(rng[0])), int(math.ceil(rng[1]))
    acc.SetComponentOrigin(0,0,0)
    acc.SetComponentSpacing(1,1,1)
    acc.SetComponentExtent(low, high, 0,0,0,0)
    acc.Update()
    scal = acc.GetOutput().GetPointData().GetScalars()
    labels = [v for i,v in enumerate(range(low, high+1)) if scal.GetTuple1(i) > 0]

    for lv in labels:
        out = f"{out_prefix}_label{lv}.vtk"
        convert_nifti_label_to_vtk(nii_path, lv, out, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="NIfTI segmentation (.nii/.nii.gz) → VTK (Slicer-like), with label union support")
    p.add_argument("--nifti", required=False, help="Path to NIfTI segmentation (labeled or binary)",
                   default="/home/florian/Documents/Dataset/dHCP/parcellations")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--label", type=int, help="Single label value to extract", default=2)
    g.add_argument("--labels", type=int, nargs="+", help="List of label values to fuse (union), e.g. --labels 2 3 4", default=[3, 4])
    p.add_argument("--out", default="./segment.vtk", help="Output VTK path")
    p.add_argument("--out_mask", default=None, help="Optional path to save fused binary mask NIfTI (only with --labels)")
    p.add_argument("--method", choices=["flying_edges","surface_nets"], default="surface_nets")
    p.add_argument("--smoothing", type=float, default=0.5, help="0..1 (default 0.5)")
    p.add_argument("--decimate", type=float, default=0.0, help="Target reduction 0..1 (default 0)")
    p.add_argument("--internal_surfnet_smoothing", default=False,
                   help="Use SurfaceNets internal smoothing (only with --method surface_nets)")
    p.add_argument("--normals",  default=True, help="Compute normals (used with FlyingEdges)")
    p.add_argument("--all_labels", action="store_true", help="Export all labels to separate VTKs")
    args = p.parse_args()
    files = glob.glob(args.nifti + "/*")
    print("Files to process:", files)
    for file in files:
        if args.all_labels:
            export_all_labels_to_vtk(
                file, (file.rsplit(".")[0] + ".vtk"),
                decimation_factor=args.decimate,
                smoothing_factor=args.smoothing,
                compute_surface_normals=args.normals,
                method=args.method,
                surf_nets_internal_smoothing=args.internal_surfnet_smoothing
            )
        elif args.labels:
            path_out = file.replace(".nii.gz", ".vtk")
            convert_nifti_labels_union_to_vtk(
                file, args.labels, path_out,
                out_mask_nifti=args.out_mask,
                decimation_factor=args.decimate,
                smoothing_factor=args.smoothing,
                compute_surface_normals=args.normals,
                method=args.method,
                surf_nets_internal_smoothing=args.internal_surfnet_smoothing
            )
        elif args.label is not None:
            convert_nifti_label_to_vtk(
                file, args.label, (file.rsplit(".")[0] + ".vtk"),
                decimation_factor=args.decimate,
                smoothing_factor=args.smoothing,
                compute_surface_normals=args.normals,
                method=args.method,
                surf_nets_internal_smoothing=args.internal_surfnet_smoothing
            )
        else:
            raise SystemExit("Please provide either --label or --labels.")

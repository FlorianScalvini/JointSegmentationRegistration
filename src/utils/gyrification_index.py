"""
Gyrification Index (GI) computation for 3-D brain surface meshes.

The Gyrification Index measures the degree of cortical folding as the ratio
of the total folded surface area to the area of the smooth (convex-hull)
reference surface:

    GI = Area(folded mesh) / Area(smooth reference mesh)

A GI of 1 means a perfectly smooth surface; higher values indicate more
folding, as observed during brain development.

Two variants are provided:

* **STL / meshio** (primary) – works with any mesh format supported by
  ``meshio`` (``.stl``, ``.vtk``, ``.vtu``, …).  These are the functions
  used by the main evaluation pipeline.
* **XDMF / FEniCS** (legacy) – works with FEniCS ``Mesh`` and
  ``BoundaryMesh`` objects loaded from ``.xdmf`` files.

Based on
--------
`FetalFoldSim <https://github.com/annekerachni/FetalFoldSim>`_ by
Anne Kerachni — original GI computation and mesh rescaling logic adapted
from that repository.

Author : Florian Scalvini
"""

# --- Standard library ---
import os
import csv
import glob
import argparse
from typing import Any

# --- Third-party ---
import meshio
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  XDMF / FEniCS variant  (legacy – requires dolfin)
# ──────────────────────────────────────────────────────────────────────────────

def rescale_initial_smooth_mesh_to_folded_mesh_XDMF(
    initial_smooth_mesh: Any,
    folded_mesh: Any,
) -> Any:
    """Rescale a FEniCS smooth reference mesh to match the bounding box of the folded mesh.

    Each axis is scaled independently so that the extent of
    *initial_smooth_mesh* matches that of *folded_mesh* along X, Y and Z.
    This is the FEniCS / XDMF variant; see
    :func:`rescale_initial_smooth_mesh_to_folded_mesh` for the meshio
    equivalent.

    Based on `FetalFoldSim <https://github.com/annekerachni/FetalFoldSim>`_.

    Args:
        initial_smooth_mesh: FEniCS ``Mesh`` representing the convex-hull
            (unfolded) reference surface.
        folded_mesh: FEniCS ``Mesh`` representing the folded brain surface
            at a given developmental time-point.

    Returns:
        The *initial_smooth_mesh* with its coordinates rescaled in-place.
    """
    L1 = (max(folded_mesh.coordinates()[:, 0]) - min(folded_mesh.coordinates()[:, 0])) / (
                max(initial_smooth_mesh.coordinates()[:, 0]) - min(initial_smooth_mesh.coordinates()[:, 0]))
    L2 = (max(folded_mesh.coordinates()[:, 1]) - min(folded_mesh.coordinates()[:, 1])) / (
                max(initial_smooth_mesh.coordinates()[:, 1]) - min(initial_smooth_mesh.coordinates()[:, 1]))
    L3 = (max(folded_mesh.coordinates()[:, 2]) - min(folded_mesh.coordinates()[:, 2])) / (
                max(initial_smooth_mesh.coordinates()[:, 2]) - min(initial_smooth_mesh.coordinates()[:, 2]))
    initial_smooth_mesh.coordinates()[:, 0] = L1 * initial_smooth_mesh.coordinates()[:, 0]
    initial_smooth_mesh.coordinates()[:, 1] = L2 * initial_smooth_mesh.coordinates()[:, 1]
    initial_smooth_mesh.coordinates()[:, 2] = L3 * initial_smooth_mesh.coordinates()[:, 2]

    return initial_smooth_mesh


def compute_gyrification_index_XDMF(
    rescaled_initial_smooth_bmesh: Any,
    folded_bmesh: Any,
) -> float:
    """Compute the Gyrification Index from two FEniCS boundary meshes.

    Iterates over triangle faces of each mesh and accumulates total surface
    area using the cross-product formula.  GI = folded area / convex-hull area.

    This is the FEniCS / XDMF variant; see :func:`compute_gyrification_index`
    for the meshio equivalent.

    Based on `FetalFoldSim <https://github.com/annekerachni/FetalFoldSim>`_.

    Args:
        rescaled_initial_smooth_bmesh: FEniCS ``BoundaryMesh`` of the
            rescaled smooth (convex-hull) surface.
        folded_bmesh: FEniCS ``BoundaryMesh`` of the folded brain surface.

    Returns:
        GI as a float (≥ 1).  Values close to 1 indicate a smooth surface;
        higher values indicate stronger gyrification.
    """
    ### Convex hull
    Area_convex_hull = 0.0  # = area of the initial unfolded mesh

    for face in rescaled_initial_smooth_bmesh.cells():  # e.g. triangle=[0, 2, 4], with 0, 2, 4 node indices
        Ntmp = np.cross(
            rescaled_initial_smooth_bmesh.coordinates()[face[1]] - rescaled_initial_smooth_bmesh.coordinates()[face[0]],
            rescaled_initial_smooth_bmesh.coordinates()[face[2]] - rescaled_initial_smooth_bmesh.coordinates()[face[0]])
        Area_convex_hull += 0.5 * np.linalg.norm(Ntmp)

    ### Folded mesh
    Area_folded_mesh = 0.0

    for face in folded_bmesh.cells():
        Ntmp_2 = np.cross(folded_bmesh.coordinates()[face[1]] - folded_bmesh.coordinates()[face[0]],
                          folded_bmesh.coordinates()[face[2]] - folded_bmesh.coordinates()[face[0]])
        Area_folded_mesh += 0.5 * np.linalg.norm(Ntmp_2)

    GI = Area_folded_mesh / Area_convex_hull

    return float(GI)


# ──────────────────────────────────────────────────────────────────────────────
#  STL / meshio variant  (primary API)
# ──────────────────────────────────────────────────────────────────────────────

def rescale_initial_smooth_mesh_to_folded_mesh(
    initial_smooth_mesh: meshio.Mesh,
    folded_mesh: meshio.Mesh,
) -> meshio.Mesh:
    """Rescale a smooth reference mesh to match the bounding box of the folded mesh.

    Each spatial axis is scaled independently by the ratio of the folded
    mesh extent to the smooth mesh extent, so that both meshes occupy the
    same bounding box before area comparison.

    Based on `FetalFoldSim <https://github.com/annekerachni/FetalFoldSim>`_.

    Args:
        initial_smooth_mesh: ``meshio.Mesh`` representing the unfolded,
            convex-hull reference surface (e.g. a sphere or initial smooth
            brain surface at the start of development).
        folded_mesh: ``meshio.Mesh`` representing the folded brain surface
            at a given developmental time-point.

    Returns:
        The *initial_smooth_mesh* with its ``.points`` array rescaled
        in-place along each axis.
    """
    L1 = (max(folded_mesh.points[:, 0]) - min(folded_mesh.points[:, 0])) / (
                max(initial_smooth_mesh.points[:, 0]) - min(initial_smooth_mesh.points[:, 0]))
    L2 = (max(folded_mesh.points[:, 1]) - min(folded_mesh.points[:, 1])) / (
                max(initial_smooth_mesh.points[:, 1]) - min(initial_smooth_mesh.points[:, 1]))
    L3 = (max(folded_mesh.points[:, 2]) - min(folded_mesh.points[:, 2])) / (
                max(initial_smooth_mesh.points[:, 2]) - min(initial_smooth_mesh.points[:, 2]))
    initial_smooth_mesh.points[:, 0] = L1 * initial_smooth_mesh.points[:, 0]
    initial_smooth_mesh.points[:, 1] = L2 * initial_smooth_mesh.points[:, 1]
    initial_smooth_mesh.points[:, 2] = L3 * initial_smooth_mesh.points[:, 2]

    return initial_smooth_mesh


def compute_gyrification_index(
    rescaled_initial_smooth_bmesh: meshio.Mesh,
    folded_bmesh: meshio.Mesh,
) -> float:
    """Compute the Gyrification Index from two ``meshio`` triangle meshes.

    Iterates over all triangular faces and accumulates surface area using
    the cross-product formula (``area = 0.5 * |v1 × v2|``), then returns
    the ratio of the folded area to the convex-hull area.

    Based on `FetalFoldSim <https://github.com/annekerachni/FetalFoldSim>`_.

    Args:
        rescaled_initial_smooth_bmesh: ``meshio.Mesh`` of the rescaled smooth
            (convex-hull) reference surface.  Must contain a ``'triangle'``
            cell block.
        folded_bmesh: ``meshio.Mesh`` of the folded brain surface at a given
            developmental time-point.  Must contain a ``'triangle'`` cell
            block.

    Returns:
        GI as a float (≥ 1).  Values close to 1 indicate a smooth surface;
        higher values reflect stronger gyrification.
    """
    ### Convex hull
    Area_convex_hull = 0.0  # = area of the initial unfolded mesh
    """
    Ntmp = np.cross(initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,1]] - initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,0]],
                    initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,2]] - initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,0]])
    """
    for face in rescaled_initial_smooth_bmesh.cells_dict["triangle"]:  # e.g. triangle=[0, 2, 4], with 0, 2, 4 node indices
        Ntmp = np.cross(rescaled_initial_smooth_bmesh.points[face[1]] - rescaled_initial_smooth_bmesh.points[face[0]],
                        rescaled_initial_smooth_bmesh.points[face[2]] - rescaled_initial_smooth_bmesh.points[face[0]])
        Area_convex_hull += 0.5 * np.linalg.norm(Ntmp)

    ### Folded mesh
    Area_folded_mesh = 0.0
    """
    Ntmp_2 = np.cross(rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,1]] - rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,0]],
                      rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,2]] - rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,0]])
    """
    for face in folded_bmesh.cells_dict["triangle"]:
        Ntmp_2 = np.cross(folded_bmesh.points[face[1]] - folded_bmesh.points[face[0]],
                          folded_bmesh.points[face[2]] - folded_bmesh.points[face[0]])
        Area_folded_mesh += 0.5 * np.linalg.norm(Ntmp_2)

    GI = Area_folded_mesh / Area_convex_hull

    return float(GI)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(path: str, initial_path: str, age: list[int], name_csv: str) -> None:
    """Batch-compute GI for a directory of STL meshes and write results to CSV.

    Args:
        path: Directory containing the folded brain surface meshes (``.stl``).
        initial_path: Path to the initial smooth (convex-hull) reference mesh.
        age: List of age values corresponding to each STL file (sorted order).
        name_csv: Base name of the output CSV file (without extension).
    """
    stl_files = glob.glob(os.path.join(path, "*.stl"))
    stl_files.sort()
    initial_smooth_mesh = meshio.read(initial_path)
    x = age
    y = np.zeros(len(x))
    print("stl_files:", stl_files)


    header = ["time", "GI"]
    with open(f'./{name_csv}.csv', mode='w', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for i in range(len(stl_files)):
            print(f"Processing {stl_files[i]}")
            folded_mesh = meshio.read(stl_files[i])
            # rescale initial smooth brain mesh onto the folded brain mesh
            rescaled_initial_smooth_mesh = rescale_initial_smooth_mesh_to_folded_mesh(initial_smooth_mesh, folded_mesh)

            # compute gyrification index
            GI = compute_gyrification_index(rescaled_initial_smooth_mesh, folded_mesh)
            row = []
            row.append(x[i])
            row.append(GI)
            writer.writerow(row)
        # export json file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference Registration 3D Images')
    parser.add_argument('--stl_paths', type=str, help='STL Path', required=False, default="")
    parser.add_argument('--initial_path', type=str, help='Baseline', required=False, default="")
    parser.add_argument('--name_csv', type=str, help='Method name', required=False, default="ants_borgne")
    parser.add_argument('--age', type=int, help='ANTS benchmark', required=False, nargs="+", default=[85 ,97, 110, 122, 135, 147, 155])
    args = parser.parse_args()
    ages = args.age

    main(args.stl_paths, args.initial_path, ages, args.name_csv)

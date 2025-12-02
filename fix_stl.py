#!/usr/bin/env python

import argparse

import nksr
import numpy as np
import torch
import trimesh


def stl_to_oriented_points(mesh: trimesh.Trimesh, n_points: int):
    """
    Sample oriented points (xyz + normals) from a mesh surface.
    Normals kommen aus den Facenormalen.
    """
    # Kleine Aufräumrunde
    mesh.remove_unreferenced_vertices()
    # mesh.remove_degenerate_faces()
    mesh.remove_infinite_values()

    # Punkte gleichmäßig über die Fläche sampeln
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)

    face_normals = mesh.face_normals
    normals = face_normals[face_idx]

    return points.astype(np.float32), normals.astype(np.float32)


def reconstruct_with_nksr(
    points: np.ndarray,
    normals: np.ndarray,
    detail_level: float = 1.0,
    mise_iter: int = 1,
    device_str: str | None = None,
) -> trimesh.Trimesh:
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    xyz = torch.from_numpy(points).to(device)
    nrm = torch.from_numpy(normals).to(device)

    reconstructor = nksr.Reconstructor(device)
    reconstructor.chunk_tmp_device = torch.device("cpu")

    # WICHTIG: kein chunk_size hier, normaler Pfad
    field = reconstructor.reconstruct(
        xyz,
        nrm,
        # detail_level=detail_level,
        chunk_size=25.0,
        # preprocess_fn=nksr.get_estimate_normal_preprocess_fn(64, 85.0),
    )

    mesh = field.extract_dual_mesh(mise_iter=mise_iter)
    tm = trimesh.Trimesh(
        vertices=mesh.v,
        faces=mesh.f,
        process=False,  # kein zusätzliches "Aufräumen" durch trimesh
    )
    return tm


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct / 'fix' STL meshes using NKSR (simple, non-chunked)."
    )
    parser.add_argument("--input_stl", help="Path to input STL")
    parser.add_argument("--output_stl", help="Path to output STL")

    parser.add_argument(
        "--points",
        type=int,
        default=150_000,
        help="Number of sampled surface points (default: 150000)",
    )
    parser.add_argument(
        "--detail",
        type=float,
        default=1.0,
        help="NKSR detail_level (higher = finer geometry, slower). Default: 1.0",
    )
    parser.add_argument(
        "--mise_iter",
        type=int,
        default=1,
        help="Dual mesh upsampling iterations (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="e.g. 'cuda:0' or 'cpu' (default: auto)",
    )

    args = parser.parse_args()

    print(f"[+] Loading mesh: {args.input_stl}")
    mesh_in = trimesh.load(args.input_stl, force="mesh")

    # print(f"[+] Sampling surface points (n={args.points})")
    # pts, nrm = stl_to_oriented_points(mesh_in, args.points)
    # if pts.shape[0] == 0:
    #     raise RuntimeError(
    #         "Sampling produced 0 points – STL mesh might be empty or invalid."
    #     )
    pts, nrm = (
        mesh_in.vertices.astype(np.float32),
        mesh_in.vertex_normals.astype(np.float32),
    )

    print(
        f"[+] Reconstructing with NKSR (detail={args.detail}, mise_iter={args.mise_iter})"
    )
    mesh_out = reconstruct_with_nksr(
        pts,
        nrm,
        detail_level=args.detail,
        mise_iter=args.mise_iter,
        device_str=args.device,
    )

    print(f"[+] Exporting reconstructed mesh to: {args.output_stl}")
    mesh_out.export(args.output_stl)

    print("[✓] Done.")


if __name__ == "__main__":
    main()

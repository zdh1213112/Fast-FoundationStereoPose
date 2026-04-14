from pathlib import Path

import numpy as np
import open3d as o3d


CONFIG = {
    # Input / output
    "ply_path": str(Path(__file__).resolve().parent / "aaaa.ply"),
    "save_denoised": True,
    "out_path": str(Path(__file__).resolve().parent / "aaaa_denoised_largest_cluster.ply"),
    # Denoise
    "voxel_divisor": 80.0,
    "nb_neighbors": 30,
    "std_ratio": 1,
    "radius_multiplier": 1.8,      # radius = voxel_size * radius_multiplier
    "min_radius_neighbors": 8,
    # DBSCAN clustering (after denoise)
    "cluster_eps_multiplier": 2.2,  # eps = voxel_size * cluster_eps_multiplier
    "cluster_min_points": 10,
    # Curvature (on final processed cloud = largest cluster)
    "curvature_knn": 30,
    "curvature_hist_bins": 10,
    "curvature_threshold": 0.02,
    "curvature_select": "low",  # "high" or "low"
    "save_curvature_segment": True,
    "curvature_segment_out_path": str(Path(__file__).resolve().parent / "aaaa_curvature_segment.ply"),
    # Visualization
    "visualize": True,
}


def load_point_cloud(ply_path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"Failed to load point cloud or file is empty: {ply_path}")

    points = np.asarray(pcd.points)
    finite_mask = np.isfinite(points).all(axis=1)
    finite_indices = np.where(finite_mask)[0]
    pcd = pcd.select_by_index(finite_indices)
    if len(pcd.points) < 3:
        raise ValueError("Not enough valid points after finite filtering.")
    return pcd


def denoise_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_divisor: float,
    nb_neighbors: int,
    std_ratio: float,
    radius_multiplier: float,
    min_radius_neighbors: int,
) -> tuple[o3d.geometry.PointCloud, float, dict]:
    points = np.asarray(pcd.points)
    bbox_size = points.max(axis=0) - points.min(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_size))
    voxel_size = max(bbox_diag / voxel_divisor, 1e-5)

    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    stat_filtered, _ = downsampled.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    radius = voxel_size * radius_multiplier
    denoised, _ = stat_filtered.remove_radius_outlier(
        nb_points=min_radius_neighbors,
        radius=radius,
    )

    if len(denoised.points) < 3:
        raise ValueError("Too few points after denoising. Please relax filter parameters.")

    stage_counts = {
        "original": len(pcd.points),
        "downsampled": len(downsampled.points),
        "stat_filtered": len(stat_filtered.points),
        "denoised": len(denoised.points),
    }
    return denoised, voxel_size, stage_counts


def extract_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    eps: float,
    min_points: int,
) -> tuple[o3d.geometry.PointCloud, int, int]:
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False,
        )
    )
    valid = labels >= 0
    if not np.any(valid):
        raise ValueError(
            "DBSCAN found no valid cluster. "
            "Try increasing cluster_eps_multiplier or decreasing cluster_min_points."
        )

    cluster_labels, cluster_counts = np.unique(labels[valid], return_counts=True)
    largest_label = int(cluster_labels[np.argmax(cluster_counts)])
    largest_count = int(np.max(cluster_counts))
    indices = np.where(labels == largest_label)[0]
    largest_cluster = pcd.select_by_index(indices)
    return largest_cluster, largest_label, largest_count


def compute_pca_pose(
    pcd: o3d.geometry.PointCloud,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(pcd.points)
    if points.shape[0] < 3:
        raise ValueError("Not enough points for PCA.")

    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1.0
    return center, eigenvectors, eigenvalues


def compute_obb_from_pca(
    pcd: o3d.geometry.PointCloud,
    pca_center: np.ndarray,
    pca_axes: np.ndarray,
) -> o3d.geometry.OrientedBoundingBox:
    points = np.asarray(pcd.points)
    local_points = (points - pca_center) @ pca_axes

    local_min = local_points.min(axis=0)
    local_max = local_points.max(axis=0)
    extent = np.maximum(local_max - local_min, 1e-6)
    center_local = (local_min + local_max) * 0.5
    center_world = pca_center + pca_axes @ center_local

    obb = o3d.geometry.OrientedBoundingBox(center=center_world, R=pca_axes, extent=extent)
    obb.color = (1.0, 0.2, 0.2)
    return obb


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


def compute_point_curvatures(
    pcd: o3d.geometry.PointCloud,
    knn: int,
) -> np.ndarray:
    points = np.asarray(pcd.points)
    n = points.shape[0]
    if n < 3:
        raise ValueError("Not enough points for curvature computation.")

    k = max(3, min(int(knn), n))
    tree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = np.zeros(n, dtype=np.float64)

    for i in range(n):
        nn_count, idx, _ = tree.search_knn_vector_3d(points[i], k)
        if nn_count < 3:
            curvatures[i] = 0.0
            continue

        neighborhood = points[np.asarray(idx, dtype=np.int64)]
        centered = neighborhood - neighborhood.mean(axis=0)
        cov = (centered.T @ centered) / max(nn_count - 1, 1)
        evals = np.linalg.eigvalsh(cov)
        evals = np.clip(evals, 0.0, None)
        denom = float(evals.sum())
        curvatures[i] = float(evals[0] / denom) if denom > 1e-12 else 0.0

    return curvatures


def print_curvature_distribution(curvatures: np.ndarray, hist_bins: int) -> None:
    pcts = np.percentile(curvatures, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    print("Curvature stats       :")
    print(
        "  min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
            float(curvatures.min()),
            float(curvatures.max()),
            float(curvatures.mean()),
            float(curvatures.std()),
        )
    )
    print(
        "  p01={:.6f}, p05={:.6f}, p10={:.6f}, p25={:.6f}, p50={:.6f}, p75={:.6f}, p90={:.6f}, p95={:.6f}, p99={:.6f}".format(
            *[float(v) for v in pcts]
        )
    )

    bins = max(3, int(hist_bins))
    hist, edges = np.histogram(curvatures, bins=bins)
    print(f"Curvature histogram   : {bins} bins")
    for i in range(bins):
        left = float(edges[i])
        right = float(edges[i + 1])
        count = int(hist[i])
        ratio = count / max(len(curvatures), 1) * 100.0
        print(f"  [{left:.6f}, {right:.6f}) -> {count:5d} ({ratio:6.2f}%)")


def segment_by_curvature(
    pcd: o3d.geometry.PointCloud,
    curvatures: np.ndarray,
    threshold: float,
    select_mode: str,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    mode = str(select_mode).strip().lower()
    if mode not in {"high", "low"}:
        raise ValueError("CONFIG['curvature_select'] must be 'high' or 'low'.")

    if mode == "high":
        mask = curvatures >= threshold
    else:
        mask = curvatures <= threshold

    indices = np.where(mask)[0]
    segmented = pcd.select_by_index(indices)
    return segmented, mask


def build_main_axis_lineset(
    center: np.ndarray,
    main_vector: np.ndarray,
    axis_length: float,
) -> o3d.geometry.LineSet:
    axis_length = max(float(axis_length), 1e-3)

    p0 = center - main_vector * (axis_length * 0.5)
    p1 = center + main_vector * (axis_length * 0.5)

    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.vstack([p0, p1]))
    axis.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    axis.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.1, 0.1]], dtype=np.float64))
    return axis


def visualize(
    pcd: o3d.geometry.PointCloud,
    obb: o3d.geometry.OrientedBoundingBox,
    main_vector: np.ndarray,
    curvature_mask: np.ndarray | None = None,
) -> None:
    pcd_vis = o3d.geometry.PointCloud(pcd)
    if curvature_mask is None:
        pcd_vis.paint_uniform_color([0.75, 0.75, 0.75])
    else:
        n = len(pcd.points)
        if curvature_mask.shape[0] != n:
            raise ValueError("Curvature mask size mismatch.")
        colors = np.tile(np.array([0.70, 0.70, 0.70], dtype=np.float64), (n, 1))
        colors[curvature_mask] = np.array([1.0, 0.2, 0.2], dtype=np.float64)
        pcd_vis.colors = o3d.utility.Vector3dVector(colors)

    axis = build_main_axis_lineset(obb.center, main_vector, axis_length=obb.extent[0])
    obb_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    obb_lines.paint_uniform_color([1.0, 0.2, 0.2])

    diag = max(float(np.linalg.norm(obb.extent)), 1e-3)

    center_marker = o3d.geometry.TriangleMesh.create_sphere(radius=diag * 0.02)
    center_marker.paint_uniform_color([0.1, 0.9, 0.1])
    center_marker.translate(obb.center)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=diag * 0.18, origin=[0.0, 0.0, 0.0])
    coord.rotate(obb.R, center=np.zeros(3))
    coord.translate(obb.center)

    title = "Largest Cluster + 6D OBB + PCA Main Vector"
    if curvature_mask is not None:
        title = "Largest Cluster + Curvature Segmentation + 6D OBB"
    o3d.visualization.draw_geometries([pcd_vis, obb_lines, axis, center_marker, coord], window_name=title)


def main() -> None:
    ply_path = Path(CONFIG["ply_path"]).expanduser().resolve()
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    pcd = load_point_cloud(ply_path)
    denoised, voxel_size, stage_counts = denoise_point_cloud(
        pcd=pcd,
        voxel_divisor=float(CONFIG["voxel_divisor"]),
        nb_neighbors=int(CONFIG["nb_neighbors"]),
        std_ratio=float(CONFIG["std_ratio"]),
        radius_multiplier=float(CONFIG["radius_multiplier"]),
        min_radius_neighbors=int(CONFIG["min_radius_neighbors"]),
    )

    cluster_eps = voxel_size * float(CONFIG["cluster_eps_multiplier"])
    largest_cluster, largest_label, largest_count = extract_largest_cluster(
        pcd=denoised,
        eps=cluster_eps,
        min_points=int(CONFIG["cluster_min_points"]),
    )

    pca_center, pca_axes, eigenvalues = compute_pca_pose(largest_cluster)
    main_vector = pca_axes[:, 0]
    obb = compute_obb_from_pca(
        pcd=largest_cluster,
        pca_center=pca_center,
        pca_axes=pca_axes,
    )
    quat_wxyz = rotation_matrix_to_quaternion(obb.R)

    print(f"Input file            : {ply_path}")
    print(f"Original points       : {stage_counts['original']}")
    print(f"After voxel           : {stage_counts['downsampled']}")
    print(f"After stat filter     : {stage_counts['stat_filtered']}")
    print(f"After radius filter   : {stage_counts['denoised']}")
    print(f"DBSCAN eps            : {cluster_eps:.6f}")
    print(f"Largest cluster label : {largest_label}")
    print(f"Largest cluster points: {largest_count}")
    reduction = (1.0 - largest_count / max(stage_counts["original"], 1)) * 100.0
    print(f"Total reduction ratio : {reduction:.2f}%")
    print("PCA eigenvalues       : " + np.array2string(eigenvalues, precision=6, suppress_small=True))
    print("PCA main vector       : " + np.array2string(main_vector, precision=6, suppress_small=True))
    print("OBB center (xyz)      : " + np.array2string(obb.center, precision=6, suppress_small=True))
    print("OBB extent (xyz)      : " + np.array2string(obb.extent, precision=6, suppress_small=True))
    print("OBB rotation R        : " + np.array2string(obb.R, precision=6, suppress_small=True))
    print("OBB quat (wxyz)       : " + np.array2string(quat_wxyz, precision=6, suppress_small=True))

    curvatures = compute_point_curvatures(
        largest_cluster,
        knn=int(CONFIG["curvature_knn"]),
    )
    print_curvature_distribution(
        curvatures,
        hist_bins=int(CONFIG["curvature_hist_bins"]),
    )

    curvature_threshold = float(CONFIG["curvature_threshold"])
    curvature_select = str(CONFIG["curvature_select"])
    curvature_segment, curvature_mask = segment_by_curvature(
        pcd=largest_cluster,
        curvatures=curvatures,
        threshold=curvature_threshold,
        select_mode=curvature_select,
    )
    seg_count = len(curvature_segment.points)
    seg_ratio = seg_count / max(len(largest_cluster.points), 1) * 100.0
    print(f"Curvature threshold   : {curvature_threshold:.6f}")
    print(f"Curvature select mode : {curvature_select}")
    print(f"Curvature seg points  : {seg_count} ({seg_ratio:.2f}%)")

    if bool(CONFIG["save_denoised"]):
        out_path = Path(CONFIG["out_path"]).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = o3d.io.write_point_cloud(str(out_path), largest_cluster)
        if not ok:
            raise RuntimeError(f"Failed to write output point cloud: {out_path}")
        print(f"Saved largest cluster : {out_path}")

    if bool(CONFIG["save_curvature_segment"]):
        seg_path = Path(CONFIG["curvature_segment_out_path"]).expanduser().resolve()
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        ok = o3d.io.write_point_cloud(str(seg_path), curvature_segment)
        if not ok:
            raise RuntimeError(f"Failed to write curvature segment point cloud: {seg_path}")
        print(f"Saved curvature seg   : {seg_path}")

    if bool(CONFIG["visualize"]):
        visualize(largest_cluster, obb, main_vector, curvature_mask=curvature_mask)


if __name__ == "__main__":
    main()

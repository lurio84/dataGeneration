import open3d as o3d
import numpy as np
import time
import os

# Open3D's viewer uses GLEW/GLX, which breaks on native Wayland; XWayland works.
if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
    os.environ["XDG_SESSION_TYPE"] = "x11"

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])

geom_visualize = []
geom_visualize.append(axis)

STEP = 10000
rng = np.random.default_rng(42)

def make_pointcloud_capture_like(
    pcd: o3d.geometry.PointCloud,
    rng: np.random.Generator,
    noise_std: float = 18.0,
    dropout_ratio: float = 0.24,
    outlier_ratio: float = 0.05,
    voxel_size: float = 35.0,
    local_outlier_std: float = 80.0,
    apply_outliers: bool = True,
) -> o3d.geometry.PointCloud:
    """
    Applies imperfections to simulate a real capture:
    - downsampling (limited resolution)
    - ruido gaussiano
    - random dropout of points
    - local outliers (near the surface)
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd

    # Downsampling
    degraded = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts = np.asarray(degraded.points)
    if pts.size == 0:
        return pcd

    # Noise
    noisy_pts = pts + rng.normal(0.0, noise_std, size=pts.shape)
    degraded.points = o3d.utility.Vector3dVector(noisy_pts)

    # Dropout
    pts = np.asarray(degraded.points)
    keep_mask = rng.random(len(pts)) > dropout_ratio
    kept_pts = pts[keep_mask]
    if len(kept_pts) == 0:
        kept_pts = pts[:1]

    # Outliers
    if apply_outliers:
        n_outliers = max(1, int(len(kept_pts) * outlier_ratio))
        anchor_idx = rng.integers(0, len(kept_pts), size=n_outliers)
        anchors = kept_pts[anchor_idx]
        local_offsets = rng.normal(0.0, local_outlier_std, size=(n_outliers, 3))
        outliers = anchors + local_offsets

        all_pts = np.vstack([kept_pts, outliers])
    else:
        all_pts = kept_pts
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(all_pts)
    return final_pcd

N_SAMPLES = 100

for i in range(N_SAMPLES):
    width, height, depth = rng.uniform(400.0, 1400.0, size=3)
    box = o3d.geometry.TriangleMesh.create_box(
        width=float(width),
        height=float(height),
        depth=float(depth),
    )

    SIDEWAYS_OFFSET = 100.0
    FORKLIFT_OFFSET = 300.0
    box = box.translate((-500 + rng.uniform(0.0, SIDEWAYS_OFFSET), 100, 0 + rng.uniform(0.0, FORKLIFT_OFFSET)))
    # box.compute_vertex_normals()

    forklift = o3d.io.read_triangle_mesh("../data/forklift.stl")
    forklift.compute_vertex_normals()
    forklift = forklift.translate((0, 0, -1280))
    # R = forklift.get_rotation_matrix_from_xyz((0, 0, 0))
    # forklift.rotate(R, center=(0, 0, 0))
    

    pcd_box = box.sample_points_uniformly(number_of_points=10000)
    forklift_box = forklift.sample_points_uniformly(number_of_points=10000)
    pcd_box = make_pointcloud_capture_like(pcd_box, rng)
    forklift_box = make_pointcloud_capture_like(forklift_box, rng, apply_outliers=True)


    forklift.translate((i*STEP,0,0))
    box.translate((i*STEP,0,0))
    pcd_box.translate((3000 + i*STEP,0,0)).paint_uniform_color([1,1,0])
    forklift_box.translate((3000 + i*STEP,0,0)).paint_uniform_color([1,0,1])
    
    geom_visualize.append(box)
    geom_visualize.append(forklift)
    geom_visualize.append(pcd_box)
    geom_visualize.append(forklift_box)



vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Mesh + PointCloud Camera Animation", width=1280, height=720)

for g in geom_visualize:
    vis.add_geometry(g)

ctr = vis.get_view_control()

# Camera: advances in X, elevated in Z and with a downward inclination.
front = np.array([1.0, 0.0, 0], dtype=float)
front /= np.linalg.norm(front)
up = np.array([0.0, 1.0, 0.0], dtype=float)
ctr.set_front(front.tolist())
ctr.set_up(up.tolist())
ctr.set_zoom(0.35)

theta = np.pi/10  # angle in radians
front = np.array([np.cos(theta), np.sin(theta), 0], dtype=float)
front /= np.linalg.norm(front)
ctr.set_front(front.tolist())


x_start = -2000.0
x_end = STEP * N_SAMPLES + 6000.0
frames = 4000

for x_cam in np.linspace(x_start, x_end, frames):
    # Moving the interest point along with the camera in X.
    ctr.set_lookat([float(x_cam + 2000.0), 0.0, 0.0])
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.02)

vis.run()
vis.destroy_window()

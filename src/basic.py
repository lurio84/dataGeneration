import os

# Open3D's viewer uses GLEW/GLX, which breaks on native Wayland; XWayland works.
if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
    os.environ["XDG_SESSION_TYPE"] = "x11"

import open3d as o3d
import numpy as np
import copy

box = o3d.geometry.TriangleMesh.create_box(width=1000.0, height=1000.0, depth=1000.0)
box = box.translate((-500,100,0))
# box.compute_vertex_normals()

forklift = o3d.io.read_triangle_mesh("../data/forklift.stl")
forklift.compute_vertex_normals()
forklift = forklift.translate((0, 0, -1280))
R = forklift.get_rotation_matrix_from_xyz((0, 0, 0))
forklift.rotate(R, center=(0, 0, 0))
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])


pcd_box = box.sample_points_uniformly(number_of_points=10000)
forklift_box = forklift.sample_points_uniformly(number_of_points=10000)

pcd_box.translate((5000,0,0)).paint_uniform_color([1,1,0])
forklift_box.translate((5000,0,0)).paint_uniform_color([1,0,1])


o3d.visualization.draw_geometries([box, forklift, axis, pcd_box, forklift_box],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

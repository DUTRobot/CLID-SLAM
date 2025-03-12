#!/usr/bin/env python3
import os
import sys

import open3d as o3d

# 可视化已经保存的PLY文件

def vis_mesh():
    mesh_file = "./ours_mesh_20cm.ply"  # the path of the mesh which you want to visualize
    if not os.path.exists(mesh_file):
        sys.exit(f"Mesh file {mesh_file} does not exist.")
    print("Loading Mesh file: ", mesh_file)

    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    # Check if the mesh was loaded successfully
    if not mesh.has_vertices():
        sys.exit("Failed to load the mesh. No vertices found.")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Mesh Visualization")
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.light_on = True  # Enable lighting to show the mesh color
    opt.mesh_show_back_face = True
    # Enable shortcuts in the console (e.g., Ctrl+9)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    vis_mesh()

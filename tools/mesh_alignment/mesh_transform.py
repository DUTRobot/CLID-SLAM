#!/usr/bin/env python3

import os
import sys

import numpy as np
import open3d as o3d

# 将Mesh与真值Mesh对齐

def mesh_transform():
    mesh_file = "./math_easy.ply"
    output_file = "./math_easy_transformed.ply"

    if not os.path.exists(mesh_file):
        sys.exit(f"Mesh file {mesh_file} does not exist.")
    print("Loading Mesh file: ", mesh_file)

    # Load Mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    transformation_matrix = np.array([[-4.20791735e-01, -9.07157072e-01,  6.01526210e-04, -2.37152142e+01],
                                      [ 9.07112929e-01, -4.20764518e-01,  1.01663692e-02, -3.12685037e+01],
                                      [-8.96939285e-03,  4.82357635e-03,  9.99948140e-01,  1.02807732e+00],
                                      [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    mesh.transform(transformation_matrix)
    o3d.io.write_triangle_mesh(output_file, mesh)


if __name__ == "__main__":
    mesh_transform()
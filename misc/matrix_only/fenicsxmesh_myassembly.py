

"""
use the mesh fenicsx produces, and then 
use my local and global matrix code on top
 to check if it produces the same thing as fenicsx
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from ufl import dx, grad, inner
import petsc4py.PETSc as PETSc
from dolfinx.mesh import *
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from dolfinx.mesh import (CellType, DiagonalType)

# -----------------------
# 1. Generate mesh in FEniCSx
# -----------------------
Nx = 3
Ny = 3
msh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny,
                              mesh.CellType.triangle,
                              diagonal=mesh.DiagonalType.right)

coords = msh.geometry.x[:, :2]
conn_array = msh.topology.connectivity(msh.topology.dim, 0).array.reshape((-1, 3))
nodes = np.array(coords)
elements = np.array(conn_array, dtype=int)

# -----------------------
# 2. Define local stiffness function
# -----------------------
def reference_gradients():
    return np.array([
        [-1.0, -1.0],
        [ 1.0,  0.0],
        [ 0.0,  1.0],
    ])

def triangle_area_and_transform(tri_nodes):
    v0, v1, v2 = tri_nodes
    J = np.column_stack((v1 - v0, v2 - v0))
    area = 0.5 * np.abs(np.linalg.det(J))
    return area, J

def local_stiffness(tri_nodes):
    grads_ref = reference_gradients()
    area, J = triangle_area_and_transform(tri_nodes)
    JT_inv = np.linalg.inv(J).T
    grads_phys = grads_ref @ JT_inv
    Ke = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            Ke[i,j] = np.dot(grads_phys[i], grads_phys[j]) * area
    return Ke

# -----------------------
# 3. Assemble global matrix manually
# -----------------------
N_nodes = nodes.shape[0]
K_manual = np.zeros((N_nodes, N_nodes))

for elem in elements:
    tri_nodes = nodes[elem]
    Ke = local_stiffness(tri_nodes)
    for i_local, i_global in enumerate(elem):
        for j_local, j_global in enumerate(elem):
            K_manual[i_global, j_global] += Ke[i_local, j_local]


# -----------------------
# 4. Assemble FEniCSx matrix
# -----------------------
V = fem.functionspace(msh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = fem.form(inner(grad(u), grad(v)) * dx)

A = assemble_matrix(a, bcs=[])
A.assemble()
# A_dense = A.mat.getDenseArray()

rows, cols = A.getSize()
dense_array = A.getValues(range(rows), range(cols))
np.set_printoptions(threshold=10000, precision=3, suppress=True, linewidth=1000)


# -----------------------
# 5. Compare
# -----------------------
np.set_printoptions(precision=3, suppress=True)

print("\nManual assembly global matrix:")
print(K_manual)
print("\nFEniCSx global matrix:")
print(dense_array)
print("\nMax absolute difference:", np.max(np.abs(K_manual - dense_array)))

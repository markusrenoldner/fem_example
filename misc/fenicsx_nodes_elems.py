



import numpy as np
from mpi4py import MPI
from dolfinx.mesh import *
from dolfinx import fem, mesh, plot
import ufl
from ufl import dx, grad, inner
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from petsc4py import PETSc
from dolfinx.mesh import (CellType, DiagonalType)

Nx=3
Ny=3

msh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny, 
                                mesh.CellType.triangle,
                                diagonal=DiagonalType.right)





print(msh.geometry.x[:,0:2])
print(msh.topology.connectivity(msh.topology.dim, 0).array.reshape((-1, 3)))
boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.full(x.shape[1], True))
facet_to_nodes = msh.topology.connectivity(msh.topology.dim - 1, 0)
boundary_nodes = np.unique(np.hstack([facet_to_nodes.links(f) for f in boundary_facets]))
print(boundary_nodes)



V = fem.functionspace(msh, ("Lagrange", 1))
import pyvista
import matplotlib as mpl
cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
# grid.point_data["u"] = uh.x.array
# grid.set_active_scalars("u")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.show_axes()
plotter.show()


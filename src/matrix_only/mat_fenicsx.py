



import numpy as np
from mpi4py import MPI
from dolfinx.mesh import *
from dolfinx import fem, mesh, plot
import ufl
from ufl import dx, grad, inner
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from petsc4py import PETSc
from dolfinx.mesh import (CellType, DiagonalType)

def assembly_mat(N,plotting=False):
    Nx=N
    Ny=N

    msh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny, 
                                  mesh.CellType.triangle,
                                  diagonal=DiagonalType.right)
    
    # print(msh.geometry.x[:,0:2])
    # print(msh.topology.connectivity(msh.topology.dim, 0).array.reshape((-1, 3)))
    # boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.full(x.shape[1], True))
    # facet_to_nodes = msh.topology.connectivity(msh.topology.dim - 1, 0)
    # boundary_nodes = np.unique(np.hstack([facet_to_nodes.links(f) for f in boundary_facets]))
    # print(boundary_nodes)

    V = fem.functionspace(msh, ("Lagrange", 1))

    # BC
    facets = mesh.locate_entities_boundary(msh, dim=1,
                                        marker=lambda x: np.logical_or.reduce((
                                            np.isclose(x[0], 0.),
                                            np.isclose(x[0], 1.0),
                                            np.isclose(x[1], 0.0),
                                            np.isclose(x[1], 1.))))
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    
    # no BC
    # bc = fem.dirichletbc(0.0, dofs=dofs, V=V) 

    # Next, the LHS of the variational problem is defined:
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = inner(grad(u), grad(v)) * dx 
    bilinear_form = fem.form(a)
    uh = fem.Function(V)
    uh.name = "uh"

    # A = assemble_matrix(bilinear_form, bcs=[bc])
    A = assemble_matrix(bilinear_form, bcs=[]) #no BC
    A.assemble()

    # print matrix
    rows, cols = A.getSize()
    dense_array = A.getValues(range(rows), range(cols))
    np.set_printoptions(threshold=10000, precision=3, suppress=True, linewidth=1000)
    print(dense_array)
    


if __name__ == "__main__":
    np.set_printoptions(threshold=10000, precision=3, suppress=True, linewidth=1000)
    
    assembly_mat(3,False)

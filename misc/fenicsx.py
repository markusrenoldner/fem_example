



import numpy as np
from mpi4py import MPI
from dolfinx.mesh import *
from dolfinx import fem, mesh, plot
import ufl
from ufl import dx, grad, inner
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from petsc4py import PETSc
from dolfinx.mesh import (CellType, DiagonalType)

def solve_poisson_fenicsx(N,plotting=False):

    # mesh
    Nx=N
    Ny=N
    msh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny, 
                                  mesh.CellType.triangle,
                                  diagonal=DiagonalType.right)
    
    # print nodes and elems
    # print(msh.geometry.x[:,0:2])
    # print(msh.topology.connectivity(msh.topology.dim, 0).array.reshape((-1, 3)))
    # boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.full(x.shape[1], True))
    # facet_to_nodes = msh.topology.connectivity(msh.topology.dim - 1, 0)
    # boundary_nodes = np.unique(np.hstack([facet_to_nodes.links(f) for f in boundary_facets]))
    # print(boundary_nodes)

    # space
    V = fem.functionspace(msh, ("Lagrange", 1))

    # BC
    facets = mesh.locate_entities_boundary(msh, dim=1,
                                        marker=lambda x: np.logical_or.reduce((
                                            np.isclose(x[0], 0.),
                                            np.isclose(x[0], 1.0),
                                            np.isclose(x[1], 0.0),
                                            np.isclose(x[1], 1.))))
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc = fem.dirichletbc(0.0, dofs=dofs, V=V)

    # Next, the variational problem is defined:
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 1
    a = inner(grad(u), grad(v)) * dx 
    L = inner(f, v) * dx
    bilinear_form = fem.form(a)
    linear_form = fem.form(L)
    uh = fem.Function(V)
    uh.name = "uh"

    # assembly
    # A = assemble_matrix(bilinear_form, bcs=[bc])
    A = assemble_matrix(bilinear_form, bcs=[])
    A.assemble()
    b = create_vector(linear_form)
    assemble_vector(b, linear_form)
    apply_lifting(b, [bilinear_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # print matrix
    rows, cols = A.getSize()
    dense_array = A.getValues(range(rows), range(cols))
    np.set_printoptions(threshold=10000, precision=3, suppress=True, linewidth=1000)
    print(dense_array)
    
    # solver
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    print("max" , uh.x.array.max())

    # plots
    if plotting:
        import pyvista
        import matplotlib as mpl
        cells, types, x = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = uh.x.array
        grid.set_active_scalars("u")
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=False)
        plotter.show_axes()
        plotter.show()

        # import pyvista
        # import dolfinx
        # # PLOT MESH:
        # plotter = pyvista.Plotter()
        # ugrid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(msh))
        # # if msh.geometry.cmaps[0].degree > 1:
        # #     plotter.add_mesh(ugrid, style="points", color="b", point_size=10)
        # #     ugrid = ugrid.tessellate()
        # #     show_edges = False
        # # else:
        # show_edges = True
        # plotter.add_mesh(ugrid, show_edges=show_edges)

        # plotter.show_axes()
        # plotter.view_xy()
        # # plotter.save_graphic("test.pdf")
        # plotter.show()

def conv_test(n_max):
    raise NotImplementedError("error not implemented")

    for n in range(2, n_max):
        uh, L2_error = solve_poisson_fenicsx(n)
        print(f"n={n}, L2 error={L2_error:.6f}")

    return

if __name__ == "__main__":
    np.set_printoptions(threshold=10000, precision=3, suppress=True, linewidth=1000)
    
    solve_poisson_fenicsx(3,False)

    # conv_test(7)
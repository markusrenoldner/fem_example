

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.tri as mtri


def generate_structured_triangular_mesh(nx, ny):

    # Generate nodes
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    nodes = np.column_stack([xv.ravel(), yv.ravel()])  

    # Map 2D grid index to 1D node index
    def node_id(i, j):
        return i * (ny + 1) + j

    # Generate elements
    elements = []
    for i in range(nx):
        for j in range(ny):
            n0 = node_id(i, j)
            n1 = node_id(i + 1, j)
            n2 = node_id(i, j + 1)
            n3 = node_id(i + 1, j + 1)
            # Two triangles per square
            elements.append([n0, n1, n3])  # lower right triangle
            elements.append([n0, n3, n2])  # upper left triangle
            # elements.append([n0, n2, n3])
    elements = np.array(elements, dtype=int)

    # Identify boundary nodes
    boundary_nodes = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            if i == 0 or i == nx or j == 0 or j == ny:
                boundary_nodes.append(node_id(i, j))
    boundary_nodes = np.array(sorted(set(boundary_nodes)), dtype=int)

    return nodes, elements, boundary_nodes

def reference_gradients():
    return np.array([
        [-1.0, -1.0],  # grad phi_1
        [ 1.0,  0.0],  # grad phi_2
        [ 0.0,  1.0],  # grad phi_3
    ])

def triangle_area_and_transform(tri_nodes):
    v0, v1, v2 = tri_nodes
    J = np.column_stack((v1 - v0, v2 - v0))  # shape (2,2)
    detJ = np.linalg.det(J)
    area = 0.5 * abs(detJ)
    return area, J, detJ

# def local_stiffness(tri_nodes):
#     grads_ref = reference_gradients()  # shape (3,2)
#     area, J, detJ = triangle_area_and_transform(tri_nodes)
#     JT_inv = np.linalg.inv(J).T  # J^{-T}
    
#     grads_phys = grads_ref @ JT_inv  # shape (3,2)

#     Ke = np.zeros((3, 3))
#     for i in range(3):
#         for j in range(3):
#             Ke[i, j] = np.dot(grads_phys[i], grads_phys[j]) * area
#     return Ke
# def local_stiffness(tri_nodes):
#     # Reorder to ensure (v0, v1, v2) forms a positively oriented triangle
#     v0, v1, v2 = tri_nodes
#     J = np.column_stack((v1 - v0, v2 - v0))
#     if np.linalg.det(J) < 0:
#         tri_nodes[[1, 2]] = tri_nodes[[2, 1]]  # consistent with ref triangle
#     grads_ref = reference_gradients()
#     area, J, _ = triangle_area_and_transform(tri_nodes)
#     JT_inv = np.linalg.inv(J).T
#     grads_phys = grads_ref @ JT_inv
#     return grads_phys @ grads_phys.T * area
def local_stiffness(tri_nodes):
    v0, v1, v2 = tri_nodes
    J = np.column_stack((v1 - v0, v2 - v0))
    if np.linalg.det(J) < 0:
        tri_nodes = tri_nodes[[0, 2, 1]]  # reassign reordered copy

    grads_ref = reference_gradients()
    area, J, _ = triangle_area_and_transform(tri_nodes)
    JT_inv = np.linalg.inv(J).T
    grads_phys = grads_ref @ JT_inv
    return grads_phys @ grads_phys.T * area

def assemble_global_matrix(nodes, elements):
    N_nodes = nodes.shape[0]
    K = np.zeros((N_nodes, N_nodes))
    print(elements)
    for elem,i in zip(elements,range(len(elements))):
        elem = elements[i]
        if i>20: continue
        tri_nodes = nodes[elem]  # (3,2)
        Ke = local_stiffness(tri_nodes)
        for i_local, i_global in enumerate(elem):
            for j_local, j_global in enumerate(elem):
                K[i_global, j_global] += Ke[i_local, j_local]

    return K




if __name__ == "__main__":

    np.set_printoptions(threshold=10000, precision=3, suppress=True, linewidth=1000)


    nx, ny = 3,3
    # nodes, elements, boundary_nodes = generate_structured_triangular_mesh(nx, ny)
    # print(nodes)
    # print(elements)
    # print(boundary_nodes)

    from mpi4py import MPI
    from dolfinx.mesh import *
    from dolfinx import mesh
    from dolfinx.mesh import DiagonalType
    
    nx, ny = 3,3
    msh = mesh.create_unit_square(MPI.COMM_WORLD, nx,ny, 
                                  mesh.CellType.triangle,
                                  diagonal=DiagonalType.right)
    
    nodes=msh.geometry.x[:,0:2]
    elements=msh.topology.connectivity(msh.topology.dim, 0).array.reshape((-1, 3))
    # boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.full(x.shape[1], True))
    # facet_to_nodes = msh.topology.connectivity(msh.topology.dim - 1, 0)
    # boundary_nodes = np.unique(np.hstack([facet_to_nodes.links(f) for f in boundary_facets]))
    print(nodes)
    print(elements)




    K = assemble_global_matrix(nodes, elements)
    print(K)
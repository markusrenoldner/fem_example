

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.tri as mtri


def generate_structured_triangular_mesh(nx, ny):
    # Generate nodes
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    
    xv, yv = np.meshgrid(x, y, indexing='ij')
        # xv, yv = np.meshgrid(x, y, indexing='xy')

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

def plot_mesh(nodes, elements):
    """
    Plot a 2D triangular mesh.

    Parameters:
        nodes (ndarray): shape (num_nodes, 2), coordinates of nodes
        elements (ndarray): shape (num_elements, 3), indices of triangle vertices
    """
    fig, ax = plt.subplots()
    
    # Draw triangle edges
    edges = []
    for tri in elements:
        for i in range(3):
            a = nodes[tri[i]]
            b = nodes[tri[(i + 1) % 3]]
            edges.append([a, b])
    edge_collection = LineCollection(edges, colors='k', linewidths=0.5)
    ax.add_collection(edge_collection)

    # Draw nodes
    ax.plot(nodes[:, 0], nodes[:, 1], 'o', markersize=5, color='red')

    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("2D Triangular Mesh")
    plt.tight_layout()
    plt.show()


def reference_gradients():
    return np.array([
        [-1.0, -1.0],  # grad phi_1
        [ 1.0,  0.0],  # grad phi_2
        [ 0.0,  1.0],  # grad phi_3
    ])

def triangle_area_and_transform(tri_nodes):
    v0, v1, v2 = tri_nodes
    J = np.column_stack((v1 - v0, v2 - v0))  # shape (2,2)
    area = 0.5 * np.abs(np.linalg.det(J))
    return area, J

def local_stiffness(tri_nodes):
    grads_ref = reference_gradients()  # shape (3,2)
    area, J = triangle_area_and_transform(tri_nodes)
    JT_inv = np.linalg.inv(J).T  # J^{-T}
    
    grads_phys = grads_ref @ JT_inv  # shape (3,2)

    Ke = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Ke[i, j] = np.dot(grads_phys[i], grads_phys[j]) * area
    return Ke


def assemble_global_matrix(nodes, elements):
    N_nodes = nodes.shape[0]
    K = np.zeros((N_nodes, N_nodes))

    for elem in elements:
        tri_nodes = nodes[elem]  # (3,2)
        Ke = local_stiffness(tri_nodes)
        for i_local, i_global in enumerate(elem):
            for j_local, j_global in enumerate(elem):
                K[i_global, j_global] += Ke[i_local, j_local]

    return K


def apply_dirichlet_bc(K, boundary_nodes, g, nodes):
    K_mod = K.copy()

    for node in boundary_nodes:
        K_mod[node, :] = 0
        K_mod[:, node] = 0
        K_mod[node, node] = 1
    return K_mod




if __name__ == "__main__":
    np.set_printoptions(threshold=10000, precision=3, suppress=True, linewidth=1000)


    nx = 3
    ny = 3
    nodes, elements, boundary_nodes = generate_structured_triangular_mesh(nx, ny)

    # plot_mesh(nodes, elements)
    # print(nodes)
    # print(elements)
    # print(boundary_nodes)


    K = assemble_global_matrix(nodes, elements)
    # print(K)

    # K_mod = apply_dirichlet_bc(K, boundary_nodes, lambda x, y: 0, nodes)
    # print(K_mod)

    ##########################################################
    # print local matrix

    # bottom-left square
    i = 0
    j = 0
    ny = 3

    def node_id(i, j):
        return i * (ny + 1) + j

    n0 = node_id(0, 0)
    n1 = node_id(1, 0)
    n2 = node_id(0, 1)
    n3 = node_id(1, 1)

    tris = [
        [n0, n1, n3],
        [n0, n3, n2],
    ]

    for k, tri in enumerate(tris):
        tri_nodes = nodes[tri]
        print(f"\nYour triangle {k}")
        print(tri_nodes)
        print(local_stiffness(tri_nodes))

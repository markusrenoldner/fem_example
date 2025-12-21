

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.tri as mtri


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
    
    # grads_phys = grads_ref @ JT_inv  # shape (3,2) # doesnt work
    grads_phys = np.linalg.solve(J.T, grads_ref.T).T # works

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


def local_load_vector(tri_nodes, f):
    area, _ = triangle_area_and_transform(tri_nodes)
    midpoint = tri_nodes.mean(axis=0)
    f_val = f(midpoint[0], midpoint[1])
    # P1 basis functions at midpoint are all 1/3
    fe = np.full(3, f_val * area / 3)
    return fe

def local_load_vector_higher(tri_nodes, f):
    area, _ = triangle_area_and_transform(tri_nodes)
    # 3-point quadrature for triangles (degree 2 exact)
    qp_bary = np.array([
        [1/6, 1/6, 2/3],
        [1/6, 2/3, 1/6],
        [2/3, 1/6, 1/6]
    ])
    weights = np.array([1/3, 1/3, 1/3])
    fe = np.zeros(3)
    for k in range(3):
        l1, l2, l3 = qp_bary[k]
        x_qp = l1*tri_nodes[0,0] + l2*tri_nodes[1,0] + l3*tri_nodes[2,0]
        y_qp = l1*tri_nodes[0,1] + l2*tri_nodes[1,1] + l3*tri_nodes[2,1]
        f_val = f(x_qp, y_qp)
        # P1 basis functions evaluated at quadrature point
        phi = np.array([l1, l2, l3])
        fe += weights[k] * f_val * phi
    fe *= area
    return fe



def assemble_load_vector(nodes, elements, f):
    N_nodes = nodes.shape[0]
    b = np.zeros(N_nodes)

    for elem in elements:
        tri_nodes = nodes[elem]
        fe = local_load_vector_higher(tri_nodes, f)
        for i_local, i_global in enumerate(elem):
            b[i_global] += fe[i_local]

    return b

def apply_dirichlet_bc(K, b, boundary_nodes, g, nodes):
    K_mod = K.copy()
    b_mod = b.copy()


    for node in boundary_nodes:
        value = g(*nodes[node])

        # RHS correction using ORIGINAL column
        b_mod -= K_mod[:, node] * value

        # Enforce Dirichlet
        K_mod[node, :] = 0.0
        K_mod[:, node] = 0.0
        K_mod[node, node] = 1.0
        b_mod[node] = value

    # for node in boundary_nodes:
    #     K_mod[node, :] = 0
    #     K_mod[:, node] = 0
    #     K_mod[node, node] = 1

    #     # Set prescribed value
    #     b_mod[node] = g(*nodes[node])

    return K_mod, b_mod

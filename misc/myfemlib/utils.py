

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.tri as mtri
# from mesh import triangle_area_and_transform
from assembly import reference_gradients, triangle_area_and_transform


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


def solve_system(K, b):
    u = np.linalg.solve(K, b)
    return u


def fem_p1_function(u, nodes, elements):
    """
    Return a function representing the P1 FEM solution.

    Inputs:
        u: (N_nodes,) array of nodal values
        nodes: (N_nodes, 2) node coordinates
        elements: (N_elements, 3) node indices per triangle

    Output:
        A function sol(x,y) that returns the FEM solution at (x,y)
    """
    
    raise NotImplementedError("this function may be wrong")


    from scipy.spatial import Delaunay
    tri = Delaunay(nodes)

    def sol(x, y):
        p = np.array([x, y])
        simplex = tri.find_simplex(p)
        if simplex == -1:
            raise ValueError("Point outside mesh")

        vert_indices = elements[simplex]
        verts = nodes[vert_indices]

        # Compute barycentric coordinates
        T = np.vstack((verts.T, np.ones(3)))
        v = np.array([x, y, 1])
        bary = np.linalg.solve(T, v)

        return np.dot(bary, u[vert_indices])

    return sol




def plot_fem_solution(nodes, elements, u):
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    plt.figure()
    plt.tripcolor(triang, u, shading='gouraud')
    plt.triplot(triang, color='k', linewidth=0.6, alpha=0.8)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('FEM P1 solution')
    plt.gca().set_aspect('equal')
    plt.show()


def compute_L2_error(nodes, elements, u_h, u_exact):
    error_sq = 0.0
    for el in elements:
        tri_nodes = nodes[el]
        area, J = (tri_nodes)
        # Midpoint of triangle for quadrature (P1 exact)
        midpoint = np.mean(tri_nodes, axis=0)
        
        # FEM solution at midpoint (P1 interpolation)
        bary_coords = np.array([1/3, 1/3, 1/3])
        u_h_mid = np.dot(u_h[el], bary_coords)
        
        u_ex_mid = u_exact(midpoint[0], midpoint[1])
        error_sq += area * (u_h_mid - u_ex_mid)**2
    return np.sqrt(error_sq)

def compute_L2_error_higher(nodes, elements, u_h, u_exact):
    # 3-point quadrature for triangles (degree 2 exact)
    # Use the same rule as local_load_vector_higher for consistency
    qp_bary = np.array([
        [1/6, 1/6, 2/3],
        [1/6, 2/3, 1/6],
        [2/3, 1/6, 1/6]
    ])
    weights = np.array([1/3, 1/3, 1/3])

    error_sq = 0.0
    for el in elements:
        tri_nodes = nodes[el]  # (3,2)
        area, _ = triangle_area_and_transform(tri_nodes)

        u_el = u_h[el]  # FEM nodal values

        for k in range(3):
            l1, l2, l3 = qp_bary[k]
            # Evaluate FEM solution at quadrature point
            u_h_qp = l1*u_el[0] + l2*u_el[1] + l3*u_el[2]
            # Map barycentric to physical coordinates
            x_qp = l1*tri_nodes[0,0] + l2*tri_nodes[1,0] + l3*tri_nodes[2,0]
            y_qp = l1*tri_nodes[0,1] + l2*tri_nodes[1,1] + l3*tri_nodes[2,1]
            u_ex_qp = u_exact(x_qp, y_qp)
            error_sq += weights[k] * area * (u_h_qp - u_ex_qp)**2

    return np.sqrt(error_sq)



def compute_H1_error(nodes, elements, u_h, grad_u_exact):
    error_sq = 0.0
    for el in elements:
        tri_nodes = nodes[el]
        area, J = triangle_area_and_transform(tri_nodes)
        JT_inv = np.linalg.inv(J).T
        
        grads_ref = reference_gradients()  # shape (3,2)
        # grads_phys = grads_ref @ JT_inv  # shape (3,2)
        grads_phys = np.linalg.solve(J.T, grads_ref.T).T
        
        # FEM solution gradient on element (constant)
        grad_u_h = np.zeros(2)
        for i in range(3):
            grad_u_h += u_h[el[i]] * grads_phys[i]
        
        # Midpoint for exact gradient evaluation
        midpoint = np.mean(tri_nodes, axis=0)
        grad_u_ex = np.array(grad_u_exact(midpoint[0], midpoint[1]))
        
        error_sq += area * np.sum((grad_u_h - grad_u_ex)**2)
    return np.sqrt(error_sq)


def conv_plot(errors_L2, errors_H1, ns=None, title="Convergence (log-log)"):
    """
    L2 and H1 error on log-log scale, with reference
    trendlines for O(h^2) and O(h) convergence.

    ns: mesh subdivisions per side (e.g., n = 2**k).
        If None, assumes doubling per step and sets h = 2^{-k}.

    """

    k = np.arange(len(errors_L2))
    if ns is None:
        # Assume n doubles: h_k = 2^{-k}
        h = 2.0 ** (-k)
    else:
        ns = np.asarray(ns, dtype=float)
        h = 1.0 / ns

    errors_L2 = np.asarray(errors_L2, dtype=float)
    errors_H1 = np.asarray(errors_H1, dtype=float)

    # Build reference lines anchored at the last error
    C2 = errors_L2[-1] / (h[-1] ** 2)
    ref_L2 = C2 * (h ** 2)
    C1 = errors_H1[-1] / (h[-1] ** 1)
    ref_H1 = C1 * (h ** 1)

    plt.figure()
    plt.loglog(h, errors_L2, 'o-', label='L2 error')
    plt.loglog(h, errors_H1, 'o-', label='H1 error')
    plt.loglog(h, ref_L2, '--', color='black', label='O(h^2)')
    plt.loglog(h, ref_H1, '--', color='black', alpha=0.6, label='O(h)')
    # plt.gca().invert_xaxis()  # optional: show refinement to the right
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.xlabel('h')
    plt.ylabel('error')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


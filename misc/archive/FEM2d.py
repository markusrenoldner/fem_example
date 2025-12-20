import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def generate_structured_triangular_mesh(nx, ny):
    """
    Generate a structured triangular mesh on the unit square [0,1]x[0,1].

    Returns:
        nodes (ndarray):          shape (num_nodes, 2), coordinates of nodes
        elements (ndarray):       shape (num_elements, 3), indices of triangle vertices
        boundary_nodes (ndarray): sorted array of boundary node indices
    """
    # Grid spacing
    hx = 1.0 / nx
    hy = 1.0 / ny

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
    """
    Returns gradients of P1 basis functions on the reference triangle
    with vertices (0,0), (1,0), (0,1).
    
    Output:
        grads: (3, 2) array with gradients of phi_1, phi_2, phi_3
    """
    return np.array([
        [-1.0, -1.0],  # grad phi_1
        [ 1.0,  0.0],  # grad phi_2
        [ 0.0,  1.0],  # grad phi_3
    ])

def triangle_area_and_transform(tri_nodes):
    """
    Given 3 physical triangle vertices (shape (3,2)),
    compute area and Jacobian of the affine transformation
    from reference triangle.

    Input:
        tri_nodes: (3,2) array with coordinates of triangle vertices
    
    Returns:
        area: scalar
        J: (2,2) Jacobian matrix
    """
    v0, v1, v2 = tri_nodes
    J = np.column_stack((v1 - v0, v2 - v0))  # shape (2,2)
    area = 0.5 * np.abs(np.linalg.det(J))
    return area, J

def local_stiffness(tri_nodes):
    """
    Compute local stiffness matrix for a triangle.

    Input:
        tri_nodes: (3,2) array with coordinates of triangle vertices
    
    Output:
        Ke: (3,3) local stiffness matrix

    The P1 gradients are constant so:
    K_ij = ∫_T grad(phi_i) * grad(phi_j) dA
         = A_T * grad(phi_i) * grad(phi_j)

    """
    # Ensure a consistent (positive) orientation for the affine map
    v0, v1, v2 = tri_nodes
    J0 = np.column_stack((v1 - v0, v2 - v0))
    if np.linalg.det(J0) < 0:
        tri_nodes = tri_nodes[[0, 2, 1]]

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
    """
    Assemble global stiffness matrix for the mesh.

    Inputs:
        nodes: (N_nodes, 2) array of node coordinates
        elements: (N_elements, 3) array of node indices per triangle

    Output:
        K: (N_nodes, N_nodes) global stiffness matrix (dense)
    """
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
    """
    Compute local load vector for a triangle using midpoint quadrature.

    Inputs:
        tri_nodes: (3,2) array of triangle vertices
        f: function f(x,y) defining RHS
    
    Output:
        fe: (3,) local load vector
    """
    area, _ = triangle_area_and_transform(tri_nodes)
    midpoint = tri_nodes.mean(axis=0)
    f_val = f(midpoint[0], midpoint[1])
    # P1 basis functions at midpoint are all 1/3
    fe = np.full(3, f_val * area / 3)
    return fe

def assemble_load_vector(nodes, elements, f):
    """
    Assemble global load vector.

    Inputs:
        nodes: (N_nodes, 2) array of node coordinates
        elements: (N_elements, 3) array of node indices per triangle
        f: function f(x,y) RHS
    
    Output:
        b: (N_nodes,) load vector
    """
    N_nodes = nodes.shape[0]
    b = np.zeros(N_nodes)

    for elem in elements:
        tri_nodes = nodes[elem]
        fe = local_load_vector(tri_nodes, f)
        for i_local, i_global in enumerate(elem):
            b[i_global] += fe[i_local]

    return b

def apply_dirichlet_bc(K, b, boundary_nodes, g, nodes):
    """
    Apply Dirichlet BC to the system.

    Inputs:
        K: (N,N) stiffness matrix
        b: (N,) load vector
        boundary_nodes: indices of Dirichlet nodes
        g: function g(x,y) prescribing Dirichlet values

    Outputs:
        K_mod: modified stiffness matrix
        b_mod: modified load vector
    """
    K_mod = K.copy()
    b_mod = b.copy()

    for node in boundary_nodes:
        K_mod[node, :] = 0
        K_mod[:, node] = 0
        K_mod[node, node] = 1

        # Set prescribed value
        b_mod[node] = g(*nodes[node])
    
        # print(f"Applying Dirichlet BC at node {node}, position {nodes[node]}, value {g(*nodes[node])}")


    return K_mod, b_mod

def solve_system(K, b):
    """
    Solve the linear system Ku = b.

    Inputs:
        K: (N,N) stiffness matrix (after BC applied)
        b: (N,) load vector (after BC applied)

    Output:
        u: (N,) solution vector
    """
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

import matplotlib.tri as mtri

def plot_fem_solution(nodes, elements, u):
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    plt.figure()
    plt.tripcolor(triang, u, shading='gouraud', cmap='viridis')
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
        area, J = triangle_area_and_transform(tri_nodes)
        # Midpoint of triangle for quadrature (P1 exact)
        midpoint = np.mean(tri_nodes, axis=0)
        
        # FEM solution at midpoint (P1 interpolation)
        bary_coords = np.array([1/3, 1/3, 1/3])
        u_h_mid = np.dot(u_h[el], bary_coords)
        
        u_ex_mid = u_exact(midpoint[0], midpoint[1])
        error_sq += area * (u_h_mid - u_ex_mid)**2
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


def u_exact(x, y):
    # return np.sin(np.pi * x) * np.sin(np.pi * y)
    return np.sin(np.pi*x) * np.sinh(np.pi*y)

def grad_u_exact(x, y):
    # return np.pi*np.array([np.cos(np.pi * x) * np.sin(np.pi * y),
    #                        np.sin(np.pi * x) * np.cos(np.pi * y)])
    ux = np.pi * np.cos(np.pi*x) * np.sinh(np.pi*y)
    uy = np.pi * np.sin(np.pi*x) * np.cosh(np.pi*y)
    return np.array([ux, uy])


from mpi4py import MPI
from dolfinx.mesh import *
from dolfinx import mesh
from dolfinx.mesh import DiagonalType

# nx, ny = 3,3
def solve_poission(n=2,plotting=False):

    nx, ny = n,n
    nodes, elements, boundary_nodes = generate_structured_triangular_mesh(nx, ny)
    # print("Nodes:\n", nodes)
    # print("Elements:\n", elements)
    # print("Boundary Nodes:\n", boundary_nodes)
    # plot_mesh(nodes, elements)

    # nx, ny = n,n
    # msh = mesh.create_unit_square(MPI.COMM_WORLD, nx,ny, 
    #                             mesh.CellType.triangle,
    #                             diagonal=DiagonalType.right)
    
    # nodes=msh.geometry.x[:,0:2]
    # elements=msh.topology.connectivity(msh.topology.dim, 0).array.reshape((-1, 3))
    # boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, lambda x: np.full(x.shape[1], True))
    # facet_to_nodes = msh.topology.connectivity(msh.topology.dim - 1, 0)
    # boundary_nodes = np.unique(np.hstack([facet_to_nodes.links(f) for f in boundary_facets]))

    K = assemble_global_matrix(nodes, elements)
    # print(K)

    # f = lambda x, y: -2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
    f = lambda x, y: 1.0
    f = lambda x, y: 0.0
    # f = lambda x, y: 2*(np.pi**2)* np.sin(np.pi*x)*np.sin(np.pi*y)

    b = assemble_load_vector(nodes, elements, f)
    # print(b)

    # dirichlet
    g = lambda x, y: 0.0 
    g = lambda x, y: u_exact(x,y)
    # print(u_exact(0.234,0))

    K_mod, b_mod = apply_dirichlet_bc(K, b, boundary_nodes, g, nodes)
    # print(K_mod)
    u = solve_system(K_mod, b_mod)
    # print("max", u.max())
    if plotting: plot_fem_solution(nodes, elements, u)

    L2_error = compute_L2_error(nodes, elements, u, u_exact)
    H1_error = compute_H1_error(nodes, elements, u, grad_u_exact)


    return u, L2_error, H1_error

def conv_test(k_max):

    for k in range(0,k_max):
        n=2**k
        u, L2_error, H1_error = solve_poission(n)
        # print(f"n={n}, L2 error={L2_error:.6f}")
        print(L2_error)#,"\t", H1_error)

    return


if __name__ == "__main__":

    np.set_printoptions(threshold=10000, precision=3, suppress=True, linewidth=1000)

    # solve_poission(5, plotting=False)

    conv_test(6)

    from fenicsx import solve_poisson_fenicsx

    # solve_poisson_fenicsx(5,plotting=False)
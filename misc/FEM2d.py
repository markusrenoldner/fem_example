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


if __name__ == "__main__":

    np.set_printoptions(threshold=10000, precision=3, suppress=True, linewidth=1000)

    # Example usage
    nx, ny = 10,10
    nodes, elements, boundary_nodes = generate_structured_triangular_mesh(nx, ny)
    
    print("Nodes:\n", nodes)
    print("Elements:\n", elements)
    print("Boundary Nodes:\n", boundary_nodes)
    
    # Check the shape of the outputs
    print("Number of nodes:", nodes.shape[0])
    print("Number of elements:", elements.shape[0])
    print("Number of boundary nodes:", boundary_nodes.shape[0])


    # Plot the mesh
    # plot_mesh(nodes, elements)

    # Test reference gradients
    grads = reference_gradients()
    print("Reference gradients:\n", grads)
    # Test triangle area and transformation
    tri_nodes = np.array([[0, 0], [1, 0], [0, 1]])
    area, J = triangle_area_and_transform(tri_nodes)
    print("Area of reference triangle:", area)
    print("Jacobian of reference triangle:\n", J)

    # Test local stiffness matrix
    Ke = local_stiffness(tri_nodes)
    print("Local stiffness matrix:\n", Ke)
    # Example of local stiffness matrix for a triangle with specific vertices
    tri_nodes = np.array([[0, 0], [1, 0], [0, 1]])
    Ke = local_stiffness(tri_nodes)

    # test global stiffness matrix assembly
    K = assemble_global_matrix(nodes, elements)
    print("Global stiffness matrix shape:", K.shape)
    print("Global stiffness matrix:\n", K)
    # Check symmetry of the global stiffness matrix
    print("Is global stiffness matrix symmetric?", np.allclose(K, K.T))
    # Check if the global stiffness matrix is positive definite
    eigenvalues = np.linalg.eigvalsh(K)
    print("Eigenvalues of global stiffness matrix:", eigenvalues)
    print("Are all eigenvalues positive?", np.all(eigenvalues > 0))
    # Check if the global stiffness matrix is sparse
    print("Is global stiffness matrix sparse?", np.count_nonzero(K) < 0.1 * K.size)

    # test load vector
    def f(x, y):
        return 100*y**5 # + y  # Example function
    b = assemble_load_vector(nodes, elements, f)
    print("Global load vector shape:", b.shape)
    print("Global load vector:\n", b)
    
    # apply Dirichlet boundary conditions
    def g(x, y):
        return 0.
    
    K_mod, b_mod = apply_dirichlet_bc(K, b, boundary_nodes, g,nodes)
    print("Modified stiffness matrix shape:", K_mod.shape)
    print("Modified stiffness matrix:\n", K_mod)
    print("Modified load vector shape:", b_mod.shape)
    print("Modified load vector:\n", b_mod)
    # check if load vec is zero at Dirichlet nodes
    print("Load vector at Dirichlet nodes:", b_mod[boundary_nodes])


    # solve the system
    u = solve_system(K_mod, b_mod)
    
    print("Solution vector:\n", u)

    # plot
    plot_fem_solution(nodes, elements, u)
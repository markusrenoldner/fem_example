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
            # elements.append([n0, n1, n3])  # lower right triangle
            # elements.append([n0, n3, n2])  # upper left triangle
            elements.append([n0, n1, n3])  # lower right triangle
            elements.append([n0, n2, n3])  # upper left triangle (flipped)

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
    
    # grads_phys = grads_ref @ JT_inv  # shape (3,2)
    grads_phys = np.linalg.solve(J.T, grads_ref.T).T

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
        K_mod[node, :] = 0
        K_mod[:, node] = 0
        K_mod[node, node] = 1

        # Set prescribed value
        b_mod[node] = g(*nodes[node])

    return K_mod, b_mod

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

def compute_L2_error_higher(nodes, elements, u_h, u_exact):
    # 3-point quadrature for triangles (degree 2 exact)
    # Points in barycentric coordinates and weights
    qp_bary = np.array([
        [1/2, 1/2, 0],
        [0, 1/2, 1/2],
        [1/2, 0, 1/2]
    ])
    weights = np.array([1/3, 1/3, 1/3])  # all weights sum to 1

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


def u_exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def solve_poission(n=2,plotting=False):

    nx, ny = n,n
    nodes, elements, boundary_nodes = generate_structured_triangular_mesh(nx, ny)

    K = assemble_global_matrix(nodes, elements)
    # print(K)

    f = lambda x, y: +2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)

    b = assemble_load_vector(nodes, elements, f)
    # print(b)

    # dirichlet
    g = lambda x, y: 0.0 

    K_mod, b_mod = apply_dirichlet_bc(K, b, boundary_nodes, g, nodes)

    u = solve_system(K_mod, b_mod)

    if plotting: plot_fem_solution(nodes, elements, u)

    L2_error = compute_L2_error_higher(nodes, elements, u, u_exact)


    return u, L2_error

def conv_test(k_max):
    errors = []
    for k in range(0,k_max):
        n=2**k
# def conv_test(n_max):

#     for n in range(2,n_max):
        u, L2_error = solve_poission(n)
        # print(f"n={n}, L2 error={L2_error:.6f}")
        errors.append(L2_error)

    for i in range(0, len(errors)):
        rate = np.log(errors[i-1]/errors[i]) / np.log(2)
        print(errors[i], "\t", rate)

    return


if __name__ == "__main__":

    np.set_printoptions(threshold=10000, precision=3, suppress=True, linewidth=1000)

    # solve_poission(4, plotting=True)
    conv_test(5)

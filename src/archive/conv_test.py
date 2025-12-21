from FEM2d import *

"""
This script tests the convergence of a finite element method (FEM) solution
to the exact solution of a 2D Poisson equation with Dirichlet boundary conditions.
It uses linear Lagrange elements (P1) on triangular meshes.

exact solution: u(x,y) = sin(pi*x) * sin(pi*y)
right hand side: f(x,y) = 2 * pi^2 * sin(pi*x) * sin(pi*y)
"""


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

def compute_H1_error(nodes, elements, u_h, u_exact, grad_u_exact):
    error_sq = 0.0
    for el in elements:
        tri_nodes = nodes[el]
        area, J = triangle_area_and_transform(tri_nodes)
        JT_inv = np.linalg.inv(J).T
        
        grads_ref = reference_gradients()  # shape (3,2)
        grads_phys = grads_ref @ JT_inv  # shape (3,2)
        
        # FEM solution gradient on element (constant)
        grad_u_h = np.zeros(2)
        for i in range(3):
            grad_u_h += u_h[el[i]] * grads_phys[i]
        
        # Midpoint for exact gradient evaluation
        midpoint = np.mean(tri_nodes, axis=0)
        grad_u_ex = np.array(grad_u_exact(midpoint[0], midpoint[1]))
        
        error_sq += area * np.sum((grad_u_h - grad_u_ex)**2)
    return np.sqrt(error_sq)



def grad_u_exact(x, y):
    return (np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
            np.pi * np.sin(np.pi * x) * np.cos(np.pi * y))


def u_exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def solve_poisson(iter):


    nx, ny = 2**iter, 2**iter  # Number of elements in x and y direction
    # print(nx)
    nodes, elements, boundary_nodes = generate_structured_triangular_mesh(nx, ny)
    
    grads = reference_gradients()
    tri_nodes = np.array([[0, 0], [1, 0], [0, 1]])
    area, J = triangle_area_and_transform(tri_nodes)
    

    Ke = local_stiffness(tri_nodes)
    
    tri_nodes = np.array([[0, 0], [1, 0], [0, 1]])
    Ke = local_stiffness(tri_nodes)

    # test global stiffness matrix assembly
    K = assemble_global_matrix(nodes, elements)
    def f(x, y):
        return - 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    b = assemble_load_vector(nodes, elements, f)
    def g(x, y):
        return 0.
    

    K_mod, b_mod = apply_dirichlet_bc(K, b, boundary_nodes, g, nodes)

    # solve the system
    u = solve_system(K_mod, b_mod)

    L2_error = compute_L2_error(nodes, elements, u, u_exact)
    H1_error = compute_H1_error(nodes, elements, u, u_exact, grad_u_exact)
    return L2_error, H1_error

if __name__ == "__main__":

    for iter in range(1,6):
        L2_error, H1_error = solve_poisson(iter)

        # compute errors
        print(f"L2 error: {L2_error:.6f}",f"H1 error: {H1_error:.6f}")
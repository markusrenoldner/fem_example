

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.tri as mtri


from assembly import *
from utils import *
from mesh import *



def solve_poisson(n,f, g, plotting=False):

    nx, ny = n,n
    nodes, elements, boundary_nodes = generate_structured_triangular_mesh(nx, ny)

    if plotting: plot_mesh(nodes, elements)

    K = assemble_global_matrix(nodes, elements)
    # print(K)


    b = assemble_load_vector(nodes, elements, f)
    # print(b)


    K_mod, b_mod = apply_dirichlet_bc(K, b, boundary_nodes, g, nodes)

    u = solve_system(K_mod, b_mod)

    if plotting: plot_fem_solution(nodes, elements, u)


    L2_error = compute_L2_error_higher(nodes, elements, u, u_exact)
    H1_error = compute_H1_error(nodes, elements, u, grad_u_exact)

    return u, L2_error, H1_error

def conv_test(k_max, f, g):
    errors = []
    errors1 = []

    # solve poisson on several meshes
    for k in range(0,k_max):
        n=2**k
        
        u, L2_error, H1_error = solve_poisson(n, f, g)
        
        errors.append(L2_error)
        errors1.append(H1_error)

    # print L2 err
    print("L2 error \t\t rate")
    for i in range(0, len(errors)):
        rate = np.log(errors[i-1]/errors[i]) / np.log(2)
        if i==0: rate=0
        print(errors[i], "\t", rate)
    
    # print H1 err
    print("\nH1 error \t\t rate")
    for i in range(0,len(errors1)):
        rate = np.log(errors1[i-1]/errors1[i]) / np.log(2)
        if i==0: rate=0
        print(errors1[i], "\t", rate)

    return errors, errors1


def data(testcase=1):

    if testcase==1:

        def u_exact(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        def grad_u_exact(x, y):
            ux = np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)
            uy = np.pi * np.sin(np.pi*x) * np.cos(np.pi*y)
            return np.array([ux, uy])
        
        f = lambda x, y: +2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)

        g = lambda x, y: 0.0 

    elif testcase == 2:
        def u_exact(x, y):
            return np.sin(np.pi * x) * np.sinh(np.pi * y)

        def grad_u_exact(x, y):
            ux = np.pi * np.cos(np.pi*x) * np.sinh(np.pi*y)
            uy = np.pi * np.sin(np.pi*x) * np.cosh(np.pi*y)
            return np.array([ux, uy])

        f = lambda x, y: 0
        g = lambda x, y: u_exact(x,y) 

    elif testcase == 3:
        u_exact =lambda x,y: 3*x + 4*y + 5
        grad_u_exact =lambda x,y: np.array([3,4])
        f =lambda x,y: 0
        g =lambda x,y: u_exact(x,y) 

    elif testcase == 4:
        u_exact =lambda x,y: 3*x**2 + y + 5
        grad_u_exact =lambda x,y: np.array([6*x,1])
        f =lambda x,y: 6
        g =lambda x,y: u_exact(x,y) 

        print("for some reason test 4 fails\n")


    return u_exact, grad_u_exact, f, g


if __name__ == "__main__":

    np.set_printoptions(threshold=10000, precision=3, suppress=True, linewidth=1000)

    # define sol and RHS
    u_exact, grad_u_exact, f, g = data(testcase=2)

    # visualise solution
    solve_poisson(4, f, g, plotting=True) 

    # error conv
    errors, errors1 = conv_test(7, f, g)

    # conv plot
    conv_plot(errors, errors1)


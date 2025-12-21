
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




# TODO: use unstruc mesh from gmsh

import numpy as np
import matplotlib.pyplot as plt
import gmsh

def generate_gmsh_mesh(nx,ny):

    # mesh
    gmsh.initialize()
    gmsh.clear()
    gdim = 2
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(gdim, [rectangle], tag=1)
    gmsh.model.mesh.generate(gdim)
    gmsh.fltk.run() # view mesh
    gmsh.finalize()

    return

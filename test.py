

import numpy as np
import matplotlib.pyplot as plt
import gmsh


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

# local stiffness matrix



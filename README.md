
# custom P1 FEM assembly code

Custom 2D Lagrange P1 finite element assembly library that also constructs a simplicial (triangular) mesh from scratch. Built for educational purposes to illustrate local-to-global assembly, Dirichlet boundary conditions, solving, and error estimation on structured meshes.

Custom Lagrange P1 finite element library with built-in simplicial (triangular) mesh generation. Designed for education, demonstrating 

- mesh generation
- local-to-global assembly
- Dirichlet boundary conditions, 
- system solving, 
- error estimation


## file tree

```
fem_example
├── LICENSE
├── README.md
└── src
	├── FEM1d.py                     # 1D example
	├── FEM2d_minimal.py             # minimal 2D example
	├── fenicsx.py                   # fenicsx comparison
	├── myfemlib
	│   ├── assembly.py              # assemble matrix and rhs
	│   ├── mesh.py                  # create mesh
	│   ├── poisson.py               # assembly FEM problem
	│   ├── utils.py                 # plotting and FEM-errors
	├── matrix_only
	│   ├── fenicsx_vs_mymatrix.py   # for debugging
	│   ├── justmatrix.py            # for debugging
	│   └── mat_fenicsx.py           # for debugging
	└── archive                      # outdated
```

## install and run

```
# Clone
cd ~
git clone git@github.com:markusrenoldner/fem_example.git
cd fem_example

# (Optional) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install numpy matplotlib

# Run the solver with plotting and convergence
python src/myfemlib/poisson.py

# Optional: FEniCSx example (requires dolfinx)
# Follow dolfinx install guide, then:
python misc/fenicsx.py
```



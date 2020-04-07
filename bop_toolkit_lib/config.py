# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

######## Basic ########

# Folder with the BOP datasets.
datasets_path = r'/content/download'

# Folder with pose results to be evaluated.
results_path = r'/content/results'

# Folder for the calculated pose errors and performance scores.
eval_path = r'/content/eval'

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r'/content/outputs'

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/content/renderer_cpp'

# Executable of the MeshLab server.
meshlab_server_path = r'/path/to/meshlabserver.exe'

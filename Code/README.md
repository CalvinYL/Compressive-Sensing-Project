The first of our implementation is from the Hou’s saliency model, which implements
an algorithm based only on a spectral residual approach to generate a saliency map
from RGB images. 

The second is known as the Low-rank and Structured sparse
Matrix Decomposition (SMD) model which uses low-rank and structured-sparsity
regularization for image backgrounds and salient objects, respectively. This model
first extracts features from an image (such as textures and colors) into a matrix, and
takes prior knowledge (such as location and background) to form a high-level prior
map. It then decomposes the feature matrix F into the summation of a low-rank
part L and a structured-sparse S separating the salient object from the background
and producing the saliency map.

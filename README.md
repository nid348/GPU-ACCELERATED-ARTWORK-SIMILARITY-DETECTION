GPU-Accelerated Artwork Similarity Detection
This project implements a high-performance artwork similarity detection system using CUDA-enabled GPU acceleration. It combines color-based and structural features to accurately identify visually similar artworks and detect potential plagiarism.

Key Features
GPU-accelerated cosine similarity using Numba CUDA
Feature extraction using color histograms (512-dim vectors)
Structural similarity via ORB keypoint matching
Combined similarity scoring for robust detection
Significant speedup over CPU-based approaches

Tech Stack
Python
CUDA (Numba JIT)
OpenCV
NumPy
Matplotlib

Methodology
Image preprocessing (resize, normalize)
Feature extraction (color histogram)
GPU-based cosine similarity computation
ORB keypoint matching (structural similarity)
Final similarity score:
50% color similarity
50% structural similarity

Results
Successfully distinguishes similar vs unrelated artworks
AI-generated images show high similarity to reference artwork
Unrelated images (e.g., Mona Lisa) score significantly lower
Achieves faster computation using GPU parallelism

How to Run
pip install numba opencv-python numpy matplotlib

Applications
Art plagiarism detection
Content-based image retrieval
Digital copyright protection

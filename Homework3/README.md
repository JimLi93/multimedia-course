# Homework 3

## Motion Estimation Using Block Matching Methods: Full Search and 2D Logarithmic Search

### Goal
Implement the full search and the 2D logarithmic search method to find motion vectors for motion estimation.

### Command 
`python3 hw3.py` 

### Image used
`img/40.jpg img/42.jpg img/51.jpg`

## Method

Use two search ranges (8 , 16) for two macroblock sizes (8x8 and 16x16) for two search methods.

### Result

It will save 24 images in `out` folder.
* 8 images for the predicted images using different method, search range, and macroblock size.
* 8 images for motion vectors images.
* 8 images for residual images.


# Homework 1

## Part 1 : Color Quantization and Dithering

### Goal
Implement median-cut color quantization to transform 24-bit color image to n-bit color image.

Design diffusion dithering technique.

### Command 
`python3 1.py` (About 1 minute)

### Image used
`img/Lenna.jpg`

### Result
It will print out the MSE of with and without error diffusion dithering after color quantization( `n=3` and `n=6` ).

Also, it will save 4 images in `out` folder.
* `median_cut3.jpg` : 3-bit color image
* `median_cut6.jpg` : 6-bit color image
* `error_diffusion_dithering_3.jpg` : 3-bit color image with diffusion dithering
* `error_diffusion_dithering_6.jpg` : 6-bit color image with diffusion dithering

## Part 2 : Interpolation

### Goal
Implement Nearest-neighbor interpolation and Bilinear interpolation to upsample image to 4 times the width and height.

### Command 
`python3 2.py`

### Image used
`img/bee.jpg`

### Result
It will save 2 images in `out` folder.
* `bee_near.jpg` : The result of nearest-neighbor interpolation.
* `bee_linear.jpg` : The result of bilinear interpolation.

## Part 3 : Photo Enhancement

### Goal
Implement photo enhancement by converting RGB color space to YIQ, do gamma transform to Y channel, and convert back to RGB.

### Command 
`python3 3.py`

### Image used
`img/lake.jpg`

### Result

It will save 3 images in `out` folder.
* `Y_hist.jpg`: The histogram of Y channel after RGB color space to YIQ.
* `Y_hist_gamma.jpg`: The histogram of Y channel after RGB color space to YIQ and gamma transform.
* `gamma_img.jpg`: The enhanced image.
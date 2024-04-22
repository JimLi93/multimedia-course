# Homework 2

## Part 1: DCT Image Compression

### Goal
Implement DCT compression and inverse DCT with `n = 2, 4` and `m = 4, 8`. 

### Command 
`python3 Q1/1.py` 

### Image used
`barbara.jpg` `cat.jpg`

### Result

It will print out the PSNR of 16 images, including 8 from `barbara.jpg` and `cat.jpg`
Also, it will save 16 images in `output` folder.
* 4 images of different `n & m` for 2 input images. (Total 8 images)
* 4 images of different `n & m` for 2 input images with transforming to YCbCr color space. (Total 8 images)

## Part 2: Create your own FIR filter to filter audio signal

### Goal
Implement FIR filters to filter audio signal out of the mixed of 3 songs.

### Command 
`python3 Q2/2.py` (About 40 minutes)

### Audio used
`HW2_Mix.wav`

It will save image results and audio results in `output` folder.

### Result
Image results
* `input.png` :<br> The spectrum of the input audio signal.
* `output_by_Lowpass.png output_by_Highpass.png output_by_Bandpass.png` :<br> The spectrums of the output signals after applying three different filters to the input audio signal.
* `Lowpass_spectrum.png Highpass_spectrum.png Bandpass_spectrum.png` : <br>The spectrum of three filters, namely low-pass filter, high-pass filter, and bandpass filter.
* `Lowpass_shape.png Highpass_shape.png Bandpass_shape.png` : <br>The shape of three filters.

Audio results
* `Lowpass_400.wav Highpass_720.wav Bandpass_400_720.wav` : <br>Output signals after applying three filters.
* `Lowpass_400_2kHZ.wav Highpass_720_2kHZ.wav Bandpass_400_720_2kHZ.wav` : <br>Output signals after applying three filters and reducing the sampling rate. 
* `Echo_one.wav` : <br>Output signal after applying low-pass filter and one-fold echo.
* `Echo_multiple.wav` : <br>Output signal after applying low-pass filter and multiple-fold echo.

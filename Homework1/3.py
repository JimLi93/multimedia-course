import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# Read image and convert
original_lake_img = cv2.imread('img/lake.jpg')
lake_img = cv2.cvtColor(original_lake_img, cv2.COLOR_BGR2RGB)
#plt.imshow(lake_img)
#print(lake_img.shape)

#Convert RGB to YIQ
rows, cols, channels = lake_img.shape
output_lake_YIQ = np.zeros((rows, cols, channels))
for i in range(rows):
  for j in range(cols):
    output_lake_YIQ[i][j][0] = 0.299 * lake_img[i][j][0] + 0.587 * lake_img[i][j][1] + 0.114 * lake_img[i][j][2]
    output_lake_YIQ[i][j][1] = 0.595879 * lake_img[i][j][0] - 0.274133 * lake_img[i][j][1] - 0.321746 * lake_img[i][j][2]
    output_lake_YIQ[i][j][2] = 0.211205 * lake_img[i][j][0] - 0.523083 * lake_img[i][j][1] + 0.311878 * lake_img[i][j][2]

#Function histogram
def create_histogram(input_img, channel):
  histogram = [0] * 256
  for i in range(rows):
    for j in range(cols):
      tmp = round(input_img[i][j][channel])
      histogram[tmp] += 1
  return histogram

#Create histogram of channel Y
histogram_Y = create_histogram(output_lake_YIQ, 0)

#Function Gamma Transform
def gamma_trans(input_img, gamma):
  output_img = np.zeros(input_img.shape)
  rows, cols, channels = input_img.shape
  for i in range(rows):
    for j in range(cols):
        output_img[i][j][0] = math.pow((input_img[i][j][0] / 255), gamma) * 255
        output_img[i][j][1] = input_img[i][j][1]
        output_img[i][j][2] = input_img[i][j][2]
  return output_img

#Run gamma transform and create histogram of channel Y
output_lake_YIQ_gt = gamma_trans(output_lake_YIQ, 4.5)
histogram_Y_gt = create_histogram(output_lake_YIQ_gt, 0)
#plt.plot(histogram_Y_gt)

#Convert YIQ to RGB
output_lake_final = np.zeros((rows, cols, channels))
for i in range(rows):
  for j in range(cols):
    output_lake_final[i][j][0] = 1.0000 * output_lake_YIQ_gt[i][j][0] + 0.95630 * output_lake_YIQ_gt[i][j][1] + 0.62103 * output_lake_YIQ_gt[i][j][2]
    output_lake_final[i][j][1] = 1.0000 * output_lake_YIQ_gt[i][j][0] - 0.27256 * output_lake_YIQ_gt[i][j][1] - 0.64671 * output_lake_YIQ_gt[i][j][2]
    output_lake_final[i][j][2] = 1.0000 * output_lake_YIQ_gt[i][j][0] - 1.10474 * output_lake_YIQ_gt[i][j][1] + 1.70116 * output_lake_YIQ_gt[i][j][2]

for i in range(rows):
    for j in range(cols):
        for k in range(channels):
            if(output_lake_final[i][j][k] > 255):
                output_lake_final[i][j][k] = 255
            if(output_lake_final[i][j][k] < 0):
                output_lake_final[i][j][k] = 0

plt.imsave("out/gamma_img.jpg", output_lake_final/255)

plt.plot(histogram_Y)
plt.savefig("out/Y_hist.jpg")

plt.clf()
plt.plot(histogram_Y_gt)
plt.savefig("out/Y_hist_gamma.jpg")

#plt.imsave("out/Y_hist.jpg", histogram_Y)
#plt.imsave("out/Y_hist_gamma.jpg", histogram_Y_gt)

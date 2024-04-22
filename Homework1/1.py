import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

#read image
original_img = cv2.imread('img/Lenna.jpg')

#convert
img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

#plt.imshow(img)

#Create flatten array
rows,cols,channels = img.shape
flatten_img = []
for i in range(rows):
  for j in range(cols):
    flatten_img.append([img[i][j][0], img[i][j][1], img[i][j][2], i, j, 0, 0, 0])
flatten_img = np.array(flatten_img)

#Function median cut
def median_cut(stage_rgb, rounds, input_flatten_img, output_img, output_lup, lup, r, g, b, n):
  if(rounds <= 0):
    #write output_img
    r_mean = round(np.mean(input_flatten_img[:, 0]))
    g_mean = round(np.mean(input_flatten_img[:, 1]))
    b_mean = round(np.mean(input_flatten_img[:, 2]))

    tmp = math.pow(4,n) * r + math.pow(2,n) * g + b
    tmp = int(tmp)

    for i in range(len(input_flatten_img)):
      output_img[input_flatten_img[i][3]][input_flatten_img[i][4]] = [r_mean, g_mean, b_mean]
      output_lup[input_flatten_img[i][3]][input_flatten_img[i][4]] = tmp
    lup[tmp] = [r_mean, g_mean, b_mean]
    return
  input_flatten_img = input_flatten_img[input_flatten_img[:,stage_rgb].argsort()]
  median_idx = int((len(input_flatten_img) + 1)/2)
  #print(input_flatten_img[median_idx][stage_rgb], stage_rgb, rounds)
  #threshold_color.append(input_flatten_img[median_idx][stage_rgb])
  #print(median_idx)
  #print(stage_rgb, rounds)
  for i in range(median_idx):
    input_flatten_img[i][5+stage_rgb] = input_flatten_img[i][5+stage_rgb] * 2
  for i in range(median_idx, len(input_flatten_img)):
    input_flatten_img[i][5+stage_rgb] = input_flatten_img[i][5+stage_rgb] * 2 + 1
  if stage_rgb == 2:
    median_cut(0, rounds - 1, input_flatten_img[0:median_idx], output_img, output_lup, lup, r, g, b*2, n)
    median_cut(0, rounds - 1, input_flatten_img[median_idx:], output_img, output_lup, lup, r, g, b*2+1, n)
  elif stage_rgb == 1:
    median_cut(2, rounds, input_flatten_img[0:median_idx], output_img, output_lup, lup, r, g*2, b, n)
    median_cut(2, rounds, input_flatten_img[median_idx:], output_img, output_lup, lup, r, g*2+1, b, n)
  else :
    median_cut(1, rounds, input_flatten_img[0:median_idx], output_img, output_lup, lup, r*2, g, b, n)
    median_cut(1, rounds, input_flatten_img[median_idx:], output_img, output_lup, lup, r*2+1, g, b, n)
  return output_img, output_lup, lup

#Get 3-bit image and 6-bit image and their look-up-table
output_img_n3 = np.zeros(img.shape)
output_img_n6 = np.zeros(img.shape)
output_lup_n3 = np.zeros(img.shape)
output_lup_n6 = np.zeros(img.shape)
lup_n3 = np.zeros((8,3))
lup_n6 = np.zeros((64,3))

output_img_n3, output_lup_n3, lup_n3 = median_cut(0,1,flatten_img, output_img_n3, output_lup_n3, lup_n3, 0, 0, 0, 1)
output_img_n6, output_lup_n6, lup_n6 = median_cut(0,2,flatten_img, output_img_n6, output_lup_n6, lup_n6, 0, 0, 0, 2)

'''
Check if the color is out of range
for i in range(rows):
  for j in range(cols):
    tmp = 0
    for k in range(8):
      if tmp == 1:
        break
      if(output_img_n3[i][j][0] == lup_n3[k][0] and output_img_n3[i][j][1] == lup_n3[k][1] and output_img_n3[i][j][2] == lup_n3[k][2]):
        tmp = 1
    if(tmp == 0):
      print("ERROR")
'''

#plt.imshow((output_img_n3).astype(np.uint8))
#plt.imshow((output_img_n6).astype(np.uint8))

#Calculate mean squared quantization error
mean_error_n3 = 0
mean_error_n6 = 0
for i in range(rows):
  for j in range(cols):
    for k in range(channels):
      tmp_n3 = img[i][j][k] - output_img_n3[i][j][k]
      tmp_n6 = img[i][j][k] - output_img_n6[i][j][k]
      mean_error_n3 += tmp_n3 * tmp_n3
      mean_error_n6 += tmp_n6 * tmp_n6
mean_error_n3 = mean_error_n3 / rows / cols/ channels
mean_error_n6 = mean_error_n6 / rows / cols/ channels
print("After Color Quantization")
print("MSE of n=3", mean_error_n3)
print("MSE of n=6", mean_error_n6)

#Error Diffusion Dithering
output_img_n3_edd = np.zeros(img.shape)
output_img_n6_edd = np.zeros(img.shape)
for i in range(rows):
  for j in range(cols):
    for k in range(channels):
      output_img_n3_edd[i][j][k] = img[i][j][k]

for i in range(rows):
  for j in range(cols):
    for k in range(channels):
      output_img_n6_edd[i][j][k] = img[i][j][k]

up = 255
down = 0
for j in range(cols):
  for i in range(rows):
    min_dist = 10000000
    min_idx = 100
    for x in range(8):
      tmp = 0
      for y in range(3):
        tmp += (output_img_n3_edd[i][j][y] -  lup_n3[x][y]) * (output_img_n3_edd[i][j][y] -  lup_n3[x][y])
      if(tmp < min_dist):
        min_idx = x
        min_dist = tmp
    for k in range(3):
        old_color = output_img_n3_edd[i][j][k]
        quant_error = old_color - lup_n3[min_idx][k]
        if(i+1 < rows):
          output_img_n3_edd[i+1][j][k] += quant_error * 7/16.0
          if output_img_n3_edd[i+1][j][k] > up:
            output_img_n3_edd[i+1][j][k] = up
          elif output_img_n3_edd[i+1][j][k] < down:
            output_img_n3_edd[i+1][j][k] = down
        if(i-1 >= 0 and j+1 < cols):
          output_img_n3_edd[i-1][j+1][k] += quant_error * 3/16.0
          if output_img_n3_edd[i-1][j+1][k] > up:
            output_img_n3_edd[i-1][j+1][k] = up
          elif output_img_n3_edd[i-1][j+1][k] < down:
            output_img_n3_edd[i-1][j+1][k] = down
        if(j+1 < cols):
          output_img_n3_edd[i][j+1][k] += quant_error * 5/16.0
          if output_img_n3_edd[i][j+1][k] > up:
            output_img_n3_edd[i][j+1][k] = up
          elif output_img_n3_edd[i][j+1][k] < down:
            output_img_n3_edd[i][j+1][k] = down
        if(i+1 < rows and j+1 < cols):
          output_img_n3_edd[i+1][j+1][k] += quant_error * 1/16.0
          if output_img_n3_edd[i+1][j+1][k] > up:
            output_img_n3_edd[i+1][j+1][k] = up
          elif output_img_n3_edd[i+1][j+1][k] < down:
            output_img_n3_edd[i+1][j+1][k] = down



up = 255
down = 0
for j in range(cols):
  for i in range(rows):
    min_dist = 10000000
    min_idx = 100
    for x in range(64):
      tmp = 0
      for y in range(channels):
        tmp += (output_img_n6_edd[i][j][y] -  lup_n6[x][y]) * (output_img_n6_edd[i][j][y] -  lup_n6[x][y])
      if(tmp < min_dist):
        min_idx = x
        min_dist = tmp
    for k in range(channels):
        old_color = output_img_n6_edd[i][j][k]
        quant_error = old_color - lup_n6[min_idx][k]
        if(i+1 < rows):
          output_img_n6_edd[i+1][j][k] += quant_error * 7/16.0
          if output_img_n6_edd[i+1][j][k] > up:
            output_img_n6_edd[i+1][j][k] = up
          elif output_img_n6_edd[i+1][j][k] < down:
            output_img_n6_edd[i+1][j][k] = down
        if(i-1 >= 0 and j+1 < cols):
          output_img_n6_edd[i-1][j+1][k] += quant_error * 3/16.0
          if output_img_n6_edd[i-1][j+1][k] > up:
            output_img_n6_edd[i-1][j+1][k] = up
          elif output_img_n6_edd[i-1][j+1][k] < down:
            output_img_n6_edd[i-1][j+1][k] = down
        if(j+1 < cols):
          output_img_n6_edd[i][j+1][k] += quant_error * 5/16.0
          if output_img_n6_edd[i][j+1][k] > up:
            output_img_n6_edd[i][j+1][k] = up
          elif output_img_n6_edd[i][j+1][k] < down:
            output_img_n6_edd[i][j+1][k] = down
        if(i+1 < rows and j+1 < cols):
          output_img_n6_edd[i+1][j+1][k] += quant_error * 1/16.0
          if output_img_n6_edd[i+1][j+1][k] > up:
            output_img_n6_edd[i+1][j+1][k] = up
          elif output_img_n6_edd[i+1][j+1][k] < down:
            output_img_n6_edd[i+1][j+1][k] = down


#Find nearest color in look up table
for i in range(rows):
  for j in range(cols):
    min_dist = 1000000
    min_idx = 0
    for k in range(8):
      tmp = 0
      for x in range(channels):
        tmp += math.pow((output_img_n3_edd[i][j][x] -  lup_n3[k][x]), 2)
      if(tmp < min_dist):
        min_idx = k
        min_dist = tmp
    output_img_n3_edd[i][j][0] = lup_n3[min_idx][0]
    output_img_n3_edd[i][j][1] = lup_n3[min_idx][1]
    output_img_n3_edd[i][j][2] = lup_n3[min_idx][2]

for i in range(rows):
  for j in range(cols):
    min_dist = 1000000
    min_idx = 0
    for k in range(64):
      tmp = 0
      for x in range(channels):
        tmp += math.pow((output_img_n6_edd[i][j][x] -  lup_n6[k][x]), 2)
      if(tmp < min_dist):
        min_idx = k
        min_dist = tmp
    output_img_n6_edd[i][j][0] = lup_n6[min_idx][0]
    output_img_n6_edd[i][j][1] = lup_n6[min_idx][1]
    output_img_n6_edd[i][j][2] = lup_n6[min_idx][2]
    
#plt.imshow((output_img_n3_edd).astype(np.uint8))
#plt.imshow((output_img_n6_edd).astype(np.uint8))

#Calculate mean squared quantization error
mean_error_n3_edd = 0
mean_error_n6_edd = 0
for i in range(rows):
  for j in range(cols):
    for k in range(channels):
      tmp_n3_edd = img[i][j][k] - output_img_n3_edd[i][j][k]
      tmp_n6_edd = img[i][j][k] - output_img_n6_edd[i][j][k]
      mean_error_n3_edd += tmp_n3_edd * tmp_n3_edd
      mean_error_n6_edd += tmp_n6_edd * tmp_n6_edd
mean_error_n3_edd = mean_error_n3_edd / rows / cols/ channels
mean_error_n6_edd = mean_error_n6_edd / rows / cols/ channels
print("After Error Diffusion Dithering")
print("MSE of n=3", mean_error_n3_edd)
print("MSE of n=6", mean_error_n6_edd)

plt.imsave("out/median_cut3.png", output_img_n3/255)
plt.imsave("out/median_cut6.png", output_img_n6/255)
plt.imsave("out/error_diffusion_dithering_3.png", output_img_n3_edd/255)
plt.imsave("out/error_diffusion_dithering_6.png", output_img_n6_edd/255)
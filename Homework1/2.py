import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

###  Nearest-neighbor interpolation

# Read image and convert
original_bee_img = cv2.imread('img/bee.jpg')
bee_img = cv2.cvtColor(original_bee_img, cv2.COLOR_BGR2RGB)
#plt.imshow(bee_img)

#Nearest-neighbor interpolation
rows, cols, channels = bee_img.shape
output_bee_nni = np.zeros((4*rows, 4*cols, channels))
for i in range(4*rows):
  for j in range(4*cols):
    for k in range(channels):
      tmp_x = math.floor(i/4)
      tmp_y = math.floor(j/4)
      output_bee_nni[i][j][k] = bee_img[tmp_x][tmp_y][k]
      
#Save image
plt.imsave("out/bee_near.jpg", output_bee_nni/255)

### Bilinear Interpolation

#Bilinear Interpolation
output_bee_bi = np.zeros((4*rows, 4*cols, channels))
for i in range(4*rows):
  for j in range(4*cols):
      tmp_x = math.floor(i/4)
      tmp_y = math.floor(j/4)
      if(tmp_x + 1 < rows and tmp_y + 1 < cols):
        color_tmp = (((tmp_x+1) - i/4) * ((tmp_y+1) - j/4) * bee_img[tmp_x][tmp_y] + (i/4 - tmp_x) * ((tmp_y+1) - j/4) * bee_img[tmp_x+1][tmp_y] +
         ((tmp_x+1) - i/4) * (j/4 - tmp_y) * bee_img[tmp_x][tmp_y+1] + (i/4 - tmp_x) * (j/4 - tmp_y) * bee_img[tmp_x+1][tmp_y+1])
        output_bee_nni[i][j] = color_tmp
      elif(tmp_x + 1 < rows):
        color_tmp = ((tmp_x+1) - i/4) * bee_img[tmp_x][tmp_y] + (i/4 - tmp_x) * bee_img[tmp_x+1][tmp_y] 
      elif(tmp_y + 1 < cols):
        color_tmp = ((tmp_y+1) - j/4) * bee_img[tmp_x][tmp_y] + (j/4 - tmp_y) * bee_img[tmp_x][tmp_y+1] 
      else:
        color_tmp = bee_img[tmp_x][tmp_y]
      output_bee_bi[i][j] = color_tmp

#Save image
plt.imsave("out/bee_linear.jpg", output_bee_nni/255)
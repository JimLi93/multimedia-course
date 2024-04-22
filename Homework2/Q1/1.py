import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

original_img = cv2.imread('Q1/barbara.jpg')
img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
rows,cols,channels = img.shape

def DCT(input_img, block_size):
  rows,cols,channels = input_img.shape
  output_img = np.zeros(input_img.shape)
  r_t = int(rows / block_size)
  c_t = int(cols / block_size)
  T_block = np.zeros((block_size, block_size))
  for i in range(block_size):
    for j in range(block_size):
      if(i==0):
        T_block[i][j] = 0.5 / math.sqrt(2)
      else:
        T_block[i][j] = 0.5 * math.cos((2*j+1)*i*math.pi/16)
  for a in range(r_t):
    for b in range(c_t):
      for c in range(channels):
        block = input_img[a*block_size:(a+1)*block_size, b*block_size:(b+1)*block_size, c]
        result = np.dot(np.dot(T_block, block), np.linalg.inv(T_block))
        output_img[a*block_size:(a+1)*block_size, b*block_size:(b+1)*block_size, c] = result
  return output_img

def IDCT(input_img, block_size):
  rows,cols,channels = input_img.shape
  output_img = np.zeros(input_img.shape)
  r_t = int(rows / block_size)
  c_t = int(cols / block_size)
  T_block = np.zeros((block_size, block_size))
  for i in range(block_size):
    for j in range(block_size):
      if(i==0):
        T_block[i][j] = 0.5 / math.sqrt(2)
      else:
        T_block[i][j] = 0.5 * math.cos((2*j+1)*i*math.pi/16)
  for a in range(r_t):
    for b in range(c_t):
      for c in range(channels):
        block = input_img[a*block_size:(a+1)*block_size, b*block_size:(b+1)*block_size, c]
        result = np.dot(np.dot(np.linalg.inv(T_block), block), T_block)
        output_img[a*block_size:(a+1)*block_size, b*block_size:(b+1)*block_size, c] = result
  return output_img

def Round_co(input_img, block_size, n, q_table):
  rows,cols,channels = input_img.shape
  output_img = np.zeros(input_img.shape)
  r_t = int(rows / block_size)
  c_t = int(cols / block_size)
  tmp_min = np.zeros((channels))
  tmp_max = np.zeros((channels))
  for i in range(channels):
    tmp_min[i] = 100000
    tmp_max[i] = -100000
  for a in range(r_t):
    for b in range(c_t):
      for c in range(channels):
        for j in range(n):
          for k in range(n):
            output_img[a*block_size+j][b*block_size+k][c] = round(input_img[a*block_size+j][b*block_size+k][c] / q_table[j][k])
            if(output_img[a*block_size+j][b*block_size+k][c] > tmp_max[c]):
              tmp_max[c] = output_img[a*block_size+j][b*block_size+k][c]
            if(output_img[a*block_size+j][b*block_size+k][c] < tmp_min[c]):
              tmp_min[c] = output_img[a*block_size+j][b*block_size+k][c]

  return output_img, tmp_min, tmp_max

def uniform_quantize_co(input_img, block_size, min_value, max_value, m, n):
  rows,cols,channels = input_img.shape
  interval_len = np.zeros((channels))
  interval_len = (max_value - min_value) / (math.pow(2, m)-1)
  output_img = np.zeros(input_img.shape)
  r_t = int(rows / block_size)
  c_t = int(cols / block_size)
  for a in range(r_t):
    for b in range(c_t):
      for c in range(channels):
        for j in range(n):
          for k in range(n):
            value = input_img[a*block_size+j][b*block_size+k][c]
            output_img[a*block_size+j][b*block_size+k][c] = round((value - min_value[c]) / interval_len[c])
  return output_img

def inverse_uniform_quantize_co(input_img, block_size, min_value, max_value, m, n):
  rows,cols,channels = input_img.shape
  interval_len = np.zeros((channels))
  interval_len = (max_value - min_value) / (math.pow(2, m)-1)
  output_img = np.zeros(input_img.shape)
  r_t = int(rows / block_size)
  c_t = int(cols / block_size)
  for a in range(r_t):
    for b in range(c_t):
      for c in range(channels):
        for j in range(n):
          for k in range(n):
            value = input_img[a*block_size+j][b*block_size+k][c]
            value = min_value[c] + value * interval_len[c]
            output_img[a*block_size+j][b*block_size+k][c] = value
  return output_img

def inverse_Round_co(input_img, block_size, n, q_table):
  rows,cols,channels = input_img.shape
  output_img = np.zeros(input_img.shape)
  r_t = int(rows / block_size)
  c_t = int(cols / block_size)
  for a in range(r_t):
    for b in range(c_t):
      for c in range(channels):
        for j in range(n):
          for k in range(n):
            output_img[a*block_size+j][b*block_size+k][c] = input_img[a*block_size+j][b*block_size+k][c] * q_table[j][k]
  return output_img

def whole_process_a(input_img, block_size, n, m, q_table):
  
  DCT_img = DCT(input_img, 8)

  DCT_R_img, min_array, max_array = Round_co(DCT_img, block_size, n, q_table)

  DCT_R_Q_img = uniform_quantize_co(DCT_R_img, block_size, min_array, max_array, m, n)

  DCT_R_Q_UQ_img = inverse_uniform_quantize_co(DCT_R_Q_img, block_size, min_array, max_array, m, n)

  DCT_R_Q_UQ_IR_img = inverse_Round_co(DCT_R_Q_UQ_img, block_size, n, q_table)

  DCT_R_Q_UQ_IR_IDCT_img = IDCT(DCT_R_Q_UQ_IR_img, block_size)

  output_img = np.zeros(input_img.shape)
  for i in range(rows):
    for j in range(cols):
      for k in range(channels):
        output_img[i][j][k] = DCT_R_Q_UQ_IR_IDCT_img[i][j][k]
        if(DCT_R_Q_UQ_IR_IDCT_img[i][j][k] > 255):
          output_img[i][j][k] = 255
        elif(DCT_R_Q_UQ_IR_IDCT_img[i][j][k] < 0):
          output_img[i][j][k] = 0

  return output_img

def psnr(input1, input2):
  mse = np.mean((input1 - input2) ** 2)
  return 10 * np.log10(255*255/mse)

q_table1 = [[8,6,6,7,6,5,8,7],
      [7,7,9,9,8,10,12,20],
      [13,12,11,11,12,25,18,19],
      [15,20,29,26,31,30,29,26],
      [28,28,32,36,46,39,32,34],
      [44,35,28,28,40,55,41,44],
      [48,49,52,52,52,31,39,57],
      [61,56,50,60,46,51,52,50]]
bar_n2m4_a = whole_process_a(img, 8, 2, 4, q_table1)
bar_n2m8_a = whole_process_a(img, 8, 2, 8, q_table1)
bar_n4m4_a = whole_process_a(img, 8, 4, 4, q_table1)
bar_n4m8_a = whole_process_a(img, 8, 4, 8, q_table1)

plt.imsave("Q1/output/bar_n2m4_a.jpg", bar_n2m4_a/255)
plt.imsave("Q1/output/bar_n2m8_a.jpg", bar_n2m8_a/255)
plt.imsave("Q1/output/bar_n4m4_a.jpg", bar_n4m4_a/255)
plt.imsave("Q1/output/bar_n4m8_a.jpg", bar_n4m8_a/255)

def rgb2yCbCr(input_img):
  rows,cols,channels = input_img.shape
  Y = np.zeros((rows,cols,1))
  CbCr = np.zeros((int(rows/2),int(cols/2),2))
  output_img = np.zeros(input_img.shape)
  for i in range(rows):
    for j in range(cols):
      Y[i][j][0]=16+0.257*input_img[i][j][0]+0.564*input_img[i][j][1]+0.098*input_img[i][j][2]
      if(i%2==0 and j%2==0):
        CbCr[int(i/2)][int(j/2)][0]=128-0.148*input_img[i][j][0]-0.291*input_img[i][j][1]+0.439*input_img[i][j][2]
        CbCr[int(i/2)][int(j/2)][1]=128+0.439*input_img[i][j][0]-0.368*input_img[i][j][1]-0.071*input_img[i][j][2]
  return Y,CbCr

def yCbCr2rgb(Y, CbCr):
  rows,cols,channels = Y.shape
  output_img = np.zeros((rows, cols, 3))
  for i in range(rows):
    for j in range(cols):
      output_img[i][j][0] = 1.164*(Y[i][j][0]-16)+1.596*(CbCr[int(i/2)][int(j/2)][1]-128)
      output_img[i][j][1] = 1.164*(Y[i][j][0]-16)-0.382*(CbCr[int(i/2)][int(j/2)][0]-128)-0.813*(CbCr[int(i/2)][int(j/2)][1]-128)
      output_img[i][j][2] = 1.164*(Y[i][j][0]-16)+2.017*(CbCr[int(i/2)][int(j/2)][0]-128)

  return output_img

def whole_process_b(input_img, block_size, n, m, q_table1, q_table2):

  Y, C = rgb2yCbCr(input_img)

  DCT_Y = DCT(Y, 8)
  DCT_C = DCT(C, 8)

  DCT_R_Y, min_array, max_array = Round_co(DCT_Y, block_size, n, q_table1)
  DCT_R_C, min_array, max_array = Round_co(DCT_C, block_size, n, q_table2)

  DCT_R_Q_Y = uniform_quantize_co(DCT_R_Y, block_size, min_array, max_array, m, n)
  DCT_R_Q_C = uniform_quantize_co(DCT_R_C, block_size, min_array, max_array, m, n)

  DCT_R_Q_UQ_Y = inverse_uniform_quantize_co(DCT_R_Q_Y, block_size, min_array, max_array, m, n)
  DCT_R_Q_UQ_C = inverse_uniform_quantize_co(DCT_R_Q_C, block_size, min_array, max_array, m, n)

  DCT_R_Q_UQ_IR_Y = inverse_Round_co(DCT_R_Q_UQ_Y, block_size, n, q_table1)
  DCT_R_Q_UQ_IR_C = inverse_Round_co(DCT_R_Q_UQ_C, block_size, n, q_table2)

  DCT_R_Q_UQ_IR_IDCT_Y = IDCT(DCT_R_Q_UQ_IR_Y, block_size)
  DCT_R_Q_UQ_IR_IDCT_C = IDCT(DCT_R_Q_UQ_IR_C, block_size)

  output_img = yCbCr2rgb(DCT_R_Q_UQ_IR_IDCT_Y, DCT_R_Q_UQ_IR_IDCT_C)

  for i in range(rows):
    for j in range(cols):
      for k in range(channels):
        if(output_img[i][j][k] > 255):
          output_img[i][j][k] = 255
        elif(output_img[i][j][k] < 0):
          output_img[i][j][k] = 0

  return output_img

q_table_lum = [[16,11,10,16,24,40,51,61],
       [12,12,14,19,26,58,60,55],
       [14,13,16,24,40,57,69,56],
       [14,17,22,26,51,87,80,62],
       [18,22,37,56,68,109,103,77],
       [24,36,55,64,81,104,113,92],
       [49,64,78,87,103,121,120,101],
       [72,92,95,98,112,100,103,99]]
q_table_chr = [[17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99]]

bar_n2m4_b = whole_process_b(img, 8, 2, 4, q_table_lum, q_table_chr)
bar_n2m8_b = whole_process_b(img, 8, 2, 8, q_table_lum, q_table_chr)
bar_n4m4_b = whole_process_b(img, 8, 4, 4, q_table_lum, q_table_chr)
bar_n4m8_b = whole_process_b(img, 8, 4, 8, q_table_lum, q_table_chr)

plt.imsave("Q1/output/bar_n2m4_b.jpg", bar_n2m4_b/255)
plt.imsave("Q1/output/bar_n2m8_b.jpg", bar_n2m8_b/255)
plt.imsave("Q1/output/bar_n4m4_b.jpg", bar_n4m4_b/255)
plt.imsave("Q1/output/bar_n4m8_b.jpg", bar_n4m8_b/255)

print("PSNR of Barbara.jpg in process a")
print("n2m4 :", psnr(img, bar_n2m4_a))
print("n2m8 :", psnr(img, bar_n2m8_a))
print("n4m4 :", psnr(img, bar_n4m4_a))
print("n4m8 :", psnr(img, bar_n4m8_a))

print("PSNR of Barbara.jpg in process b")
print("n2m4 :", psnr(img, bar_n2m4_b))
print("n2m8 :", psnr(img, bar_n2m8_b))
print("n4m4 :", psnr(img, bar_n4m4_b))
print("n4m8 :", psnr(img, bar_n4m8_b))

original_img = cv2.imread('Q1/cat.jpg')
img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

cat_n2m4_a = whole_process_a(img, 8, 2, 4, q_table1)
cat_n2m8_a = whole_process_a(img, 8, 2, 8, q_table1)
cat_n4m4_a = whole_process_a(img, 8, 4, 4, q_table1)
cat_n4m8_a = whole_process_a(img, 8, 4, 8, q_table1)

plt.imsave("Q1/output/cat_n2m4_a.jpg", cat_n2m4_a/255)
plt.imsave("Q1/output/cat_n2m8_a.jpg", cat_n2m8_a/255)
plt.imsave("Q1/output/cat_n4m4_a.jpg", cat_n4m4_a/255)
plt.imsave("Q1/output/cat_n4m8_a.jpg", cat_n4m8_a/255)

cat_n2m4_b = whole_process_b(img, 8, 2, 4, q_table_lum, q_table_chr)
cat_n2m8_b = whole_process_b(img, 8, 2, 8, q_table_lum, q_table_chr)
cat_n4m4_b = whole_process_b(img, 8, 4, 4, q_table_lum, q_table_chr)
cat_n4m8_b = whole_process_b(img, 8, 4, 8, q_table_lum, q_table_chr)

plt.imsave("Q1/output/cat_n2m4_b.jpg", cat_n2m4_b/255)
plt.imsave("Q1/output/cat_n2m8_b.jpg", cat_n2m8_b/255)
plt.imsave("Q1/output/cat_n4m4_b.jpg", cat_n4m4_b/255)
plt.imsave("Q1/output/cat_n4m8_b.jpg", cat_n4m8_b/255)

print("PSNR of cat.jpg in process a")
print("n2m4 :", psnr(img, cat_n2m4_a))
print("n2m8 :", psnr(img, cat_n2m8_a))
print("n4m4 :", psnr(img, cat_n4m4_a))
print("n4m8 :", psnr(img, cat_n4m8_a))

print("PSNR of cat.jpg in process b")
print("n2m4 :", psnr(img, cat_n2m4_b))
print("n2m8 :", psnr(img, cat_n2m8_b))
print("n4m4 :", psnr(img, cat_n4m4_b))
print("n4m8 :", psnr(img, cat_n4m8_b))
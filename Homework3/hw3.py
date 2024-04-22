import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time

# Read the 3 images
org_img_40 = cv2.imread('img/40.jpg')
img_40 = cv2.cvtColor(org_img_40, cv2.COLOR_BGR2RGB)
org_img_42 = cv2.imread('img/42.jpg')
img_42 = cv2.cvtColor(org_img_42, cv2.COLOR_BGR2RGB)
org_img_51 = cv2.imread('img/51.jpg')
img_51 = cv2.cvtColor(org_img_51, cv2.COLOR_BGR2RGB)

# 2D logarithmic search
def twoD_log_search(ref_img, tar_img, p_range, block_size):
  rows,cols,channels = ref_img.shape
  # Define predicted image
  output_img = np.zeros(ref_img.shape)
  # Define motion vector
  motion_vector = np.copy(tar_img)
  # Find the search length n
  n_plum = math.floor(math.log(p_range,2))
  n = max(2, math.pow(2,(n_plum-1)))
  n = int(n)

  total_SAD = 0
  for x in range(0, rows, block_size):
    for y in range(0, cols, block_size):
      # Find the search length n
      n = max(2, math.pow(2,(n_plum-1)))
      n = int(n)
      # coordinate in reference image
      ans_x = x
      ans_y = y
      # while loop when search length n > 1
      while(n > 1):
        add_SAD, offset = find_min_block_offset(n, ref_img, tar_img, block_size, [x,y], [ans_x, ans_y])
        if(offset == [0,0]):
          n = n / 2
        else:
          ans_x = ans_x + offset[0]
          ans_y = ans_y + offset[1]
      # search length n = 1
      add_SAD, offset = find_min_block_offset(n, ref_img, tar_img, block_size, [x,y], [ans_x, ans_y])
      total_SAD += add_SAD
      ans_x = ans_x + offset[0]
      ans_y = ans_y + offset[1]
      ans_x = int(ans_x)
      ans_y = int(ans_y)
      # Assign macroblock to predicted image
      output_img[x:x+block_size,y:y+block_size] = ref_img[ans_x:ans_x+block_size,ans_y:ans_y+block_size]
      # Draw motion vector
      if(ans_x != x or ans_y != y):
        motion_vector = cv2.arrowedLine(motion_vector, (ans_y, ans_x), (y,x), (255,0,0),1)
  return output_img, motion_vector, total_SAD

def find_min_block_offset(S, ref_img, tar_img, block_size, tar_idx, center_idx):
  rows,cols,channels = ref_img.shape
  return_offset_idx = 0
  add_SAD = 0
  # S = 1, then 9 locations
  if(S == 1):
    M = [[0,0],[S,0],[0,S],[-S,0],[0,-S],[S,S],[S,-S],[-S,S],[-S,-S]]
    min_SAD = 1000000000
    for i in range(9):
      ref_idx = [0,0]
      for j in range(2):
        ref_idx[j] = center_idx[j] + M[i][j]
        ref_idx[j] = int(ref_idx[j])
      if(ref_idx[0]>= 0 and ref_idx[1] >= 0 and ref_idx[0]+block_size < rows and ref_idx[1]+block_size < cols):
        tar_block = tar_img[tar_idx[0]:tar_idx[0]+block_size,tar_idx[1]:tar_idx[1]+block_size]
        ref_block = ref_img[ref_idx[0]:ref_idx[0]+block_size,ref_idx[1]:ref_idx[1]+block_size]
        tmp_SAD = np.sum(np.abs(tar_block.astype(int) - ref_block.astype(int)))
        if(tmp_SAD < min_SAD):
          min_SAD = tmp_SAD
          return_offset_idx = i
    add_SAD += min_SAD
  # S != 1, then 5 locations
  else:
    M = [[0,0],[S,0],[0,S],[-S,0],[0,-S]]
    min_SAD = 1000000000
    for i in range(5):
      ref_idx = [0,0]
      # Find the coordinate in reference image
      for j in range(2):
        ref_idx[j] = center_idx[j] + M[i][j]
        ref_idx[j] = int(ref_idx[j])
      # Handle boundary condition and calculate SAD
      if(ref_idx[0]>= 0 and ref_idx[1] >= 0 and ref_idx[0]+block_size < rows and ref_idx[1]+block_size < cols):
        tar_block = tar_img[tar_idx[0]:tar_idx[0]+block_size,tar_idx[1]:tar_idx[1]+block_size]
        ref_block = ref_img[ref_idx[0]:ref_idx[0]+block_size,ref_idx[1]:ref_idx[1]+block_size]
        tmp_SAD = np.sum(np.abs(tar_block.astype(int) - ref_block.astype(int)))
        if(tmp_SAD < min_SAD):
          min_SAD = tmp_SAD
          return_offset_idx = i
    add_SAD += min_SAD
  return add_SAD, M[return_offset_idx]   

def full_search(ref_img, tar_img, p_range, block_size):
  rows,cols,channels = ref_img.shape
  # Define predicted image
  output_img = np.zeros(ref_img.shape)
  # Define motion vector
  motion_vector = np.copy(tar_img)
  total_SAD = 0
  for x in range(0, rows, block_size):
    for y in range(0, cols, block_size):
      # Search range start and end point
      start_x = x - p_range
      end_x = x + p_range
      start_y = y - p_range
      end_y = y + p_range
      # Boundary condition
      if(start_x < 0):
        start_x = 0
      if(start_y < 0):
        start_y = 0
      if(end_x+block_size > rows - 1):
        end_x = rows - block_size
      if(end_y+block_size > cols - 1):
        end_y = cols - block_size
      min_SAD = 1000000000
      # coordinate in reference image
      ans_x = x
      ans_y = y
      for i in range(start_x, end_x+1):
        for j in range(start_y, end_y+1):
          # Calculate SAD
          tar_block = tar_img[x:x+block_size,y:y+block_size]
          ref_block = ref_img[i:i+block_size,j:j+block_size]
          tmp_SAD = np.sum(np.abs(tar_block.astype(int) - ref_block.astype(int)))
          # Find min location and its coordinate
          if(tmp_SAD < min_SAD):
            min_SAD = tmp_SAD
            ans_x = i
            ans_y = j
      total_SAD += min_SAD
      # Assign macroblock to predicted image
      output_img[x:x+block_size,y:y+block_size] = ref_img[ans_x:ans_x+block_size,ans_y:ans_y+block_size]
      # Draw motion vector
      if(ans_x != x or ans_y != y):
        motion_vector = cv2.arrowedLine(motion_vector,(ans_y, ans_x), (y,x),  (255,0,0),1)

  return output_img, motion_vector, total_SAD

def calculate_psnr(img1, img2):
  mse = np.mean((img1-img2)**2)
  return 10 * np.log10(255*255/mse)

print("2D logarithmic search:")
for i in range(1,3):
  for j in range(1,3):
    start_time = time.time()
    pred_img, motion_vector, total_SAD = twoD_log_search(img_40, img_42, 8*i, 8*j)
    end_time = time.time()
    path = "out/2d_predicted_r"+str(8*i)+"_b"+str(8*j)+".jpg"
    plt.imsave(path, pred_img/255)
    path = "out/2d_motion_vector_r"+str(8*i)+"_b"+str(8*j)+".jpg"
    plt.imsave(path, motion_vector/255)
    path = "out/2d_residual_r"+str(8*i)+"_b"+str(8*j)+".jpg"
    plt.imsave(path, abs(img_42 - pred_img)/255)
    print("search range: %2d  block size: %2d " % (8*i , 8*j))
    print("psnr: %f" % calculate_psnr(pred_img, img_42))
    print("SAD: ", total_SAD)
    print("time: ", end_time - start_time)

print("Full search:")
for i in range(1,3):
  for j in range(1,3):
    start_time = time.time()
    pred_img, motion_vector, total_SAD = full_search(img_40, img_42, 8*i, 8*j)
    end_time = time.time()
    path = "out/full_predicted_r"+str(8*i)+"_b"+str(8*j)+".jpg"
    plt.imsave(path, pred_img/255)
    path = "out/full_motion_vector_r"+str(8*i)+"_b"+str(8*j)+".jpg"
    plt.imsave(path, motion_vector/255)
    path = "out/full_residual_r"+str(8*i)+"_b"+str(8*j)+".jpg"
    plt.imsave(path, abs(img_42 - pred_img)/255)
    print("search range: %2d  block size: %2d " % (8*i , 8*j))
    print("psnr: %f" % calculate_psnr(pred_img, img_42))
    print("SAD: ", total_SAD)
    print("time: ", end_time - start_time)


# reference img = 40.jpg target img = 51.jpg p=8 block=8*8
r = 8
b = 8
start_time = time.time()
pred_img, motion_vector, total_SAD = full_search(img_40, img_51, r, b)
#plt.imsave("out/pred_51.jpg", pred_img/255)
end_time = time.time()
print("")
print("ref: 40.jpg, tar: 51.jpg")
print("search range: %2d  block size: %2d " % (r , b))
print("psnr: %f" % calculate_psnr(pred_img, img_51))
print("SAD: ", total_SAD)
print("time: ", end_time - start_time)

'''
# reference img = 40.jpg target img = 51.jpg p=32 block=8*8
r = 32
b = 8
start_time = time.time()
pred_img, motion_vector, total_SAD = full_search(img_40, img_51, r, b)
#plt.imsave("out/pred_51.jpg", pred_img/255)
end_time = time.time()
print("ref: 40.jpg, tar: 51.jpg")
print("search range: %2d  block size: %2d " % (r , b))
print("psnr: %f" % calculate_psnr(pred_img, img_51))
print("SAD: ", total_SAD)
print("time: ", end_time - start_time)
'''
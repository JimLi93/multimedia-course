import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def subplot(points , result1 , result2 , img):


    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1],  s=0.5)
    plt.plot(result1[:, 0], result1[:, 1], 'b-' ,linewidth=0.5)

    plt.plot(result2[:, 0], result2[:, 1], 'r-' ,linewidth=0.5)
    plt.savefig('output/1a.png')
    plt.close()
    
    
    ##for(b)
def plot(points , result , img):
    plt.imshow(img/255)
    plt.scatter(points[:, 0], points[:, 1],  s=5)
    plt.plot(result[:, 0], result[:, 1], 'r-' ,linewidth=0.5)
    plt.savefig('output/1b.png')
    plt.close()


def create_curve(points, detail_point_amount):
  output = []
  for i in range(0,len(points)-1, 3):
    for j in range(detail_point_amount):
      t = 1/detail_point_amount*j
      output.append(math.pow((1-t),3) * points[i] \
                        + 3 * t * math.pow((1-t),2) * points[i+1] \
                        + 3 * t * t * (1-t) * points[i+2] \
                        + math.pow(t,3) * points[i+3])
  output.append(points[len(points)-1])
  output = np.array(output)
  return output

def scale_img(img, f):
  rows, cols, channels = img.shape
  output_img = np.zeros((rows*f, cols*f, channels))
  for i in range(f*rows):
    for j in range(f*cols):
      #for k in range(channels):
        tmp_x = round(i/f)
        tmp_y = round(j/f)
        if(tmp_x >= rows-1):
          tmp_x = rows-1
        if(tmp_y >= cols-1):
          tmp_y = cols-1
        output_img[i][j] = img[tmp_x][tmp_y]
  return output_img
    
    
def main():      
    # Load the image and points
    img = cv2.imread("bg.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points = np.loadtxt("points.txt")
    
    ##You shold modify result1 , result2 , result
    ## 1.a
    result1 = create_curve(points, 2)
    result2 = create_curve(points, 100)
    subplot(points  , result1 , result2 , img)
    
    # 2.a 
    scaled_img = scale_img(img, 4)
    plt.imshow(scaled_img/255)
    result = create_curve(points*4, 100)
    #print(result)
    plot(points*4  , result , scaled_img)
    
if __name__ == "__main__":
    main()
import numpy as np
import cv2

def cloud_recognition(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    width, height, _ = img.shape
    dummy = img.copy()
    for x in range(0,width):
        for y in range(0,height):
            if not((hsv[x,y][1]/256 > 0.25 and dummy[x, y][0] > dummy[x, y][2]) or (dummy[x, y][2] >= 100 and hsv[x, y][1]/256 <= 0.18 and dummy[x, y][2]/dummy[x, y][0] <= 0.9) or (hsv[x, y][0]/256 <= 0.2 and hsv[x, y][1]/256 <= 0.2) or (dummy[x, y][2] <= 80 and hsv[x, y][0]/256 <= 0.6 and hsv[x, y][1]/256 >= 0.35)):
                dummy[x,y] = (255,255,255)  
            else:
                dummy[x,y] = (0,0,0)  
    kernel = np.matrix('0 1 1 1 0; 0 1 1 1 0; 0 0 1 0 0; 0 1 1 1 0; 0 1 1 1 0', np.uint8)
    kernel_d = np.ones((7,7), np.uint8)
    bw = cv2.cvtColor(dummy, cv2.COLOR_BGR2GRAY) 
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)  
    dilated = cv2.dilate(closed, kernel_d,iterations = 5)
    eroded = cv2.erode(dilated,kernel,iterations = 5)       
    return closed

if __name__ == "__main__":
    
    # Blue level threshold to mask the sky
    low_blue = np.array([100,0,0])
    high_blue = np.array([255,255,255])
    
    img_src = cv2.imread("img/nubes_azotea/2esquina/100_0134.jpg")
    img_bw = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(img_src, low_blue, high_blue)
    res = cv2.bitwise_and(img_src, img_src, mask=mask)
    
    final = cloud_recognition(img_src)
    
    cv2.imshow("Original", img_src)
    cv2.imshow("res", final)
    
    k = cv2.waitKey(0)
    
    #cv2.imwrite('img/resultado/segmentacion.jpg', res)


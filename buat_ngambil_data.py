import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

#ABCD SETTINGS ------------------------------ STARTS
#CHOOSE IMAGE
img = cv2.imread('try1.jpg')
imy,imx,imz = np.shape(img)
shortening = 5
img = img[shortening:imy-shortening,shortening:imx-shortening,:]
imy,imx,imz = np.shape(img)
gs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#FD MAJOR AND MINOR AXIS 
maj_axis = 50
mnr_axis = 25

#ABCD SETTINGS ------------------------------ ENDS

ret1,th1 = cv2.threshold(gs,25,255,cv2.THRESH_BINARY)

ckernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
closing = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, ckernel,iterations = 4)

kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
erosion = cv2.erode(closing,kernel,iterations = 50)

blur = cv2.GaussianBlur(gs,(5,5),0)
ret,th = cv2.threshold(gs,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

nmask =  cv2.bitwise_and(erosion, th)

im2,contours,hierarchy = cv2.findContours(nmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contimg = img.copy()

height, width, channels = img.shape
all_cnt_bimg = np.zeros((height,width,1), np.uint8)
cv2.drawContours(all_cnt_bimg, contours, -1, 255, 1)



if len(contours) != 0:
    #find the biggest area
    c = max(contours, key = cv2.contourArea)

    # draw in blue the contours that were founded
    cv2.drawContours(contimg, c, -1, 255, 3)

    x,y,w,h = cv2.boundingRect(c)
    # draw the book contour (in green)
    cv2.rectangle(contimg,(x,y),(x+w,y+h),(0,255,0),2)

    #Finding center of mass
    M = cv2.moments(c)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    t_cx = cx-(imx/2)
    t_cy = cy-(imy/2)
    

    #CREATING BORDER 
    square_r = np.maximum(imx,imy)
    bimg = np.zeros((height,width,1), np.uint8)
    cv2.drawContours(bimg, c, -1, 255, 1)
    flood_img = bimg.copy()
    maskfill = np.zeros((height+2, width+2), np.uint8)    
    cv2.floodFill(flood_img, maskfill, (cx,cy), 255)
    segment = cv2.bitwise_and(img, img, mask=flood_img)
    






    
    
#-------------------------------------------------------------- ASSYMETRY START

    rhalf = (square_r/2).astype(int)

    nc_img = np.zeros((square_r,square_r,1), np.uint8)
    
    (r_x,r_y),(MA,ma),angle = cv2.minAreaRect(c)
    box = cv2.boxPoints(((r_x,r_y),(MA,ma),angle))
    box = np.int0(box)
    nf = np.zeros((imy,imx,3), np.uint8)    
    nf[:,:,0] = flood_img.copy()[:,:,0]
    nf[:,:,1] = flood_img.copy()[:,:,0]
    nf[:,:,2] = flood_img.copy()[:,:,0]

    cv2.drawContours(nf,[box],0,(0,255,0),8)

    t_cx = r_x-(rhalf)
    t_cy = r_y-(rhalf)

    xc,yc,zc = np.shape(c)        
    nc = np.zeros([xc,yc,zc])

    nc[:,0,0] = c[:,0,0]- t_cx
    nc[:,0,1] = c[:,0,1]- t_cy
    nc = nc.astype(int)
    
    
    
    cv2.drawContours(nc_img, nc, -1, 255, 1)
    maskfill = np.zeros((square_r+2, square_r+2), np.uint8)    
    cv2.floodFill(nc_img, maskfill, (rhalf,rhalf), 255)  
    rotate_M = cv2.getRotationMatrix2D((rhalf,rhalf),angle,1)            #rotate Matrix
    rotated = cv2.warpAffine(nc_img,rotate_M,(square_r,square_r))        #rotated image (1 channel)

#    rotated = np.zeros([square_r,square_r,3])                           #rotated image (3 channel)
#    rotated[:,:,0] = r_rotated
#    rotated[:,:,1] = r_rotated
#    rotated[:,:,2] = r_rotated
    
    box = cv2.boxPoints(((rhalf,rhalf),(MA,ma),0))
    box = np.int0(box)
#    cv2.drawContours(rotated,[box],0,(0,255,0),2)
    
    square_x1 = box[1][1]
    square_x2 = box[0][1]
    square_y1 = box[0][0]
    square_y2 = box[2][0]
    
    ROI = rotated[box[1][1]:box[0][1], box[0][0]:box[2][0]]

    [roiy,roix] = np.shape(ROI)

    roix_half = roix/2
    roiy_half = roiy/2
    
    roi_top = ROI[0:math.floor(roiy_half),0:roix]    
    roi_bot = ROI[math.floor(roiy_half+1):roiy,0:roix]    
    roi_left = ROI[0:roiy,0:math.floor(roix_half)]    
    roi_right = ROI[0:roiy,math.floor(roix_half+1):roix]    
    roi_right_flip = cv2.flip(roi_right, 1)
    roi_bot_flip = cv2.flip(roi_bot,0)
    
    overlap_y_1 = roi_top - roi_bot_flip
    overlap_y_2= roi_bot_flip - roi_top
    overlap_y =  cv2.bitwise_and(overlap_y_1, overlap_y_2)
    ret,overlap_y = cv2.threshold(overlap_y,0,255,cv2.THRESH_BINARY)

    overlap_x = roi_left - roi_right_flip 

    overlap_x_1 = roi_left - roi_right_flip
    overlap_x_2= roi_right_flip - roi_left
    overlap_x =  cv2.bitwise_and(overlap_x_1, overlap_x_2)
    ret,overlap_x = cv2.threshold(overlap_x,0,255,cv2.THRESH_BINARY)
    

    
    
    
    
    
    
    
    
#    square_r = np.maximum(imx,imy)
#    ell = np.zeros((square_r,square_r,1), np.uint8)
#    
#    xc,yc,zc = np.shape(c)
#    nc = np.zeros([xc,yc,zc])
#
#    t_cx = cx-(square_r/2)
#    t_cy = cy-(square_r/2)
#    
#    nc[:,0,0] = c[:,0,0]- t_cx
#    nc[:,0,1] = c[:,0,1]- t_cy
#    nc = nc.astype(int)
#
#    
#    cv2.drawContours(ell, nc, -1, 255, 1)
#    maskfill = np.zeros((square_r+2, square_r+2), np.uint8)    
#    cv2.floodFill(ell, maskfill, ((square_r/2).astype(int),(square_r/2).astype(int)), 85)
#
#
#
#    rhalf = (square_r/2).astype(int)
#
#    (el_x,el_y),(MA,ma),angle = cv2.minAreaRect(nc)
##    ell = np.round(flood_img.copy()/3)
#    ell = ell.astype(np.uint8)
#    box = cv2.boxPoints(((el_x,el_y),(MA,ma),angle))
#    box = np.int0(box)
#    cv2.drawContours(ell,[box],0,(255),2)
#
#    rotate_M = cv2.getRotationMatrix2D((rhalf,rhalf),angle,1)
#    r_rotated = cv2.warpAffine(ell,rotate_M,(square_r,square_r))
#    im2,r_contours,hierarchy = cv2.findContours(r_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#    rc = max(r_contours, key = cv2.contourArea)
#    rM = cv2.moments(rc)
#    rcx = int(rM["m10"] / rM["m00"])
#    rcy = int(rM["m01"] / rM["m00"])
#    r_imy,r_imx = np.shape(r_rotated)
#
#    rotated = np.zeros([square_r,square_r,3])
#    rotated[:,:,0] = r_rotated
#    rotated[:,:,1] = r_rotated
#    rotated[:,:,2] = r_rotated
#
#
#    center = np.array([el_x,el_y])
#
#    pMA1 = int(rhalf-ma/2)
#    pMA2 = int(rhalf+ma/2)
#    pma1 = int(rhalf-MA/2)
#    pma2 = int(rhalf+MA/2)
#    
#    cv2.line(rotated, (int(square_r/2),pMA1), (int(square_r/2),pMA2), (0,255,0), 2)     
#    cv2.line(rotated, (int(square_r/2),rcy), (int(square_r/2),rcy), (255,255,0), 2)     
#    
#    cv2.circle(rotated, (int(el_x),int(el_y)), 4, [0,0,255],3)
    
    
    
#-------------------------------------------------------------- ASSYMETRY END














#-------------------------------------------------------------- BORDER START

#-------------------------------------------------------------- BORDER FD START
    assym = np.zeros((height,width,3), np.uint8)
    assym[:,:,0] =(flood_img[:,:,0]/5)

    n_y_maj = (height/maj_axis)
    n_x_maj = (width/maj_axis)
    n_y_mnr = height/mnr_axis
    n_x_mnr = width/mnr_axis
    
    n_x_maj = int(n_x_maj)
    n_y_maj = int(n_y_maj)
    n_x_mnr = int(n_x_mnr)
    n_y_mnr = int(n_y_mnr)
    

#----------------------------------MAJOR AXIS
    for k in range (1,n_y_maj):
        for l in range (1,n_x_maj):
            flag = False
            mat_xa = ((l-1)*maj_axis)
            mat_xb = (l*maj_axis)
            mat_ya = ((k-1)*maj_axis)
            mat_yb = (k*maj_axis)
            
            mat = bimg[mat_ya:mat_yb , mat_xa:mat_xb] 
            value = np.sum(mat)
            value = int(value/255)
            if value > 0 :
                flag = True
            if flag == True:
                assym[mat_ya:mat_yb , mat_xa:mat_xb,1] =50 
                

#----------------------------------MINOR AXIS    
    for k in range (1,n_y_mnr):
        for l in range (1,n_x_mnr):
            flag = False
            mat_xa = ((l-1)*mnr_axis)
            mat_xb = (l*mnr_axis)
            mat_ya = ((k-1)*mnr_axis)
            mat_yb = (k*mnr_axis)
            
            mat = bimg[mat_ya:mat_yb , mat_xa:mat_xb] 
            value = np.sum(mat)
            value = int(value/255)
            if value > 0 :
                flag = True
            if flag == True:
                assym[mat_ya:mat_yb , mat_xa:mat_xb,2] =50 


    A_maj_mat = assym.copy()
    A_mnr_mat = assym.copy()

    A_maj_thresh_a = np.array([0,48,0])
    A_maj_thresh_b = np.array([255,52,0])
    A_mnr_thresh_a = np.array([0,0,50])
    A_mnr_thresh_b = np.array([255,0,50])

    A_maj_mat = cv2.inRange(A_maj_mat, A_maj_thresh_a, A_maj_thresh_b)
    A_mnr_mat = cv2.inRange(A_mnr_mat, A_mnr_thresh_a, A_mnr_thresh_b)

    A_maj = np.sum(A_maj_mat)/255
    A_mnr = np.sum(A_mnr_mat)/255
    A_leasion = np.sum(flood_img)/255
    
    Assym_val = (A_maj+A_mnr)/(2*A_leasion)
    print('Area major Value : ' + str(A_maj)) 
    print('Area minor Value : ' + str(A_mnr)) 
    print('Area of leasion Value : ' + str(A_leasion)) 
    print('Assymetry Value : ' + str(Assym_val)) 

#-------------------------------------------------------------- BORDER FD END



#-------------------------------------------------------------- BORDER STD START


    c_size = len(c)

    rad = [None]*(c_size-1)
    
    for k in range (0,c_size-1) :

        c_border_x = c[k][0][0]
        c_border_y = c[k][0][1]
                
        radx = c_border_x - cx
        rady = c_border_y - cy
        
        rad[k] = math.sqrt((radx*radx)+(rady*rady))
        
    std_border = np.std(rad)
    print('standar deviasi border : ' + str(std_border))
    b_ireg = std_border/np.average(rad)
    print('border ireggularity : ' + str(b_ireg))

#-------------------------------------------------------------- BORDER STD END
#-------------------------------------------------------------- BORDER END








#-------------------------------------------------------------- COLOR START
    color = ('b','g','r')
    hist=np.zeros([256,3])
    color_std = np.zeros(3)
    color_var = np.zeros([256,256,256])
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]
    
    for i in range (imy):
        for j in range (imx):
            b = blue[i,j]
            g = green[i,j]
            r = red[i,j]
            
            color_var[b,g,r] = color_var[b,g,r] + 1 
    
    color_var_bool = color_var>0
    color_variance = np.sum(color_var_bool)
    print('color variance : ' + str(color_variance))
    
    
    for i,col in enumerate(color):
        hist_n = cv2.calcHist([img],[i],flood_img,[256],[0,256])
        hist[:,i]=hist_n[:,0]
        plt.plot(hist[:,i],color = col)
        color_std[i] = np.std(hist[:,i])
        print(str(color[i]) + 'standard deviation : ' + str(color_std[i]))
        plt.xlim([0,256])
    plt.show()
#-------------------------------------------------------------- COLOR END






#-------------------------------------------------------------- DIAMETER START
    Diameter = 2*(np.max(rad))
    print('Diamater : ' + str(Diameter)) 
#-------------------------------------------------------------- DIAMETER END










# SHOWING IMAGE ALGORITHMS ---------------------------------------------------------------------- START
#Resize settings
resize = False
rheight = 600
rwidth = 800

#Show Image List
show_imagename = ['img','nmask',  'Contour Image', 'Blank Image', 'floodfill','segment','Assym','Major Mat','Minor Mat','ROI','overlap y','overlap x','rotated']
show_image = [img,nmask, contimg, bimg, nf,segment,assym,A_maj_mat,A_mnr_mat,ROI,overlap_y,overlap_x,rotated]
n_showimg = len(show_image)
#cv2.imwrite('closeimage.jpg',bimg)

#Image Showing Sequencing
for k in range (0,n_showimg):
    if resize == True:
        cv2.namedWindow(show_imagename[k],cv2.WINDOW_NORMAL)    
    cv2.imshow(show_imagename[k],show_image[k])
    if resize == True:
        cv2.resizeWindow(show_imagename[k],rwidth,rheight)    
# SHOWING IMAGE ALGORITHMS ---------------------------------------------------------------------- END





k = cv2.waitKey(0)
cv2.destroyAllWindows()

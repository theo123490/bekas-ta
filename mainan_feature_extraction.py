import cv2
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
from img_listing import image_list
from skimage.feature import greycomatrix, greycoprops
from hist_3d import hist_3d
from scipy.misc import imresize




counter=0

filename = 'rejected_image'
img_list = image_list(filename)
column = ['Diagnosis',
          'Assymetry Index', 
          'Fractal Dimension',
          'Contrast',
          'Dissimilarity',
          'Homogeneity',
          'ASM',
          'Energy',
          'Correlation',
          'GLCM_mat',
          'Normalize_Hist']
df = pd.DataFrame(columns=column)
isic_data = pd.read_csv('combine_img\metadata.csv')
isic_data.columns = isic_data.columns.str.replace('.','_')



for i in img_list :
    #ABCD SETTINGS ------------------------------ STARTS
    #CHOOSE IMAGE
    
    print('IMAGE : ' + i)
    name = i.split('.jpg')[0]

    img = cv2.imread(filename + '/' + i)
    imy,imx,imz = np.shape(img)
    shortening = 5
    img = img[shortening:imy-shortening,shortening:imx-shortening,:]
    imy,imx,imz = np.shape(img)
    gs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    
    #ABCD SETTINGS ------------------------------ ENDS
    
    ret1,th1 = cv2.threshold(gs,25,255,cv2.THRESH_BINARY)
     
    ckernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    closing = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, ckernel,iterations = 4)
    
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erosion = cv2.erode(closing,kernel,iterations = 50)
    
    derm_scope = erosion
     
    ret,th = cv2.threshold(gs,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
     
    #REMOVING HAIR ------------- START
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,((int(min([imx,imy])/400))+4,int(min([imx,imy])/400)+4))
    erosion = cv2.erode(th,kernel,iterations = int(min([imx,imy])/150))
    dilation = cv2.dilate(erosion,kernel,iterations = int(min([imx,imy])/150))
    #REMOVING HAIR ------------- END
    
    segment_mask = dilation
    
     
    nmask =  cv2.bitwise_and(derm_scope, segment_mask)
     
    im2,contours,hierarchy = cv2.findContours(nmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
     
    contimg = img.copy()
     
    height, width, channels = img.shape
    all_cnt_bimg = np.zeros((height,width,1), np.uint8)
    cv2.drawContours(all_cnt_bimg, contours, -1, 255, 1)
 
    if len(contours) != 0:
        
        diagnosis = isic_data[isic_data.name == i.split('.')[0]]['meta_clinical_diagnosis']
        diagnosis_n = []
        if diagnosis.iloc[0] == 'melanoma':
            diagnosis_n = 1
        elif diagnosis.iloc[0] == 'nevus':
            diagnosis_n = 0
            
        
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
        
    
        
        
#    #-------------------------------------------------------------- ASSYMETRY START
#    
#        rhalf = (square_r/2).astype(int)
#    
#        nc_img = np.zeros((square_r,square_r,1), np.uint8)
#        
#        (r_x,r_y),(MA,ma),angle = cv2.minAreaRect(c)
#    
#        t_cx = r_x-(rhalf)
#        t_cy = r_y-(rhalf)
#    
#        xc,yc,zc = np.shape(c)        
#        nc = np.zeros([xc,yc,zc])
#    
#        nc[:,0,0] = c[:,0,0]- t_cx
#        nc[:,0,1] = c[:,0,1]- t_cy
#        nc = nc.astype(int)
#        
#        cv2.drawContours(nc_img, nc, -1, 255, 1)
#        maskfill = np.zeros((square_r+2, square_r+2), np.uint8)    
#        cv2.floodFill(nc_img, maskfill, (rhalf,rhalf), 255)  
#        rotate_M = cv2.getRotationMatrix2D((rhalf,rhalf),angle,1)            #rotate Matrix
#        rotated = cv2.warpAffine(nc_img,rotate_M,(square_r,square_r))        #rotated image (1 channel)
#        
#        box = cv2.boxPoints(((rhalf,rhalf),(MA,ma),0))
#        box = np.int0(box)
#        
#        square_x1 = box[1][1]
#        square_x2 = box[0][1]
#        square_y1 = box[0][0]
#        square_y2 = box[2][0]
#        
#        ROI = rotated[box[1][1]:box[0][1], box[0][0]:box[2][0]]
#    
#        [roiy,roix] = np.shape(ROI)
#    
#        roix_half = roix/2
#        roiy_half = roiy/2
#        
#        roi_top = ROI[0:(math.floor(roiy_half)),0:roix]    
#        roi_bot = ROI[math.ceil(roiy_half):roiy,0:roix]    
#        roi_left = ROI[0:roiy,0:math.floor(roix_half)]    
#        roi_right = ROI[0:roiy,math.ceil(roix_half):roix]    
#        roi_right_flip = cv2.flip(roi_right, 1)
#        roi_bot_flip = cv2.flip(roi_bot,0)
#        
#        overlap_y_1 = roi_top - roi_bot_flip
#        overlap_y_2= roi_bot_flip - roi_top
#        overlap_y =  cv2.bitwise_and(overlap_y_1, overlap_y_2)
#        ret,overlap_y = cv2.threshold(overlap_y,0,255,cv2.THRESH_BINARY)
#    
#        overlap_x = roi_left - roi_right_flip 
#    
#        overlap_x_1 = roi_left - roi_right_flip
#        overlap_x_2= roi_right_flip - roi_left
#        overlap_x =  cv2.bitwise_and(overlap_x_1, overlap_x_2)
#        ret,overlap_x = cv2.threshold(overlap_x,0,255,cv2.THRESH_BINARY)
#        
#        sum_overlap_x = sum(sum(np.int64(overlap_x)))/255
#        sum_overlap_y = sum(sum(np.int64(overlap_y)))/255
#        sum_roi  = sum(sum(np.int64(ROI)))/255
#        
#        AI = sum([sum_overlap_x, sum_overlap_y])/(2*sum_roi)
#        print('Assymetry Index : ' + str(AI))
#        print('overlap_x : ' + str(sum_overlap_x))
#        print('overlap y : ' + str(sum_overlap_y))
#        print('ROI : ' + str(sum_roi))
#        
#    #-------------------------------------------------------------- ASSYMETRY END
#    
#    
#    #-------------------------------------------------------------- BORDER START
#    
#    
#    #-------------------------------------------------------------- BORDER FD START
#    
#        cv2.drawContours(nc_img, nc, -1, 255, 1)
#        assym = np.zeros((square_r,square_r,3), np.uint8)
#        assym_cont = assym.copy()
#        cv2.drawContours(assym_cont, nc, -1, 255, 1)
#        assym =assym_cont.copy()
#        
#        
#        divider = 5
#
#        FD_points = 3
#
#        FD_counter = np.zeros(FD_points)
#        FD_r = np.zeros([FD_points])
#            
#        
#        
#        for n in range (1,FD_points+1):        
#            FD_r[n-1] = int(square_r/(divider*n))
#            for k in range (0,(divider*n)):
#                for l in range (0,(divider*n)):
#                    mat_xa = int(((l-1)*FD_r[n-1]))
#                    mat_xb = int((l*FD_r[n-1]))
#                    mat_ya = int(((k-1)*FD_r[n-1]))
#                    mat_yb = int((k*FD_r[n-1]))
#                    mat = assym_cont[mat_ya:mat_yb , mat_xa:mat_xb,0] 
#                    value = np.sum(mat)
#                    value = int(value/255)
#                    if value > 0 :
#                        assym[mat_ya:mat_yb , mat_xa:mat_xb,1] =(50*n)
#                        cv2.rectangle(assym,(mat_xa,mat_ya),(mat_xb,mat_yb),[50*n,0,0],3)
#                        FD_counter [n-1] = FD_counter[n-1] + 1
#        
#        FD_log_n = np.log10(1/FD_counter)
#        FD_log_r = np.log10(FD_r)    
#
#        fit = np.polyfit(FD_log_n,FD_log_r,1)
#        fit_fn = np.poly1d(fit)
#        plt.plot(FD_log_n,FD_log_r, 'yo', FD_log_n, fit_fn(FD_log_n), '--k')
##        plt.pause(0.05)
#
#        fd = fit_fn[1]
#        print('fractal dimension : ' + str(fd))
#        
#    #-------------------------------------------------------------- BORDER FD END
#        
#        
#    #-------------------------------------------------------------- BORDER END
#    
#    
#    
#    #-------------------------------------------------------------- TEXTURE START
#    x,y,w,h = cv2.boundingRect(c)
#    # draw the book contour (in green)
#
#    col_roi = gs[ y:y+h, x:x+w]
#    glcm = greycomatrix(gs, [1], [0],levels = 256)
#    glcm_mat = glcm[:,:,0,0]
#    glcm_mat = glcm_mat.astype('uint8')
#    glcm_mat_norm = glcm_mat/np.sum(glcm_mat)
#    #GLCM HEAT MAPPING
##    fig, ax = plt.subplots()
##    im = ax.imshow(glcm_mat)
##    plt.pause(0.05)
##    plt.show()
#    
#    
#    GLCM_resize = imresize(glcm_mat, (85,85))
#
#    fig2, ax2 = plt.subplots()
#    im = ax2.imshow(GLCM_resize)
#    plt.pause(0.05)
#    plt.show()
#
#    
#    #GLCM PROPERTIES EXTRACTION
#    contrast = greycoprops(glcm, 'contrast')[0,0]
#    dissimilarity = greycoprops(glcm, 'dissimilarity')[0,0]
#    homogeneity = greycoprops(glcm, 'homogeneity')[0,0]
#    ASM = greycoprops(glcm, 'ASM')[0,0]
#    energy = greycoprops(glcm, 'energy')[0,0]
#    correlation = greycoprops(glcm, 'correlation')[0,0]
#
#
#    print('GLCM Contrast : ' + str(contrast))
#    print('GLCM Dissimiliartiy : ' + str(dissimilarity))
#    print('GLCM homogeneity : ' + str(homogeneity))
#    print('GLCM ASM : ' + str(ASM))
#    print('GLCM energy : ' + str(energy))
#    print('GLCM correlation : ' + str(correlation))
#
#        
#    #-------------------------------------------------------------- TEXTURE END
#
#
#    #-------------------------------------------------------------- COLOR START
#
#    # INVERSE SEGMENTATION OF SKIN
#    inv_flood_img = cv2.bitwise_not(flood_img)
#    inv_segment = cv2.bitwise_and(img, img, mask=inv_flood_img)
#
#    # SPLITTING RGB CHANNEL
#    s_R = inv_segment[:,:,2]
#    s_G = inv_segment[:,:,1]
#    s_B = inv_segment[:,:,0]
#
#    # FIND SKIN AVERAGE VALUE
#    ave_s_R = np.sum(s_R)/(np.sum(inv_flood_img)/255)
#    ave_s_G = np.sum(s_G)/(np.sum(inv_flood_img)/255)
#    ave_s_B = np.sum(s_B)/(np.sum(inv_flood_img)/255)
#
#    print('red skin average val :' + str(ave_s_R))
#    print('Green skin average val :' + str(ave_s_G))
#    print('Blue skin average val :' + str(ave_s_B))
#    
#    # CONVERT RGB DATA TYPE
#    R_mat = segment[:,:,2].astype('int32')
#    G_mat = segment[:,:,1].astype('int32')
#    B_mat = segment[:,:,0].astype('int32')
#
#    R_new_val = R_mat - ave_s_R
#    G_new_val = G_mat - ave_s_G
#    B_new_val = B_mat - ave_s_B
#
#    col_segment = np.zeros([imy,imx,3])
#    col_segment[:,:,2] = R_new_val
#    col_segment[:,:,1] = G_new_val
#    col_segment[:,:,0] = B_new_val
#    
#    col_segment_point = col_segment.reshape([(imx*imy),3])
#    
#    points = col_segment_point
#    
#    hist,binedges = hist_3d(col_segment_point,4)
#
#    bg_px = np.sum(inv_flood_img)/255
#    hist[0,0,0]=hist[0,0,0] - bg_px
#    norm_hist = hist/np.sum(hist)
#
#    #-------------------------------------------------------------- COLOR END
    cv2.imwrite('rejected_contimage/contour' + str(i), contimg)


#    df2 = pd.DataFrame(data = {'Diagnosis' : [diagnosis_n],
#                               'Assymetry Index' : [AI] , 
#                               'Fractal Dimension' : [fd], 
#                               'Contrast' : [contrast],
#                               'Dissimilarity':[dissimilarity],
#                               'Homogeneity': [homogeneity],
#                               'ASM':[ASM],
#                               'Energy' : [energy],
#                               'Correlation':[correlation],
#                               'GLCM_mat' : [GLCM_resize],
#                               'Normalize_Hist': [norm_hist]}, index = [name])
#    df = df.append(df2)
    
    print('\n')
    print('\n')
    print('\n')
#    
#    counter=counter + 1
#    if counter >= 10:
#        df.to_csv('Feature Data.csv')
#        df.to_pickle('Feature Data.pk')    
#        counter = 0
#    counter = counter + 1
#    

#print(df)
#df.to_csv('Feature Data.csv')
#df.to_pickle('Feature Data.pk')    

import cv2
import numpy as np
import math
import pandas as pd
from img_listing import image_list
from skimage.feature import greycomatrix, greycoprops
from hist_3d import hist_3d
from scipy.misc import imresize




counter=0

filename = 'combine_img'
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
          'R_Contrast',
          'R_Dissimilarity',
          'R_Homogeneity',
          'R_ASM',
          'R_Energy',
          'R_Correlation',
          'G_Contrast',
          'G_Dissimilarity',
          'G_Homogeneity',
          'G_ASM',
          'G_Energy',
          'G_Correlation',
          'B_Contrast',
          'B_Dissimilarity',
          'B_Homogeneity',
          'B_ASM',
          'B_Energy',
          'B_Correlation',
          'Contrast_45',
          'Dissimilarity_45',
          'Homogeneity_45',
          'ASM_45',
          'Energy_45',
          'Correlation_45',
          'R_Contrast_45',
          'R_Dissimilarity_45',
          'R_Homogeneity_45',
          'R_ASM_45',
          'R_Energy_45',
          'R_Correlation_45',
          'G_Contrast_45',
          'G_Dissimilarity_45',
          'G_Homogeneity_45',
          'G_ASM_45',
          'G_Energy_45',
          'G_Correlation_45',
          'B_Contrast_45',
          'B_Dissimilarity_45',
          'B_Homogeneity_45',
          'B_ASM_45',
          'B_Energy_45',
          'B_Correlation_45',
          
          'Contrast_90',
          'Dissimilarity_90',
          'Homogeneity_90',
          'ASM_90',
          'Energy_90',
          'Correlation_90',
          'R_Contrast_90',
          'R_Dissimilarity_90',
          'R_Homogeneity_90',
          'R_ASM_90',
          'R_Energy_90',
          'R_Correlation_90',
          'G_Contrast_90',
          'G_Dissimilarity_90',
          'G_Homogeneity_90',
          'G_ASM_90',
          'G_Energy_90',
          'G_Correlation_90',
          'B_Contrast_90',
          'B_Dissimilarity_90',
          'B_Homogeneity_90',
          'B_ASM_90',
          'B_Energy_90',
          'B_Correlation_90',
          
          'Contrast_135',
          'Dissimilarity_135',
          'Homogeneity_135',
          'ASM_135',
          'Energy_135',
          'Correlation_135',
          'R_Contrast_135',
          'R_Dissimilarity_135',
          'R_Homogeneity_135',
          'R_ASM_135',
          'R_Energy_135',
          'R_Correlation_135',
          'G_Contrast_135',
          'G_Dissimilarity_135',
          'G_Homogeneity_135',
          'G_ASM_135',
          'G_Energy_135',
          'G_Correlation_135',
          'B_Contrast_135',
          'B_Dissimilarity_135',
          'B_Homogeneity_135',
          'B_ASM_135',
          'B_Energy_135',
          'B_Correlation_135',

          'Normalize_Hist'
          ]
df = pd.DataFrame(columns=column)
isic_data = pd.read_csv('combine_img' + '\metadata1.csv')
isic_data.columns = isic_data.columns.str.replace('.','_')



for i in img_list :
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
    
    
    
    ret1,th1 = cv2.threshold(gs,25,255,cv2.THRESH_BINARY)
    
    ckernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    closing = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, ckernel,iterations = 4)
    
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erosion = cv2.erode(closing,kernel,iterations = 50)
    
    derm_scope = erosion
     
     
     
    #REMOVING HAIR FILTERING------------- START
     
    #gs_filter = cv2.medianBlur(gs,21) #median blur
    #gs_filter = cv2.GaussianBlur(gs,(31,31),0) #Gaussian Blur
    #gs_filter = cv2.blur(gs,(31,31)) #Averaging
    #REMOVING HAIR FILTERING------------- END
     
     
     
    ret,th = cv2.threshold(gs,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
     
    #REMOVING HAIR MORPH TRANSFORM------------- START
    #Morphologic Transform ------------- START
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erosion = cv2.erode(th,kernel,iterations = int(min([imx,imy])/100))
    dilation = cv2.dilate(erosion,kernel,iterations = int(min([imx,imy])/100))
    #Morphologic Transform ------------- End
    #REMOVING HAIR MORPH TRANSFORM------------- END
     
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
        cv2.drawContours(contimg, c, -1, 255, 8)
    
        x,y,w,h = cv2.boundingRect(c)
        # draw the book contour (in green)
        cv2.rectangle(contimg,(x,y),(x+w,y+h),(0,255,0),8)
    
        #Finding center of mass
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        t_cx = cx-(imx/2)
        t_cy = cy-(imy/2)
        
    
        #CREATING BORDER 
        square_r = np.maximum(imx,imy)
        bimg = np.zeros((height,width,1), np.uint8)
        cv2.drawContours(bimg, c, -1, 255, 5)
        flood_img = bimg.copy()
        maskfill = np.zeros((height+2, width+2), np.uint8)    
        cv2.floodFill(flood_img, maskfill, (cx,cy), 255)
        segment = cv2.bitwise_and(img, img, mask=flood_img)
        
    
        
        
    #-------------------------------------------------------------- ASSYMETRY START
    
        rhalf = (square_r/2).astype(int)
    
        nc_img = np.zeros((square_r,square_r,1), np.uint8)
        
        (r_x,r_y),(MA,ma),angle = cv2.minAreaRect(c)
    
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
        
        box = cv2.boxPoints(((rhalf,rhalf),(MA,ma),0))
        box = np.int0(box)
        
        square_x1 = box[1][1]
        square_x2 = box[0][1]
        square_y1 = box[0][0]
        square_y2 = box[2][0]
        
        ROI = rotated[box[1][1]:box[0][1], box[0][0]:box[2][0]]
    
        [roiy,roix] = np.shape(ROI)
    
        roix_half = roix/2
        roiy_half = roiy/2
        
        roi_top = ROI[0:(math.floor(roiy_half)),0:roix]    
        roi_bot = ROI[math.ceil(roiy_half):roiy,0:roix]    
        roi_left = ROI[0:roiy,0:math.floor(roix_half)]    
        roi_right = ROI[0:roiy,math.ceil(roix_half):roix]    
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
        
        sum_overlap_x = sum(sum(np.int64(overlap_x)))/255
        sum_overlap_y = sum(sum(np.int64(overlap_y)))/255
        sum_roi  = sum(sum(np.int64(ROI)))/255
        
        AI = sum([sum_overlap_x, sum_overlap_y])/(2*sum_roi)
#        print('Assymetry Index : ' + str(AI))
#        print('overlap_x : ' + str(sum_overlap_x))
#        print('overlap y : ' + str(sum_overlap_y))
#        print('ROI : ' + str(sum_roi))
        
    #-------------------------------------------------------------- ASSYMETRY END
    
    
    #-------------------------------------------------------------- BORDER START
    
    
    #-------------------------------------------------------------- BORDER FD START
    
        cv2.drawContours(nc_img, nc, -1, 255, 1)
        assym = np.zeros((square_r,square_r,3), np.uint8)
        assym_cont = assym.copy()
        cv2.drawContours(assym_cont, nc, -1, 255, 1)
        assym =assym_cont.copy()
        
        
        divider = 5

        FD_points = 20

        FD_counter = np.zeros(FD_points)
        FD_r = np.zeros([FD_points])
            
        
       

        for n in range (1,FD_points+1):        
             FD_r[n-1] = int(square_r/(divider+(n*1.5)))
             for k in range (int(divider+(n*1.5))):
                  for l in range (0,int(divider+(n*1.5))):
                     mat_xa = int(((l-1)*FD_r[n-1]))
                     mat_xb = int((l*FD_r[n-1]))
                     mat_ya = int(((k-1)*FD_r[n-1]))
                     mat_yb = int((k*FD_r[n-1]))
                     mat = assym_cont[mat_ya:mat_yb , mat_xa:mat_xb,0] 
                     value = np.sum(mat)
                     value = int(value/255)
                     if value > 0 :
                         assym[mat_ya:mat_yb , mat_xa:mat_xb,1] =(50*n)
                         cv2.rectangle(assym,(mat_xa,mat_ya),(mat_xb,mat_yb),[50*n,0,0],3)
                         FD_counter [n-1] = FD_counter[n-1] + 1


        FD_log_n = np.log10(1/FD_counter)
        FD_log_r = np.log10(FD_r)    

        fit = np.polyfit(FD_log_n,FD_log_r,1)
        fit_fn = np.poly1d(fit)
#        plt.plot(FD_log_n,FD_log_r, 'yo', FD_log_n, fit_fn(FD_log_n), '--k')
#        plt.pause(0.05)

        fd = fit_fn[1]
#        print('fractal dimension : ' + str(fd))
        
    #-------------------------------------------------------------- BORDER FD END
        
        
    #-------------------------------------------------------------- BORDER END
    
    
    
#-------------------------------------------------------------- TEXTURE START
    glcm_img = img.copy()
    glcm_rotate_M = cv2.getRotationMatrix2D((r_x,r_y),angle,1)
    glcm_rotated = cv2.warpAffine(glcm_img,glcm_rotate_M,glcm_img.shape[1::-1])     
    
    glcm_mask_img = flood_img.copy()
    glcm_mask_rotated = cv2.warpAffine(glcm_mask_img,glcm_rotate_M,glcm_mask_img.shape[1::-1])     
    
    

    GLCM_segment = cv2.bitwise_and(glcm_rotated, glcm_rotated, mask=glcm_mask_rotated)
    glcm_gs = cv2.cvtColor(GLCM_segment,cv2.COLOR_BGR2GRAY)

    _,glcm_contours,_ = cv2.findContours(glcm_mask_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    glcm_c =max(glcm_contours, key = cv2.contourArea)

    x,y,w,h = cv2.boundingRect(glcm_c)
    # draw the book contour (in green)
    col_roi1 = glcm_gs[ y:y+h, x:x+w]
    col_roi = cv2.resize(col_roi1, (600,400), interpolation = cv2.INTER_AREA)
    glcm = greycomatrix(col_roi, [1], [0, np.pi/4, np.pi/2, np.pi*3/4],levels = 256)
    glcm_mat = glcm[:,:,0,0]
    glcm_mat = glcm_mat.astype('uint8')
    glcm_mat_norm = glcm_mat/np.sum(glcm_mat)
    
    
    GLCM_resize = imresize(glcm_mat, (85,85))

    
    R_col_roi1 = GLCM_segment[ y:y+h, x:x+w ,0]
    G_col_roi1 = GLCM_segment[ y:y+h, x:x+w ,1]
    B_col_roi1 = GLCM_segment[ y:y+h, x:x+w ,2]

    R_col_roi = cv2.resize(R_col_roi1, (600,400), interpolation = cv2.INTER_AREA)
    G_col_roi = cv2.resize(G_col_roi1, (600,400), interpolation = cv2.INTER_AREA)
    B_col_roi = cv2.resize(B_col_roi1, (600,400), interpolation = cv2.INTER_AREA)

    R_glcm = greycomatrix(R_col_roi, [1], [0, np.pi/4, np.pi/2, np.pi*3/4],levels = 256)
    G_glcm = greycomatrix(G_col_roi, [1], [0, np.pi/4, np.pi/2, np.pi*3/4],levels = 256)
    B_glcm = greycomatrix(B_col_roi, [1], [0, np.pi/4, np.pi/2, np.pi*3/4],levels = 256)


    #GLCM PROPERTIES EXTRACTION
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    ASM = greycoprops(glcm, 'ASM')
    energy = greycoprops(glcm, 'energy')
    correlation = greycoprops(glcm, 'correlation')

    R_contrast = greycoprops(R_glcm, 'contrast')
    R_dissimilarity = greycoprops(R_glcm, 'dissimilarity')
    R_homogeneity = greycoprops(R_glcm, 'homogeneity')
    R_ASM = greycoprops(R_glcm, 'ASM')
    R_energy = greycoprops(R_glcm, 'energy')
    R_correlation = greycoprops(R_glcm, 'correlation')

    G_contrast = greycoprops(G_glcm, 'contrast')
    G_dissimilarity = greycoprops(G_glcm, 'dissimilarity')
    G_homogeneity = greycoprops(G_glcm, 'homogeneity')
    G_ASM = greycoprops(G_glcm, 'ASM')
    G_energy = greycoprops(G_glcm, 'energy')
    G_correlation = greycoprops(G_glcm, 'correlation')
    
    B_contrast = greycoprops(B_glcm, 'contrast')
    B_dissimilarity = greycoprops(B_glcm, 'dissimilarity')
    B_homogeneity = greycoprops(B_glcm, 'homogeneity')
    B_ASM = greycoprops(B_glcm, 'ASM')
    B_energy = greycoprops(B_glcm, 'energy')
    B_correlation = greycoprops(B_glcm, 'correlation')


#    print('GLCM Contrast : ' + str(contrast))
#    print('GLCM Dissimiliartiy : ' + str(dissimilarity))
#    print('GLCM homogeneity : ' + str(homogeneity))
#    print('GLCM ASM : ' + str(ASM))
#    print('GLCM energy : ' + str(energy))
#    print('GLCM correlation : ' + str(correlation))

#-------------------------------------------------------------- TEXTURE END


    #-------------------------------------------------------------- COLOR START

    # INVERSE SEGMENTATION OF SKIN


    inv_flood_img = cv2.bitwise_not(flood_img)
    inv_segment = cv2.bitwise_and(img, img, mask=inv_flood_img)
#        
#    inv_segment1 = cv2.cvtColor(inv_segment, cv2.COLOR_BGR2HSV)
#    segment1 = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)

    inv_segment1 = inv_segment
    segment1 = segment
    
    # SPLITTING RGB CHANNEL
    s_R = inv_segment1[:,:,2]
    s_G = inv_segment1[:,:,1]
    s_B = inv_segment1[:,:,0]

    # FIND SKIN AVERAGE VALUE
    ave_s_R = np.sum(s_R)/(np.sum(inv_flood_img)/255)
    ave_s_G = np.sum(s_G)/(np.sum(inv_flood_img)/255)
    ave_s_B = np.sum(s_B)/(np.sum(inv_flood_img)/255)

#    print('red skin average val :' + str(ave_s_R))
#    print('Green skin average val :' + str(ave_s_G))
#    print('Blue skin average val :' + str(ave_s_B))
#    
    # CONVERT RGB DATA TYPE
    R_mat = segment1[:,:,2].astype('int32')
    G_mat = segment1[:,:,1].astype('int32')
    B_mat = segment1[:,:,0].astype('int32')

    R_new_val = R_mat - ave_s_R
    G_new_val = G_mat - ave_s_G
    B_new_val = B_mat - ave_s_B

#    R_new_val = R_mat
#    G_new_val = G_mat
#    B_new_val = B_mat


    col_segment = np.zeros([imy,imx,3])
    col_segment[:,:,2] = R_new_val
    col_segment[:,:,1] = G_new_val
    col_segment[:,:,0] = B_new_val
    
    col_segment_point = col_segment.reshape([(imx*imy),3])
    
    points = col_segment_point
    
    hist,binedges = hist_3d(col_segment_point,16)

    bg_px = np.sum(inv_flood_img)/255
    
    a = np.array([255-ave_s_B,255-ave_s_G,255-ave_s_R])
    b = np.uint(np.floor(a*16/(256+255)))
    
    hist[b[0],b[1],b[2]]=hist[b[0],b[1],b[2]] - bg_px
    norm_hist = hist/np.sum(hist)

    #-------------------------------------------------------------- COLOR END
    cv2.imwrite('contimage/contour' + str(i), contimg)


    df2 = pd.DataFrame(data = {'Diagnosis' : [diagnosis_n],
                               'Assymetry Index' : [AI] , 
                               'Fractal Dimension' : [fd], 

                               'Contrast'              : [contrast[0,0]],
                               'Dissimilarity'         :[dissimilarity[0,0]],
                               'Homogeneity'           : [homogeneity[0,0]],
                               'ASM'                   :[ASM[0,0]],
                               'Energy'                : [energy[0,0]],
                               'Correlation'           :[correlation[0,0]],
                               'R_Contrast'            : [R_contrast[0,0]],
                               'R_Dissimilarity'       :[R_dissimilarity[0,0]],
                               'R_Homogeneity'         : [R_homogeneity[0,0]],
                               'R_ASM'                 :[R_ASM[0,0]],
                               'R_Energy'              : [R_energy[0,0]],
                               'R_Correlation'         :[R_correlation[0,0]],
                               'G_Contrast'            : [G_contrast[0,0]],
                               'G_Dissimilarity'       :[G_dissimilarity[0,0]],
                               'G_Homogeneity'         : [G_homogeneity[0,0]],
                               'G_ASM'                 :[G_ASM[0,0]],
                               'G_Energy'              : [G_energy[0,0]],
                               'G_Correlation'         :[G_correlation[0,0]],
                               'B_Contrast'            : [B_contrast[0,0]],
                               'B_Dissimilarity'       :[B_dissimilarity[0,0]],
                               'B_Homogeneity'         : [B_homogeneity[0,0]],
                               'B_ASM'                 :[B_ASM[0,0]],
                               'B_Energy'              : [B_energy[0,0]],
                               'B_Correlation'         :[B_correlation[0,0]],

                               'Contrast_45'           : [contrast[0,1]],
                               'Dissimilarity_45'      :[dissimilarity[0,1]],
                               'Homogeneity_45'        : [homogeneity[0,1]],
                               'ASM_45'                :[ASM[0,1]],
                               'Energy_45'             : [energy[0,1]],
                               'Correlation_45'        :[correlation[0,1]],
                               'R_Contrast_45'         : [R_contrast[0,1]],
                               'R_Dissimilarity_45'    :[R_dissimilarity[0,1]],
                               'R_Homogeneity_45'      : [R_homogeneity[0,1]],
                               'R_ASM_45'              :[R_ASM[0,1]],
                               'R_Energy_45'           : [R_energy[0,1]],
                               'R_Correlation_45'      :[R_correlation[0,1]],
                               'G_Contrast_45'         : [G_contrast[0,1]],
                               'G_Dissimilarity_45'    :[G_dissimilarity[0,1]],
                               'G_Homogeneity_45'      : [G_homogeneity[0,1]],
                               'G_ASM_45'              :[G_ASM[0,1]],
                               'G_Energy_45'           : [G_energy[0,1]],
                               'G_Correlation_45'      :[G_correlation[0,1]],
                               'B_Contrast_45'         : [B_contrast[0,1]],
                               'B_Dissimilarity_45'    :[B_dissimilarity[0,1]],
                               'B_Homogeneity_45'      : [B_homogeneity[0,1]],
                               'B_ASM_45'              :[B_ASM[0,1]],
                               'B_Energy_45'           : [B_energy[0,1]],
                               'B_Correlation_45'      :[B_correlation[0,1]],

                               'Contrast_90'           : [contrast[0,2]],
                               'Dissimilarity_90'      :[dissimilarity[0,2]],
                               'Homogeneity_90'        : [homogeneity[0,2]],
                               'ASM_90'                :[ASM[0,2]],
                               'Energy_90'             : [energy[0,2]],
                               'Correlation_90'        :[correlation[0,2]],
                               'R_Contrast_90'         : [R_contrast[0,2]],
                               'R_Dissimilarity_90'    :[R_dissimilarity[0,2]],
                               'R_Homogeneity_90'      : [R_homogeneity[0,2]],
                               'R_ASM_90'              :[R_ASM[0,2]],
                               'R_Energy_90'           : [R_energy[0,2]],
                               'R_Correlation_90'      :[R_correlation[0,2]],
                               'G_Contrast_90'         : [G_contrast[0,2]],
                               'G_Dissimilarity_90'    :[G_dissimilarity[0,2]],
                               'G_Homogeneity_90'      : [G_homogeneity[0,2]],
                               'G_ASM_90'              :[G_ASM[0,2]],
                               'G_Energy_90'           : [G_energy[0,2]],
                               'G_Correlation_90'      :[G_correlation[0,2]],
                               'B_Contrast_90'         : [B_contrast[0,2]],
                               'B_Dissimilarity_90'    :[B_dissimilarity[0,2]],
                               'B_Homogeneity_90'      : [B_homogeneity[0,2]],
                               'B_ASM_90'              :[B_ASM[0,2]],
                               'B_Energy_90'           : [B_energy[0,2]],
                               'B_Correlation_90'      :[B_correlation[0,2]],

                               'Contrast_135'          : [contrast[0,3]],
                               'Dissimilarity_135'     :[dissimilarity[0,3]],
                               'Homogeneity_135'       : [homogeneity[0,3]],
                               'ASM_135'               :[ASM[0,3]],
                               'Energy_135'            : [energy[0,3]],
                               'Correlation_135'       :[correlation[0,3]],
                               'R_Contrast_135'        : [R_contrast[0,3]],
                               'R_Dissimilarity_135'   :[R_dissimilarity[0,3]],
                               'R_Homogeneity_135'     : [R_homogeneity[0,3]],
                               'R_ASM_135'             :[R_ASM[0,3]],
                               'R_Energy_135'          : [R_energy[0,3]],
                               'R_Correlation_135'     :[R_correlation[0,3]],
                               'G_Contrast_135'        : [G_contrast[0,3]],
                               'G_Dissimilarity_135'   :[G_dissimilarity[0,3]],
                               'G_Homogeneity_135'     : [G_homogeneity[0,3]],
                               'G_ASM_135'             :[G_ASM[0,3]],
                               'G_Energy_135'          : [G_energy[0,3]],
                               'G_Correlation_135'     :[G_correlation[0,3]],
                               'B_Contrast_135'        : [B_contrast[0,3]],
                               'B_Dissimilarity_135'   :[B_dissimilarity[0,3]],
                               'B_Homogeneity_135'     : [B_homogeneity[0,3]],
                               'B_ASM_135'             :[B_ASM[0,3]],
                               'B_Energy_135'          : [B_energy[0,3]],
                               'B_Correlation_135'     :[B_correlation[0,3]],

                               'Normalize_Hist': [hist]}, index = [name])
    print(str(df2))
    df = df.append(df2)
    
    print('\n')
    print('\n')
    print('\n')
    
    counter=counter + 1
    if counter >= 500:
        df.to_csv('Feature Data.csv')
        df.to_pickle('Feature Data.pk')    
        counter = 0
    counter = counter + 1
    

print(df)
df.to_csv('Feature Data.csv')
df.to_pickle('Feature Data.pk')



print('DON\'T FORGET TO MANUAL CHECK!!!!!!!!!!!')    

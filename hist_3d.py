import numpy as np


def hist_3d(points,bin_n):
    min_hist = -255
    max_hist = 255
    array_edges = np.array([
            [max_hist,max_hist,max_hist],
            [max_hist,min_hist,max_hist],
            [max_hist,max_hist,min_hist],
            [max_hist,min_hist,max_hist],
            [min_hist,max_hist,max_hist],
            [min_hist,max_hist,min_hist],
            [min_hist,min_hist,max_hist],
            [min_hist,min_hist,min_hist],
            ])
    
#    points = np.random.randn(1000,3)*255
#    points[points>100] = 100
#    points[points<-100] = -100
    
    points = np.append(points,array_edges,axis = 0)
    
    hist, binedges = np.histogramdd(points, normed=False, bins = bin_n)
    
    
    bin_n_min = bin_n - 1
    hist[0,0,0] = hist[0,0,0] - 1
    hist[0,0,bin_n_min] = hist[0,0,bin_n_min] - 1
    hist[0,bin_n_min,0] = hist[0,bin_n_min,0] - 1
    hist[0,bin_n_min,bin_n_min] = hist[0,bin_n_min,bin_n_min] - 1
    hist[bin_n_min,0,0] = hist[bin_n_min,0,0] - 1
    hist[bin_n_min,0,bin_n_min] = hist[bin_n_min,0,bin_n_min] - 1
    hist[bin_n_min,bin_n_min,0] = hist[bin_n_min,bin_n_min,0] - 1
    hist[bin_n_min,bin_n_min,bin_n_min] = hist[bin_n_min,bin_n_min,bin_n_min] - 1
    
    return hist, binedges


def hist_3d2(points,bin_n):
    min_hist = -0
    max_hist = 255
    array_edges = np.array([
            [max_hist,max_hist,max_hist],
            [max_hist,min_hist,max_hist],
            [max_hist,max_hist,min_hist],
            [max_hist,min_hist,max_hist],
            [min_hist,max_hist,max_hist],
            [min_hist,max_hist,min_hist],
            [min_hist,min_hist,max_hist],
            [min_hist,min_hist,min_hist],
            ])
    
#    points = np.random.randn(1000,3)*255
#    points[points>100] = 100
#    points[points<-100] = -100
    
    points = np.append(points,array_edges,axis = 0)
    
    hist, binedges = np.histogramdd(points, normed=False, bins = bin_n)
    
    
    bin_n_min = bin_n - 1
    hist[0,0,0] = hist[0,0,0] - 1
    hist[0,0,bin_n_min] = hist[0,0,bin_n_min] - 1
    hist[0,bin_n_min,0] = hist[0,bin_n_min,0] - 1
    hist[0,bin_n_min,bin_n_min] = hist[0,bin_n_min,bin_n_min] - 1
    hist[bin_n_min,0,0] = hist[bin_n_min,0,0] - 1
    hist[bin_n_min,0,bin_n_min] = hist[bin_n_min,0,bin_n_min] - 1
    hist[bin_n_min,bin_n_min,0] = hist[bin_n_min,bin_n_min,0] - 1
    hist[bin_n_min,bin_n_min,bin_n_min] = hist[bin_n_min,bin_n_min,bin_n_min] - 1
    
    return hist, binedges


from bresenham import bresenham
import numpy as np
import random
import scipy.ndimage as nd


def mydrawPNG(vector_image):
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    final_list = []
    point_set = 0 
    for i in range( 0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])
                point_set += 1

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        
        final_list.extend([list(j) for j in cordList])
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])
    return final_list

def preprocess(sketch_points, side=800):
    sketch_points = sketch_points.astype(np.float)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([side, side])
    sketch_points[:, :2] = sketch_points[:, :2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points

def rasterize_Sketch(sketch_points):
    p1 = preprocess(sketch_points , 800)
    p1 = mydrawPNG(p1)
    random.shuffle(p1)
    intermediate_point_list = p1[:500]
    raster_image = np.zeros((int(800), int(800)), dtype=np.float32)
    for coordinate in intermediate_point_list:
        if (coordinate[0] > 0 and coordinate[1] > 0) and (coordinate[0] < 800 and coordinate[1] < 800):
                raster_image[coordinate[1], coordinate[0]] = 255.0
    raster_image = nd.binary_dilation(raster_image) * 255.0
    
    return raster_image

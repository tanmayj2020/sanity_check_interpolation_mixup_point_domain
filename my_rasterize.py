import numpy as np
import scipy.ndimage as nd



def preprocess(sketch_points, side=800):
    sketch_points = sketch_points.astype(np.float)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([side, side])
    sketch_points[:, :2] = sketch_points[:, :2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points

def rasterize_Sketch(sketch_points):
    sketch_points = np.array(sketch_points)
    p1 = preprocess(sketch_points , 800)
    raster_image = np.zeros((int(800), int(800)), dtype=np.float32)
    for coordinate in p1:
        if (coordinate[0] > 0 and coordinate[1] > 0) and (coordinate[0] < 800 and coordinate[1] < 800):
                raster_image[int(coordinate[1]), int(coordinate[0])] = 255.0
    raster_image = nd.binary_dilation(raster_image) * 255.0
    
    return raster_image

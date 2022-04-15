import math
from ortools.linear_solver import pywraplp
import numpy as np
import random
from bresenham import bresenham


def euclidean_distance(x , y):
    return math.sqrt(sum((a-b)**2 for (a ,b) in zip(x ,y)))

def data_matrix(p1 , p2):
    data_ = []
    for x in p1:
        x_list = []
        for y in p2:
            x_list.append(euclidean_distance(x , y))
        data_.append(x_list)
    return data_



def getAssignment(p1 , p2):
    assert(len(p1) == len(p2)) , "p1 and p2 must have same number of points"
    num_points = len(p1)
    data_ = data_matrix(p1 , p2)
    solver= pywraplp.Solver.CreateSolver("SCIP")

    # Creatinng the optimization variables
    x = {}
    for i in range(num_points):
        for j in range(num_points):
            x[i , j] = solver.IntVar(0 , 1 , '')
    
    # Adding the bijective Constraints
    for i in range(num_points):
        solver.Add(solver.Sum([x[i,j] for j in range(num_points)]) == 1)
    
    for j in range(num_points):
        solver.Add(solver.Sum([x[i,j] for i in range(num_points)]) == 1)
    
    # Objective function 
    objective_terms =[]
    for i in range(num_points):
        for j in range(num_points):
            objective_terms.append(data_[i][j] * x[i , j])
    solver.Minimize(solver.Sum(objective_terms))

    status = solver.Solve()
    ans_dict = {}
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print("Bijection success")
        print("EMD (minimum work) = " , solver.Objective().Value()/num_points , "\n")
        for i in range(num_points):
            for j in range(num_points):
                if x[i , j].solution_value() > 0.5:
                    ans_dict[i] = j
        return ans_dict
    else:
        print('No solution found.')



def interpolation(p1 , p2 , lmbda , ans_dict):
    assert 0 <= lmbda <= 1 , "lambda out of bounds"
    n = len(p1)
    if lmbda == 0.0:
        return p2
    if lmbda == 1.0:
        return p1
    intermediate_point_list = []
    for i in range(n):
        j = ans_dict[i]
        point1 = p1[i]
        point2 = p2[j]
        point1 = np.array(point1)
        point2 = np.array(point2)

        intermediate_point = (lmbda) * point1 + (1-lmbda) * point2
        intermediate_point_list.append(intermediate_point.tolist())
    intermediate_point_list = np.array(intermediate_point_list)
    return intermediate_point_list




def mydrawPNG(vector_image):
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    final_list = []
    for i in range( 0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

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



def strategy3(p1 , p2):
    p1 = preprocess(p1 , 800)
    p2 = preprocess(p2 , 800)
    p1 = mydrawPNG(p1)
    p2 = mydrawPNG(p2)

    random.shuffle(p1)
    point_1_list = p1[:500]

    random.shuffle(p2)
    point_2_list = p2[:500]

    return point_1_list , point_2_list


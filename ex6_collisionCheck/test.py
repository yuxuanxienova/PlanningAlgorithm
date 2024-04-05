
from pdm4ar.exercises_def.ex06.structures import *

import triangle
from triangle import convex_hull
import numpy as np
import numpy as np
from dataclasses import dataclass, replace

from typing import List, Tuple, Any, Sequence

import numpy as np
from geometry import SE2value



from pdm4ar.exercises_def.ex06.structures import (
    Polygon,
    GeoPrimitive,
    Point,
    Segment,
    Circle,
    Triangle,
    Path,
)
import matplotlib.pyplot as plt


def visualize_large_grid(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a heatmap of the grid
    cax = ax.matshow(grid, cmap='viridis')

    # Add colorbar
    plt.colorbar(cax)

    # Set axis labels
    ax.set_xticks(range(0,cols,20))
    ax.set_yticks(range(0,rows,20))
    ax.set_xticklabels(range(0, cols ,20 ))
    ax.set_yticklabels(range(0, rows ,20 ))
    
    # Add horizontal and vertical lines
    for i in range(0,rows,20):
        ax.axhline(y=i + 10, color='white', linewidth=1)
    for j in range(0,cols,20):
        ax.axvline(x=j + 10, color='white', linewidth=1)

    # Show the grid
    plt.show()
def visualize_grid(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a heatmap of the grid
    cax = ax.matshow(grid, cmap='viridis')

    # Add colorbar
    plt.colorbar(cax)

    # Set axis labels
    ax.set_xticks(range(0,cols))
    ax.set_yticks(range(0,rows))
    ax.set_xticklabels(range(0, cols ))
    ax.set_yticklabels(range(0, rows ))
    
    # Add horizontal and vertical lines
    for i in range(0,rows):
        ax.axhline(y=i + 0.5, color='white', linewidth=1)
    for j in range(0,cols):
        ax.axvline(x=j + 0.5, color='white', linewidth=1)

    # Show the grid
    plt.show()  
#------------------------------------------
def rotateZ(vec,theta):
    #theta in radius
    #vector dim = (2,1)
    vec = vec.reshape((2,1))
    RoatMat = np.array([[np.cos(theta)   ,   -np.sin(theta)   ],
                        [np.sin(theta)   ,    np.cos(theta)   ]])
    RoatMat = RoatMat.reshape((2,2))
    return RoatMat @ vec
#-----------------------------------------
def coordinateToGridIndex(x: float, y: float,dim:int):
    #in:  x,     y, dim
    #out: row, col, msg
    row_max=dim
    col_max=dim
    if(x >= row_max or y >= row_max or x < 0 or y < 0):
        print("[Error] index out of range")
        return None,None,-1
    row = round(row_max - 1 - y)
    col = round(x)
    return row,col,1

def girdIndexToCoordinates(row:int, col:int, dim:int):
    #in: row,  col, dim
    #out: x , y, msg
    if(row > dim-1 or col > dim-1 or row < 0 or col < 0):
        print("[Error] index out of range")
        return None,None,-1  
    x = col
    y = dim -1-row  
    return x,y,1
def lineToGridIndexList(p1:np.array,p2:np.array, dim:int, step=0.5):
    #out: list of index pair
    v12 = p2-p1
    v12_mag = np.linalg.norm(v12)
    v12_norm =  v12 / np.linalg.norm(v12)
    
    index_list =[]
    i = 0
    
    row,col,msg = coordinateToGridIndex(p1[0],p1[1],dim)
    if(msg == -1):
        return None, -1
    
    row_prev,col_prev = row,col
    index_list.append(np.array([row,col]))
    print("row:{0} ; col:{1}".format(row,col))
    while(v12_mag >= np.linalg.norm(i * v12_norm) ):
        p = i * v12_norm + p1
        row,col,msg = coordinateToGridIndex(p[0],p[1],dim)
        if(msg == -1):
            return None, -1
        i = i+1
        if(row_prev == row and col_prev == col):
            # print("same")
            continue
        else:
            row_prev=row
            col_prev=col
            index_list.append(np.array([row,col]))
            # print("row:{0} ; col:{1}".format(row,col))    
    return index_list,1   
def circleToGridIndexList(center:np.array, radius:float, dim: int,step=0.5):
    d_theta = step/radius # in rad
    
    
    vec_radial = radius * np.array([1,0])
    index_list = []
    i = 0
    
    p = center + vec_radial
    
    # print(p[0])
    # print(p[1])
    row,col,msg = coordinateToGridIndex(p[0],p[1],dim)
    if(msg == -1):
        return None, -1
    
    row_prev,col_prev = row,col
    index_list.append(np.array([row,col]))
    # print("row:{0} ; col:{1}".format(row,col))
    
    while (d_theta * i < 2 * np.pi):
        p = center + rotateZ(vec_radial, d_theta*i).flatten()
        print("point{0}".format(p))
        row,col,msg = coordinateToGridIndex(p.flatten()[0],p.flatten()[1],dim)
        if(msg == -1):
            return None, -1
        i+=1
        if(row_prev == row and col_prev == col):
            print("same")
            continue
        else:
            row_prev=row
            col_prev=col
            print(np.array([row,col]))
            index_list.append(np.array([row,col]))   
    
    return index_list,1  
    
     
def fillGirdWithIndex(index_list,grid):
    for indexPair in index_list:
        row=indexPair[0]
        col=indexPair[1]
        if(grid[row][col]==0):
            grid[row][col]=1
            # print("fill row{0} col{1} num{2}".format(row,col,1))
        elif(grid[row][col]==1):
            grid[row][col]=1
            #Collide!!
            # print("fill row{0} col{1} num{2}".format(row,col,2))
    return grid
def fillGeoPrimitiveWithIndex(geo:GeoPrimitive, grid):
    dim=grid.shape[0]
    print("dim={0}".format(dim))
    if(type(geo)==Point):
        row,col,msg=coordinateToGridIndex(geo.x,geo.y,dim)
        if(msg==-1):
            print("[ERROR][coordinateToGridIndex]")
            return None,-1
        grid[row][col]=1
        return grid,1
    
    if(type(geo)==Segment):
        p1 = np.array([geo.p1.x,geo.p1.y])
        p2 = np.array([geo.p2.x,geo.p2.y])
        index_list, msg = lineToGridIndexList(p1,p2, dim=dim, step=0.5)
        if(msg==-1):
            print("[ERROR][lineToGridIndexList]")
            return None,-1
        # print(index_list)
        grid = fillGirdWithIndex(index_list,grid)
        return grid,1
        
    if(type(geo)==Triangle):
        p1 = np.array([geo.v1.x,geo.v1.y])
        p2 = np.array([geo.v2.x,geo.v2.y])
        p3 = np.array([geo.v3.x,geo.v3.y])
        
        p_s = p1
        p_prev = p1
        for p in [p1,p2,p3]:
            if(p[0] == p_prev[0] and p[1] == p_prev[1]):
                continue
            
            index_list, msg = lineToGridIndexList(p_prev,p, dim=dim, step=0.5)
            if(msg==-1):
                print("[ERROR][lineToGridIndexList]")
                return None,-1            
            grid = fillGirdWithIndex(index_list,grid)
            
            p_prev=p
        
        #close the end point and starting point
        index_list, msg = lineToGridIndexList(p3,p_s, dim=dim, step=0.5)
        if(msg==-1):
            print("[ERROR][lineToGridIndexList]")
            return None,-1            
        grid = fillGirdWithIndex(index_list,grid)
        return grid,1
    if(type(geo)==Circle):
        center = np.array([geo.center.x,geo.center.y])
        radius = geo.radius
        index_list, msg = circleToGridIndexList(center, radius, dim=dim,step=0.5)
        if(msg==-1):
            print("[ERROR][circleToGridIndexList]")
            return None,-1
        
        grid = fillGirdWithIndex(index_list,grid)
        return grid,1
    if(type(geo)==Polygon): 
        
        p_s = np.array([geo.vertices[0].x,geo.vertices[0].y])
        p_e = np.array([geo.vertices[-1].x,geo.vertices[-1].y])
        p_prev = p_s
        for point in geo.vertices:
            p = np.array([point.x,point.y])
            if(p[0] == p_prev[0] and p[1] == p_prev[1]):
                continue
            
            index_list, msg = lineToGridIndexList(p_prev,p, dim=dim, step=0.5)
            if(msg==-1):
                print("[ERROR][lineToGridIndexList]")
                return None,-1            
            grid = fillGirdWithIndex(index_list,grid)
            
            p_prev=p
        
        #close the end point and starting point
        index_list, msg = lineToGridIndexList(p_e,p_s, dim=dim, step=0.5)
        if(msg==-1):
            print("[ERROR][lineToGridIndexList]")
            return None,-1            
        grid = fillGirdWithIndex(index_list,grid)
        return grid,1

if __name__ == "__main__":
    dim=20
    row_max=dim
    col_max=dim
    #test point input
    grid = np.zeros((row_max,col_max))
    point = Point(1,2)
    grid,msg=fillGeoPrimitiveWithIndex(point, grid)
    print(msg)
    visualize_grid(grid)

    #test segment input
    grid = np.zeros((row_max,col_max))
    p1 = Point(1,2)
    p2 = Point(5,6)
    segment = Segment(p1,p2)
    grid,msg=fillGeoPrimitiveWithIndex(segment, grid)
    print(msg)
    visualize_grid(grid)

    #test triangle input
    grid = np.zeros((row_max,col_max))
    p1 = Point(1,2)
    p2 = Point(8,9)
    p3 = Point(1,6)

    tri = Triangle(p1,p2,p3)
    grid,msg=fillGeoPrimitiveWithIndex(tri, grid)
    print(msg)
    visualize_grid(grid)

    #test Circle input 
    circle = Circle(center=Point(5,5),radius=3.)
    grid,msg=fillGeoPrimitiveWithIndex(circle,grid)
    print(msg)
    visualize_grid(grid)

    #test polygon input 
    grid = np.zeros((row_max,col_max))
    vertices=[]
    vertices.append(Point(2,2))
    vertices.append(Point(2,5))
    vertices.append(Point(5,5))
    vertices.append(Point(5,2))
    poly = Polygon(vertices=vertices)
    grid,msg=fillGeoPrimitiveWithIndex(poly,grid)
    print(msg)
    visualize_grid(grid)
    
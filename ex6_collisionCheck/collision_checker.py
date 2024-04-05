from typing import List
from dg_commons import SE2Transform
from pdm4ar.exercises.ex06.collision_primitives import CollisionPrimitives
from pdm4ar.exercises_def.ex06.structures import (
    Polygon,
    GeoPrimitive,
    Point,
    Segment,
    Circle,
    Triangle,
    Path,
)
import shapely
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from shapely import STRtree
from shapely import box
from shapely import LineString as LineString_s
from shapely import Point as Point_s
from shapely import Polygon as Polygon_s
#-------------my functions-----------------------------
def lineSegmentToShapely(seg:Segment)->LineString_s:
    p1 = pointToNumpy(seg.p1)
    p2 = pointToNumpy(seg.p2)
    return LineString_s([p1,p2])
def triangleToShapely(tri:Triangle)->Polygon_s:
    p1 = pointToNumpy(tri.v1)
    p2 = pointToNumpy(tri.v2)
    p3 = pointToNumpy(tri.v3)
    return Polygon_s([p1,p2,p3])
def polygonToShapely(poly:Polygon)->Polygon_s:
    p_list = []
    for point in poly.vertices:
        p = pointToNumpy(point)
        p_list.append(p)
    return Polygon_s(p_list)
def circleToShapely(circle:Circle)->Point_s:
    center = pointToNumpy(circle.center)
    return Point_s(center).buffer(circle.radius, resolution=6)
def geosToShapely(geos:GeoPrimitive):
    if(type(geos) == Segment):
        return lineSegmentToShapely(geos)
    if(type(geos)==Triangle):
        return triangleToShapely(geos)
    if(type(geos)==Polygon):
        return polygonToShapely(geos)
    if(type(geos)==Circle):
        return circleToShapely(geos)
def lineSegmentToShapelyPointList(seg:Segment,step=0.2):
    ps = pointToNumpy(seg.p1)
    pe = pointToNumpy(seg.p2)
    vec_se = pe-ps
    dis = np.linalg.norm(pe-ps)
    vec_se_norm = vec_se/dis
    i = 0
    pointList_s = []
    while i * step <= dis:
        pointList_s.append(Point_s(ps + i * step * vec_se_norm))
        i = i + 1
    return pointList_s    

def pointToNumpy(point:Point):
    return np.array([point.x,point.y])
def numpyToPoint(p:np.array):
    return Point(p[0],p[1]) 
def rotateZ(vec,theta):
    #theta in radius
    #vector dim = (2,1)
    vec = vec.reshape((2,1))
    RoatMat = np.array([[np.cos(theta)   ,   -np.sin(theta)   ],
                        [np.sin(theta)   ,    np.cos(theta)   ]])
    RoatMat = RoatMat.reshape((2,2))
    return RoatMat @ vec
def C_BI(vec,theta):# find the vector viewed from an rotated frame B, theta is the rotation of B frame relative to I
    #theta in radius
    #vector dim = (2,1)
    vec = vec.reshape((2,1))
    RoatMat = np.array([[np.cos(-theta)   ,   -np.sin(-theta)   ],
                        [np.sin(-theta)   ,    np.cos(-theta)   ]])
    RoatMat = RoatMat.reshape((2,2))
    return RoatMat @ vec
def normalGivenLineSegment(p1:Point,p2:Point):
    p1 = np.array([p1.x, p1.y]).reshape(2,1)
    p2 = np.array([p2.x, p2.y]).reshape(2,1)
    
    cen_12 = (p1 + p2)/2
    
    vec_12 = p2-p1
    vec_21 = p1-p2
    
    vec_12_mag = np.linalg.norm(vec_12)
    if(vec_12_mag == 0):
        return None, -1
    vec_12_norm = vec_12/vec_12_mag
    
    normal_12 = rotateZ(vec_12_norm , np.pi/2)
    return normal_12, 1
def coordinateToGridIndex(x: float, y: float,dim:int,anchor_x=-20,anchor_y=-20):
    #in:  x,     y, dim
    #out: row, col, msg
    row_max=dim
    col_max=dim
    x = float(x-anchor_x)
    y = float(y-anchor_y)
    if(x >= row_max or y >= row_max or x < 0 or y < 0):
        print("[Error] [coordinateToGridIndex]x,y out of range: x={0} y={0}".format(x+anchor_x,y+anchor_y))
        return None,None,-1
    row = round(row_max - 1 - y)
    col = round(x)
    return row,col,1

def girdIndexToCoordinates(row:int, col:int, dim:int,anchor_x=-20,anchor_y=-20):
    #in: row,  col, dim
    #out: x , y, msg
    if(row > dim-1 or col > dim-1 or row < 0 or col < 0):
        print("[Error] index out of range")
        return None,None,-1  
    x = col
    y = dim -1-row 
    
    x=x+anchor_x
    y=y+anchor_y
     
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
    # print("row:{0} ; col:{1}".format(row,col))
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
        # print("point{0}".format(p))
        row,col,msg = coordinateToGridIndex(p.flatten()[0],p.flatten()[1],dim)
        if(msg == -1):
            return None, -1
        i+=1
        if(row_prev == row and col_prev == col):
            # print("same")
            continue
        else:
            row_prev=row
            col_prev=col
            # print(np.array([row,col]))
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

def fillGeoPrimitive(geo:GeoPrimitive, grid):
    dim=grid.shape[0]
    # print("dim={0}".format(dim))
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
    
def checkLineSegmentCollisionWithGrid(p1:np.array,p2:np.array,grid, dim:int, step=0.5)->bool:
    #out: list of index pair
    v12 = p2-p1
    v12_mag = np.linalg.norm(v12)
    v12_norm =  v12 / np.linalg.norm(v12)
    

    i = 0
    
    row,col,msg = coordinateToGridIndex(p1[0],p1[1],dim)
    if(msg == -1):
        print(["[Error][checkLineSegmentCollisionWithGrid]"])
        return None
    
    row_prev,col_prev = row,col

    # print("row:{0} ; col:{1}".format(row,col))
    while(v12_mag >= np.linalg.norm(i * v12_norm) ):
        p = i * v12_norm + p1
        row,col,msg = coordinateToGridIndex(p[0],p[1],dim)
        if(msg == -1):
            print(["[Error][checkLineSegmentCollisionWithGrid]"])
            return None
        i = i+1
        if(row_prev == row and col_prev == col):
            # print("same")
            continue
        elif(abs(row_prev-row)==1 and abs(col_prev-col)==1):
            if(grid[row][col]==1 or grid[row+1][col]==1 or grid[row-1][col]==1 or grid[row][col+1]==1 or grid[row][col-1]==1):
                
                if(grid[row][col]==1):
                    grid[row][col]=2
                elif(grid[row+1][col]==1):
                    grid[row+1][col]=2
                elif(grid[row-1][col]==1):
                    grid[row-1][col]=2
                elif(grid[row][col+1]==1):
                    grid[row][col+1]=2
                elif(grid[row][col-1]==1):
                    grid[row][col-1]=2

                        
                return True
        else:
            row_prev=row
            col_prev=col
            if(grid[row][col]==1):
                grid[row][col]=2
                return True                          
    return False 
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
#------------------------------------------------------



##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not

COLLISION_PRIMITIVES = {
    Point: {
        Circle: lambda x, y: CollisionPrimitives.circle_point_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_point_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_point_collision(y, x),
    },
    Segment: {
        Circle: lambda x, y: CollisionPrimitives.circle_segment_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_segment_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_segment_collision_aabb(y, x),
    },
    Triangle: {
        Point: CollisionPrimitives.triangle_point_collision,
        Segment: CollisionPrimitives.triangle_segment_collision,
    },
    Circle: {
        Point: CollisionPrimitives.circle_point_collision,
        Segment: CollisionPrimitives.circle_segment_collision,
    },
    Polygon: {
        Point: CollisionPrimitives.polygon_point_collision,
        Segment: CollisionPrimitives.polygon_segment_collision_aabb,
    },
}


def check_collision(p_1: GeoPrimitive, p_2: GeoPrimitive) -> bool:
    """
    Checks collision between 2 geometric primitives
    Note that this function only uses the functions that you implemented in CollisionPrimitives class.
        Parameters:
                p_1 (GeoPrimitive): Geometric Primitive
                p_w (GeoPrimitive): Geometric Primitive
    """
    assert type(p_1) in COLLISION_PRIMITIVES, "Collision primitive does not exist."
    assert (
        type(p_2) in COLLISION_PRIMITIVES[type(p_1)]
    ), "Collision primitive does not exist."

    collision_func = COLLISION_PRIMITIVES[type(p_1)][type(p_2)]

    return collision_func(p_1, p_2)


##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not

def geo_primitive_to_shapely(p: GeoPrimitive):
    if isinstance(p, Point):
        return shapely.Point(p.x, p.y)
    elif isinstance(p, Segment):
        return shapely.LineString([[p.p1.x, p.p1.y], [p.p2.x, p.p2.y]])
    elif isinstance(p, Circle):
        return shapely.Point(p.center.x, p.center.y).buffer(p.radius)
    elif isinstance(p, Triangle):
        return shapely.Polygon([[p.v1.x, p.v1.y], [p.v2.x, p.v2.y], [p.v3.x, p.v3.y]])
    else: #Polygon
        vertices = []
        for vertex in p.vertices:
            vertices += [(vertex.x, vertex.y)]
        return shapely.Polygon(vertices)


class CollisionChecker:
    """
    This class implements the collision check ability of a simple planner for a circular differential drive robot.

    Note that check_collision could be used to check collision between given GeoPrimitives
    check_collision function uses the functions that you implemented in CollisionPrimitives class.
    """

    def __init__(self):
        pass

    def path_collision_check(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        #-------------Error Handle-------------
        if(t.waypoints == None):print("[ERROR]Path.waypoints == None")
        #--------------------------------------
        p_prev = t.waypoints[0]
        index = 0
        index_collide = []
        for p in t.waypoints:
            
            #skip the first point
            if(p == p_prev):
                continue
            
            segment = Segment(p1=p_prev, p2=p)
            
            #Handle case when segment.p1==segment.p2 and normal_12 is null
            normal_12, error_flag = normalGivenLineSegment(segment.p1,segment.p2) 
            if(error_flag == -1):
                continue
            
            #compute upper segment
            p1_upper = Point(x = segment.p1.x + normal_12[0] * r , y = segment.p1.y + normal_12[1] * r)
            p2_upper = Point(x = segment.p2.x + normal_12[0] * r , y = segment.p2.y + normal_12[1] * r)
            segment_upper = Segment(p1 = p1_upper, p2 = p2_upper )
            
            #compute lower segment
            p1_lower = Point(x = segment.p1.x - normal_12[0] * r , y = segment.p1.y - normal_12[1] * r)
            p2_lower = Point(x = segment.p2.x - normal_12[0] * r , y = segment.p2.y - normal_12[1] * r)
            segment_lower = Segment(p1 = p1_lower, p2 = p2_lower )
            
            
            for obstacle in obstacles:
                if( check_collision(p_1=obstacle, p_2=segment_upper) or check_collision(p_1=obstacle, p_2=segment_lower)):
                    index_collide.append(index)
            
            p_prev = p
            index += 1
                
        return index_collide

    def path_collision_check_occupancy_grid(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will generate an occupancy grid of the given map.
        Then, occupancy grid will be used to check collisions.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        #-------------Error Handle-------------
        if(t.waypoints == None):print("[ERROR]Path.waypoints == None")
        #--------------------------------------
        
        #--------1. Convert Obstacles to the Occupancy Grid----------------------------
        grid = np.zeros((200,200))
        dim=200
        for obstacle in obstacles:
            grid,msg=fillGeoPrimitive(obstacle,grid)
            if(msg == -1):
                print("[ERROR][fillGeoPrimitive]")
                break
        
        
        #---------2. Collision Detection in Occupancy Grid----------------------
        
        
        p_prev = t.waypoints[0]
        index = 0
        index_collide = []
        checked_segment = []
        for p in t.waypoints:
            
            #skip the first point
            if(p == p_prev):
                continue
            
            segment = Segment(p1=p_prev, p2=p)
            
            #Handle case when segment.p1==segment.p2 and normal_12 is null
            normal_12, error_flag = normalGivenLineSegment(segment.p1,segment.p2) 
            if(error_flag == -1):
                continue
            
            #compute upper segment
            p1_upper = Point(x = segment.p1.x + normal_12[0] * r , y = segment.p1.y + normal_12[1] * r)
            p2_upper = Point(x = segment.p2.x + normal_12[0] * r , y = segment.p2.y + normal_12[1] * r)
            segment_upper = Segment(p1 = p1_upper, p2 = p2_upper )
            
            #compute lower segment
            p1_lower = Point(x = segment.p1.x - normal_12[0] * r , y = segment.p1.y - normal_12[1] * r)
            p2_lower = Point(x = segment.p2.x - normal_12[0] * r , y = segment.p2.y - normal_12[1] * r)
            segment_lower = Segment(p1 = p1_lower, p2 = p2_lower )
            
            flag1 = checkLineSegmentCollisionWithGrid(p1=pointToNumpy(segment_upper.p1),p2=pointToNumpy(segment_upper.p2),grid=grid, dim=dim, step=0.5)
            flag2 = checkLineSegmentCollisionWithGrid(p1=pointToNumpy(segment_lower.p1),p2=pointToNumpy(segment_lower.p2),grid=grid, dim=dim, step=0.5)
            if(flag1 or flag2):
                index_collide.append(index)
                
            checked_segment.append(segment_lower)
            checked_segment.append(segment_upper)
          
            p_prev = p
            index += 1
        #------------Visualization-------------
        for geo in checked_segment:
            grid,msg=fillGeoPrimitive(geo, grid)
        visualize_large_grid(grid)        
        #----------------------------------
        return index_collide
        return[]

    def path_collision_check_r_tree(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will build an R-Tree of the given obstacles.
        You are free to implement your own R-Tree or you could use STRTree of shapely module.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        
        #-------------Error Handle-------------
        if(t.waypoints == None):print("[ERROR]Path.waypoints == None")
        #--------------------------------------
        
        Geos_shapely = []
        
        #1. Add obstacles to the R tree
        for obstacle in obstacles:
            Geo_shapely = geosToShapely(obstacle)
            Geos_shapely.append(Geo_shapely)
        tree = STRtree(Geos_shapely)
        
        
        #2. for each path segment, use tree to do collision detect
        p_prev = t.waypoints[0]
        index = 0
        index_collide = []
        for p in t.waypoints:
            
            #skip the first point
            if(p == p_prev):
                continue
            
            segment = Segment(p1=p_prev, p2=p)
            
            #--------------------Method 1 : check using two parallel line---------------
            # #Handle case when segment.p1==segment.p2 and normal_12 is null
            # normal_12, error_flag = normalGivenLineSegment(segment.p1,segment.p2) 
            # if(error_flag == -1):
            #     continue
            
            # #compute upper segment
            # p1_upper = Point(x = segment.p1.x + normal_12[0] * r , y = segment.p1.y + normal_12[1] * r)
            # p2_upper = Point(x = segment.p2.x + normal_12[0] * r , y = segment.p2.y + normal_12[1] * r)
            # segment_upper = Segment(p1 = p1_upper, p2 = p2_upper )
            # segment_upper_shapely_PointList = lineSegmentToShapelyPointList(segment_upper)
            
            # #compute lower segment
            # p1_lower = Point(x = segment.p1.x - normal_12[0] * r , y = segment.p1.y - normal_12[1] * r)
            # p2_lower = Point(x = segment.p2.x - normal_12[0] * r , y = segment.p2.y - normal_12[1] * r)
            # segment_lower = Segment(p1 = p1_lower, p2 = p2_lower )
            # segment_lower_shapely_PointList = lineSegmentToShapelyPointList(segment_lower)
            
            # #compute starting circle            
            # if(tree.query(segment_upper_shapely_PointList, predicate="dwithin", distance=0.1).size==0 and tree.query(segment_lower_shapely_PointList, predicate="dwithin", distance=0.1).size==0):
            #     pass#no collision
            # else:
            #     index_collide.append(index)#collide!
            
            #--------------------Method 2 : check using distance---------------
            segment_shapely_PointList = lineSegmentToShapelyPointList(segment)
            index_query, distance = tree.query_nearest(segment_shapely_PointList, return_distance=True)
            if(distance[distance<=r].size!=0):
                index_collide.append(index)
            
            p_prev = p
            index += 1
                
        return index_collide

    def collision_check_robot_frame(
        self,
        r: float,
        current_pose: SE2Transform,
        next_pose: SE2Transform,
        observed_obstacles: List[GeoPrimitive],
    ) -> bool:
        """
        Returns there exists a collision or not during the movement of a circular differential drive robot until its next pose.

            Parameters:
                    r (float): Radius of circular differential drive robot
                    current_pose (SE2Transform): Current pose of the circular differential drive robot
                    next_pose (SE2Transform): Next pose of the circular differential drive robot
                    observed_obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives in robot frame
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        Geos_shapely = []
        
        #1. Add obstacles to the R tree
        for obstacle in observed_obstacles:
            Geo_shapely = geosToShapely(obstacle)
            Geos_shapely.append(Geo_shapely)
        tree = STRtree(Geos_shapely)
        
        #2. check using diatance to a point list
        theta_pose = current_pose.theta
        I_p1 = current_pose.p
        I_p2 = next_pose.p
        I_vec_12 = I_p2-I_p1 
        R_vec_12 = rotateZ(I_vec_12,-theta_pose).flatten()
        R_p_current = np.array([0,0])
        segment_12_RFrame = Segment(numpyToPoint(R_p_current), numpyToPoint(R_p_current + R_vec_12))
        segment_shapely_PointList = lineSegmentToShapelyPointList(segment_12_RFrame)
        index_query, distance = tree.query_nearest(segment_shapely_PointList, return_distance=True)
        if(distance[distance<=r].size!=0):
            return True     
        return False

    def path_collision_check_safety_certificate(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will implement the safety certificates procedure for collision checking.
        You are free to use shapely to calculate distance between a point and a GoePrimitive.
        For more information, please check Algorithm 1 inside the following paper:
        https://journals.sagepub.com/doi/full/10.1177/0278364915625345.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        Geos_shapely = []
        
        #1. Add obstacles to the R tree
        for obstacle in  obstacles:
            Geo_shapely = geosToShapely(obstacle)
            Geos_shapely.append(Geo_shapely)
        tree_obs = STRtree(Geos_shapely)
        
        #2. for each path segment, use tree to do collision detect, USE CERTIFICATE CIRCLE
        p_prev = t.waypoints[0]
        index = 0
        index_collide = []
        cirtificate_circles = Point_s(0,0).buffer(0.1, resolution=6)
        for p in t.waypoints:     
            #skip the first point
            if(p == p_prev):
                continue
            
            segment = Segment(p1=p_prev, p2=p)
            segment_shapely_PointList = lineSegmentToShapelyPointList(segment)
            for point_s in segment_shapely_PointList:             
                index_query, distance =tree_obs.query_nearest(point_s, return_distance=True)
                
                if(cirtificate_circles.contains(point_s)):
                    continue
                
                if(distance <= r):
                    index_collide.append(index)
                    break
                else:
                    cirtificate_circles = point_s.buffer(distance-r, resolution=10)[0]                              
            p_prev = p
            index += 1
                
        return index_collide

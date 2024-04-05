from pdm4ar.exercises_def.ex06.structures import *
import numpy as np
import triangle
from triangle import triangulate
from triangle import convex_hull
from typing import List, Tuple, Any, Sequence
#---------------my function----------------------
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
def vecDir(vec):
    if(vec[0] == 0):
        if(vec[1] > 0):
            return np.pi/2
        if(vec[1] < 0):
            return np.pi*(3/2)
    if(vec[1] == 0):
        if(vec[0] > 0):
            return 0
        if(vec[0] < 0):
            return np.pi
    theta = np.arctan2(vec[1],vec[0])
    return theta
#-----------------------------------------------
class CollisionPrimitives:
    """
    Class of collusion primitives
    """

    @staticmethod
    def circle_point_collision(c: Circle, p: Point) -> bool:
        dis = np.sqrt((c.center.x - p.x)**2 + (c.center.y - p.y)**2)
        if(dis <= c.radius):
            return True
        return False

    @staticmethod
    def triangle_point_collision(t: Triangle, p: Point) -> bool:
        point = np.array([p.x , p.y ]).reshape(2,1)
        v1 = np.array([t.v1.x , t.v1.y ]).reshape(2,1)
        v2 = np.array([t.v2.x , t.v2.y ]).reshape(2,1)
        v3 = np.array([t.v3.x , t.v3.y ]).reshape(2,1)
        
        cen_12 = (v1 + v2)/2
        cen_23 = (v2 + v3)/2
        cen_31 = (v3 + v1)/2
        
        vec_12 = v2-v1
        vec_23 = v3-v2
        vec_31 = v1-v3
        
        vec_12_mag = np.linalg.norm(vec_12)
        vec_23_mag = np.linalg.norm(vec_23)
        vec_31_mag = np.linalg.norm(vec_31)
        
        vec_12_norm = vec_12/vec_12_mag
        vec_23_norm = vec_23/vec_23_mag
        vec_31_norm = vec_31/vec_31_mag
        
        normal_12 = rotateZ(vec_12_norm , np.pi/2)
        normal_23 = rotateZ(vec_23_norm , np.pi/2)
        normal_31 = rotateZ(vec_31_norm , np.pi/2)
        
        centroid = (v1 + v2 + v3)/3
        
        #filp the normal direction if it point inward
        if(((centroid - cen_12).T @ normal_12 > 0) and ((centroid - cen_23).T @ normal_23 > 0) and ((centroid - cen_31).T @ normal_31 > 0)):
            normal_12 = -normal_12
            normal_23 = -normal_23
            normal_31 = -normal_31
            
        #collision if it is in or inside the triangle
        if(((point - cen_12).T @ normal_12 <= 0 ) and ((point - cen_23).T @ normal_23 <= 0  ) and ((point - cen_31).T @ normal_31 <= 0 ) ):
            return True
         
        return False

    @staticmethod
    def polygon_point_collision(poly: Polygon, p: Point) -> bool:
        vertives = []
        for point in poly.vertices:
            vertives.append(np.array([point.x, point.y]))
            
        t = triangulate({'vertices': vertives}, 'a0.2')
        totalvertices = t['vertices'].tolist()
        for tri in t['triangles'].tolist():
            v1 = Point(x=totalvertices[tri[0]][0], y=totalvertices[tri[0]][1])
            v2 = Point(x=totalvertices[tri[1]][0], y=totalvertices[tri[1]][1])
            v3 = Point(x=totalvertices[tri[2]][0], y=totalvertices[tri[2]][1])
            triangle = Triangle(v1=v1 , v2=v2, v3=v3 )   
            if(CollisionPrimitives.triangle_point_collision(triangle,p)):
                return True
        return False

    @staticmethod
    def circle_segment_collision(c: Circle, segment: Segment) -> bool:
        center = np.array([c.center.x , c.center.y ]).reshape(2,1)
        radius = c.radius
        p1 = np.array([segment.p1.x, segment.p1.y]).reshape(2,1)
        p2 = np.array([segment.p2.x, segment.p2.y]).reshape(2,1)
        
        cen_12 = (p1 + p2)/2
        
        vec_12 = p2-p1
        vec_21 = p1-p2
        
        vec_12_mag = np.linalg.norm(vec_12)
        
        vec_12_norm = vec_12/vec_12_mag
        
        normal_12 = rotateZ(vec_12_norm , np.pi/2)
        
        vec_cen12Tocenter = cen_12 - center
        
        projection = vec_cen12Tocenter.T @ normal_12
        
        dis_p1TOc = np.linalg.norm(center - p1)
        dis_p2TOc = np.linalg.norm(center - p2)
        
        vec_p1TOc = center - p1
        vec_p2TOc = center - p2
        
        if(((np.abs(projection) <= radius) and (vec_p1TOc.T @ vec_12 >= 0) and (vec_p2TOc.T @ vec_21 >= 0)) or (dis_p1TOc <= radius) or (dis_p2TOc <= radius) ):
            return True
        
        return False

    @staticmethod
    def triangle_segment_collision(t: Triangle, segment: Segment) -> bool:
        v_a = np.array([t.v1.x , t.v1.y ])
        v_b = np.array([t.v2.x , t.v2.y ])
        v_c = np.array([t.v3.x , t.v3.y ])
        p1 = np.array([segment.p1.x, segment.p1.y])
        p2 = np.array([segment.p2.x, segment.p2.y])
        cen_12 = (p1 + p2)/2
        
        vec_12 = p2-p1
        vec_21 = p1-p2
        
        vec_12_mag = np.linalg.norm(vec_12)
        
        #-----ErrorHandle---------------
        #-------------------------------
        vec_12_norm = vec_12/vec_12_mag
        
        #calculate vertices of polytope resulting from the minkovsky sum of triangle and line segment
        vertices_np = []
        vertices_np.append(v_a + 0.5 * vec_12.flatten())
        vertices_np.append(v_a - 0.5 * vec_12.flatten())
        vertices_np.append(v_b + 0.5 * vec_12.flatten())
        vertices_np.append(v_b - 0.5 * vec_12.flatten())
        vertices_np.append(v_c + 0.5 * vec_12.flatten())
        vertices_np.append(v_c - 0.5 * vec_12.flatten())
        
        segments = convex_hull(vertices_np)
        
        vertices: List[Point] = []
        for segment in segments:
            vertice = Point(x=vertices_np[segment[0]][0], y=vertices_np[segment[0]][1])
            vertices.append(vertice)
            
        controlPoint = Point(x=cen_12[0],y=cen_12[1])    
        poly= Polygon(vertices=vertices)
        if(CollisionPrimitives.polygon_point_collision(poly,controlPoint)):
            return True
        

        return False

    @staticmethod
    def polygon_segment_collision(p: Polygon, segment: Segment) -> bool:
        p1 = np.array([segment.p1.x, segment.p1.y])
        p2 = np.array([segment.p2.x, segment.p2.y])
        cen_12 = (p1 + p2)/2    
        vec_12 = p2-p1
        
        #calculate convex hull 
        vertices_np = []
        for point in p.vertices:
            p = np.array([point.x, point.y])
            vertices_np.append(p + 0.5 * vec_12)
            vertices_np.append(p - 0.5 * vec_12)
        segments = convex_hull(vertices_np)
        
        vertices: List[Point] = []
        for segment in segments:
            vertice = Point(x=vertices_np[segment[0]][0], y=vertices_np[segment[0]][1])
            vertices.append(vertice)
            
        controlPoint = Point(x=cen_12[0],y=cen_12[1])    
        poly= Polygon(vertices=vertices)
        
        if(CollisionPrimitives.polygon_point_collision(poly,controlPoint)):
            return True
        
        return False

    @staticmethod
    def polygon_segment_collision_aabb(poly: Polygon, segment: Segment) -> bool:

        
        
        #1. Find AABB of line segment in axis parallel&orthogonal to segment
        p1 = np.array([segment.p1.x, segment.p1.y])
        p2 = np.array([segment.p2.x, segment.p2.y]) 
        vec_12 = p2-p1
        
        #find rotation angle of B frame relative to I frame,where x axis of B frame parallel to the segment
        theta = vecDir(vec_12)#in radius
        
        #view p1p2 in B frame
        B_p1 = C_BI(p1,theta)
        B_p2 = C_BI(p2,theta)
        B_cen_12 = (B_p1 + B_p2)/2
        
        #initialize p_min p_max
        B_p_seg_min_x=B_cen_12[0]
        B_p_seg_min_y=B_cen_12[1]
        B_p_seg_max_x=B_cen_12[0]
        B_p_seg_max_y=B_cen_12[1]
        
        
        for point in [segment.p1,segment.p2]:
            p = np.array([point.x, point.y])
            B_point = Point(C_BI(p,theta)[0], C_BI(p,theta)[1])
            
            if(B_point.x < B_p_seg_min_x):
                B_p_seg_min_x = B_point.x
            if(B_point.x > B_p_seg_max_x):
                B_p_seg_max_x = B_point.x
            if(B_point.y < B_p_seg_min_y):
                B_p_seg_min_y = B_point.y
            if(B_point.y > B_p_seg_max_y):
                B_p_seg_max_y = B_point.y
                
        B_p_seg_min=Point(x=B_p_seg_min_x,y=B_p_seg_min_y)
        B_p_seg_max=Point(x=B_p_seg_max_x,y=B_p_seg_max_y)
              
              
        #2. Find AABB of line polygon in axis parallel&orthogonal to segment        
        center = np.array([poly.center().x,poly.center().y])
        B_center = C_BI(center,theta)
        
        B_p_poly_min_x=B_center[0]
        B_p_poly_min_y=B_center[1]
        B_p_poly_max_x=B_center[0]
        B_p_poly_max_y=B_center[1]
        
        for point in poly.vertices:
            p = np.array([point.x, point.y])
            B_point = Point(C_BI(p,theta)[0], C_BI(p,theta)[1])
            
            if(B_point.x < B_p_poly_min_x):
                B_p_poly_min_x = B_point.x
            if(B_point.x > B_p_poly_max_x):
                B_p_poly_max_x = B_point.x
            if(B_point.y < B_p_poly_min_y):
                B_p_poly_min_y = B_point.y
            if(B_point.y > B_p_poly_max_y):
                B_p_poly_max_y = B_point.y

        #3. use two aabb to check collision
        if(B_p_poly_max_x < B_p_seg_min.x ):
            return False
        if(B_p_poly_min_x > B_p_seg_max.x ):
            return False
        if(B_p_poly_max_y < B_p_seg_min.y ):
            return False
        if(B_p_poly_min_y > B_p_seg_max.y ):
            return False
        
        return True
         
        

    # @staticmethod
    # def _poly_to_aabb(g: Polygon) -> AABB:
    #     # todo feel free to implement functions that upper-bound a shape with an
    #     #  AABB or simpler shapes for faster collision checks
        
    #     # initialize p_min and p_max
    #     center = g.center()
    #     p_min_x=center.x
    #     p_min_y=center.y
    #     p_max_x=center.x
    #     p_max_y=center.y
        
    #     for point in g.vertices:
    #         if(point.x < p_min_x):
    #             p_min_x = point.x
    #         if(point.x > p_max_x):
    #             p_max_x = point.x
    #         if(point.y < p_min_y):
    #             p_min_y = point.y
    #         if(point.y > p_max_y):
    #             p_max_y = point.y
                
            
    #     return AABB(p_min=Point(p_min_x,p_min_y), p_max=Point(p_max_x,p_max_y))



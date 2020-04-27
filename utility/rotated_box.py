import math
import numpy as np



def center_to_angle(center_y, center_x, direction = 'CCW'):
    # angle = 0 for (x, y) = (1, 0), pi for (x, y) = (-1, 0)
    # 0 <= angle < 2pi
    angle = math.atan((center_y - 0.5)/( center_x - 0.5 + 1e-20))
    angle = np.where(np.less_equal(center_x, 0.5) , math.pi - angle, -angle)
    angle = np.where(np.less_equal(angle, 0.0), math.pi + math.pi + angle, angle)
    if direction == 'CW':
        angle = 2*math.pi-angle
    return angle


def center_to_points( center_y, center_x, height, width, direction = 'CCW', is_h_larger=True):
    angle = center_to_angle(center_y, center_x, direction)
    rotation_matrix = np.stack([math.cos(angle), -math.sin(angle),  
                        math.sin(angle),math.cos(angle)], axis=0)
    rotation_matrix = np.reshape(rotation_matrix, (2, 2))
    if is_h_larger:
        height, width = width, height 
    points = np.stack([[ -width / 2,  -height / 2], [ width / 2,  -height / 2], [ width / 2,  height / 2], [ -width / 2,  height / 2] ], axis=0)
    points = np.matmul(points, rotation_matrix) + [center_x, center_y]
    return points


def dist(p1, p2):
    return math.sqrt( (p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
def point_to_center(box): 
    # h > w
    cx = (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4
    cy = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4
    size1 = (dist(box[0], box[1]) +  dist(box[2], box[3]))/2
    size2 = (dist(box[1], box[2]) +  dist(box[3], box[0]))/2
    vec_center = np.array([cx-0.5, cy-0.5])/dist([cx, cy],[0.5,0.5])
    vec_01 = np.array([box[0][0] - box[1][0], box[0][1] - box[1][1]])/dist(box[0], box[1])
    vec_21 = np.array([box[2][0] - box[1][0], box[2][1] - box[1][1]])/dist(box[2], box[1])
    if abs(np.sum(vec_center * vec_01)) > abs(np.sum(vec_center * vec_21)):
      h = size1
      w = size2
    else:
      h = size2
      w = size1
    return cx, cy, w, h
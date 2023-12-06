import numpy as np
from math import pi, cos, sin, sqrt


def skew_symmetric(v):

    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def exp_twist(theta, twist):

    omega = np.array(twist[:3])  # Angular part
    v = np.array(twist[3:])     # Linear part
    
    if np.all(omega == 0):  # Pure translation
        R = np.eye(3)
        p = v * theta
    else:  # Screw motion
        omega_skew = skew_symmetric(omega)
        omega_norm = np.linalg.norm(omega)
        
        R = np.eye(3) + np.sin(theta) * omega_skew + (1 - np.cos(theta)) * np.dot(omega_skew, omega_skew)
        p = (np.eye(3) * theta + (1 - np.cos(theta)) * omega_skew + (theta - np.sin(theta)) * np.dot(omega_skew, omega_skew)).dot(v)
    
    p = p.reshape(3, 1)
    return np.vstack((np.hstack((R, p)), np.array([0, 0, 0, 1])))

def forward_kinematics_poe(joints, L =[5,11,0,5]):

#     twists = np.array([
#         [0, 0, 1, 0, 0, 0],
#         [0, -1, 0, L[0], 0, 0],
#         [0, 0, 0, 0, 0, 1],
#         [0, -1, 0, L[0] + L[1], 0, 0]
#     ])

#     M = np.array([
#         [1, 0, 0, -L[3]],
#         [0, 1, 0, 0],
#         [0, 0, 1, L[0] + L[1]],
#         [0, 0, 0, 1]
#     ])
    twists = np.array([
        [0, 0, 1, 0, 0, 0],
        [0, -1, 0, L[0], 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, -1, 0,L[0], 0, -L[2]]
    ])

    M = np.array([
        [1, 0, 0, L[1]+L[3]],
        [0, 1, 0, 0],
        [0, 0, 1, L[0]],
        [0, 0, 0, 1]
    ])

    T = np.eye(4)
    for i in range(len(twists)):
        T = np.dot(T, exp_twist(joints[i], twists[i]))

    T = np.dot(T, M)
    T=np.around(T, 4)
    return T


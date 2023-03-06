from casadi import *
import numpy as np

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params

import time
import os

from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class Quasistatic_Control_Manager_Worm: 

    def __init__(self, robot_arm_model, solver): 

        self._robot_arm_model = robot_arm_model
        self._robot1 = Multiple_Shooting_Solver(robot_arm_model)
        self._num_tendons = self._robot_arm_model._tau.shape[0]
        self._integrator_with_boundaries = self._robot_arm_model._create_static_integrator_with_boundaries()
        self._integrator_static_full = self._robot_arm_model._create_static_integrator()
        self._solver_static, self._integrator_static = self._robot1.create_static_solver()
        self._wrench_lb = -50
        self._wrench_ub = 50
        self._tendon_radiuses = self._robot_arm_model._tendon_radiuses_numpy
        self._MAX_ITERATIONS_STATIC = 10000
        self._Kse = np.array([self._robot_arm_model.get_robot_arm_params().get_Kse()])[0, :, :]
        self._Kbt = np.array([self._robot_arm_model.get_robot_arm_params().get_Kbt()])[0, :, :]
        self._Kse_inv = np.linalg.inv(self._robot_arm_model.get_robot_arm_params().get_Kse())
        self._Kbt_inv = np.linalg.inv(self._robot_arm_model.get_robot_arm_params().get_Kbt())
        self._e3 = np.array([0, 0, 1])
        self._boundary_conditions = np.zeros((6, 1))
        self._current_states = np.zeros((19, 1))
        self._boundary_Jacobian = np.zeros((6, 9))
        self._init_wrench = np.zeros(6)
        self._current_full_states = np.zeros((13+self._num_tendons, self._robot_arm_model.get_num_integration_steps()+1))
        self._current_full_states[3, :] = 1
        self._time_step = 1e-2
        self._t = 0.0
        self._history_states = np.zeros((14+self._num_tendons, self._robot_arm_model.get_num_integration_steps()+1, 10000))
        self.wrench_lb = -50
        self.wrench_ub = 50
        self.pos_ub = 5
        self.eta_ub = 1.05
        self.tension_max = 50
        self._differential_kinematics_solver = solver

    def set_time_step(self, time_step): 
        
        self._time_step = time_step

    def set_data_diffential_solver(self, p, yref, yref_teminal, states):

        pass

    def set_tensions_MS_solver(self, tension): 

        pass 

    def solve_full_shape(self): 

        pass 

    def save_step(self): 

        pass 

    def solve_Jacobians(self): 

        pass 

    def compute_boundary_conditions(self): 

        pass 

    def solve_differential_inverse_kinematics(self, task_vel): 

        pass 

    def apply_tension_differential_position_boundary(self, delta_q_tau): 

        pass 

    def print_Jacobians(self): 

        pass 

    def print_states(self): 

        pass 

    def solve_static(self): 

        pass 

    def visualise_robot(self): 

        pass 

    def eta_to_Rotation_T_matrix(self, eta): 

        R = np.zeros((3, 3))

        R[0,0] = 2*(eta[0]**2 + eta[1]**2) - 1
        R[0,1] = 2*(eta[1]*eta[2] - eta[0]*eta[3])
        R[0,2] = 2*(eta[1]*eta[3] + eta[0]*eta[2])
        R[1,0] = 2*(eta[1]*eta[2] + eta[0]*eta[3])
        R[1,1] = 2*(eta[0]**2 + eta[2]**2) - 1
        R[1,2] = 2*(eta[2]*eta[3] - eta[0]*eta[1])
        R[2,0] = 2*(eta[1]*eta[3] - eta[0]*eta[2])
        R[2,1] = 2*(eta[2]*eta[3] + eta[0]*eta[1])
        R[2,2] = 2*(eta[0]**2 + eta[3]**2) - 1

        return R

    def DM2Arr(self, dm):
        return np.array(dm.full())

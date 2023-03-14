from casadi import *
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params

import time
import os

from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class Quasistatic_Control_Manager_Full_Robot: 

    def __init__(self, robot_arm_model, solver): 

        self._robot_arm_model = robot_arm_model
        self._robot1 = Multiple_Shooting_Solver(robot_arm_model)
        self._num_tendons = self._robot_arm_model._tau.shape[0]
        self._integrator_with_boundaries_0 = self._robot_arm_model._create_static_integrator_with_boundaries()
        self._integrator_with_boundaries_1 = self._robot_arm_model._create_static_integrator_with_boundaries()

        self._integrator_static_full = self._robot_arm_model._create_static_integrator()
        self._solver_static_full_robot, self._integrator_static_full_robot_step = self._robot1.create_static_solver_full_robot()
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
        self._current_states = np.zeros((20, 1))
        self._boundary_Jacobian = np.zeros((6, 9))
        self._init_wrench = np.zeros(6)
        self._current_full_states = np.zeros((14+self._num_tendons*2, self._robot_arm_model.get_num_integration_steps()*2+1))
        self._current_full_states[3, :] = 1
        self._time_step = 1e-2
        self._t = 0.0
        self._history_states = np.zeros((14+self._num_tendons*2, self._robot_arm_model.get_num_integration_steps()+1, 10000))
        self.wrench_lb = -50
        self.wrench_ub = 50
        self.pos_ub = 5
        self.eta_ub = 1.05
        self.tension_max = 50
        self._differential_kinematics_solver = solver
        self.tension_values = np.zeros(6)

    def set_time_step(self, time_step): 
        
        self._time_step = time_step

    def initialise_static_solver_full_robot(self, initial_solution, tension): 

        self.set_tensions_MS_solver(tension)
        initial_solution[13:19] = tension
        # initial_solution[7:13] = -0.1, 0.2, 0.1, 0.1, 0.05, 0.05
        self._solver_static_full_robot.set(0, 'x', initial_solution)

        subseq_solution = initial_solution

        for i in range(self._robot_arm_model.get_num_integration_steps()*2): 

            self._integrator_static_full_robot_step.set('x', subseq_solution)
            self._integrator_static_full_robot_step.solve()
            subseq_solution = self._integrator_static_full_robot_step.get('x')
            self._solver_static_full_robot.set(i+1, 'x', subseq_solution)

        self.INITIALISED_STATIC_SOLVER = 1

    def set_data_diffential_solver(self, p, yref, yref_teminal, states):

        pass

    def set_tensions_MS_solver(self, tension):

        lbx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self._wrench_lb*np.ones(6), tension, 0))
        ubx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self._wrench_ub*np.ones(6), tension, 0))        

        self.tension_values = tension

        W = np.eye(6)

        self._solver_static_full_robot.cost_set(20, 'W', W)

        self._solver_static_full_robot.constraints_set(0, 'lbx', lbx_0)
        self._solver_static_full_robot.constraints_set(0, 'ubx', ubx_0)

    def solve_full_shape(self): 

        pass 

    def save_step(self): 

        pass 

    def solve_Jacobians(self): 

        pass 

    def solve_differential_inverse_kinematics(self, task_vel): 

        pass 

    def apply_tension_differential_position_boundary(self, delta_q_tau): 

        pass 

    def print_Jacobians(self): 

        pass 

    def print_states(self): 

        pass 

    def solve_full_shape(self): 

        pass 

    def solve_static(self): 

        t = time.time()

        if self.INITIALISED_STATIC_SOLVER:  

            for i in range(self._MAX_ITERATIONS_STATIC): 

                self._solver_static_full_robot.solve()
                print("cost: ", self._solver_static_full_robot.get_cost())

                if self._solver_static_full_robot.get_cost() < 1e-5:

                    print("Number of iterations required: ", i+1)
                    print("Total time taken: ", (time.time() - t), 's')
                    print("Time taken per iteration: ", (time.time() - t)/(i+1), "s.")
                    self._init_sol = self._solver_static_full_robot.get(0, 'x')

                    for k in range(self._robot_arm_model.get_num_integration_steps()*2+1):

                        self._current_full_states[:, k] = self._solver_static_full_robot.get(k, 'x')
                                        
                    self._init_sol_boundaries = np.hstack((self._init_sol, 0*np.ones(3)))
                    # self._integrator_static_full.set('x', self._init_sol)
                    # self._integrator_static_full.solve()
                    # self._init_pose_plus_wrench = self._solver_static_full_robot.get(0, 'x')[0:13]
                    # self._init_wrench = self._solver_static_full_robot.get(0, 'x')[7:13]
                    # self._current_tau = self._solver_static_full_robot.get(0, 'x')[13:13+self._num_tendons]
                    # print(self._init_sol)
                    # print(self._integrator_static_full.get('x'))

                    break

                elif i == self._MAX_ITERATIONS_STATIC - 1 and self._solver_static_full_robot.get_cost() > 1e-5: 

                    print("DID NOT CONVERGE!")
        
        

    def visualise_robot(self): 

        self.solve_full_shape()

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(self._current_full_states[0, 0:self._robot_arm_model.get_num_integration_steps()], self._current_full_states[1, 0:self._robot_arm_model.get_num_integration_steps()], self._current_full_states[2, 0:self._robot_arm_model.get_num_integration_steps()])
        ax.plot(self._current_full_states[0, self._robot_arm_model.get_num_integration_steps(): 2*self._robot_arm_model.get_num_integration_steps()], self._current_full_states[1, self._robot_arm_model.get_num_integration_steps(): 2*self._robot_arm_model.get_num_integration_steps()], self._current_full_states[2, self._robot_arm_model.get_num_integration_steps(): 2*self._robot_arm_model.get_num_integration_steps()])

        # ax.legend()
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0.4, 0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        return 0

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
    
    def _skew(self, x): 

        return np.array([[0, -x[2], x[1]],
                        [x[2], 0, -x[0]],
                        [-x[1], x[0], 0]])


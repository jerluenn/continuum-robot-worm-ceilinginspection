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
        self._integrator_with_boundaries = self._robot_arm_model._create_static_integrator_with_boundaries()
        self._integrator_static = self._robot_arm_model._create_static_integrator()
        self._solver_static_robot, self._integrator_static_step = self._robot1.create_static_solver()
        self._solver_static_robot_position_boundary, self._integrator_static_position_boundary_step = self._robot1.create_static_solver_position_boundary()
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
        self._current_full_states_0 = np.zeros((13+self._num_tendons*2, self._robot_arm_model.get_num_integration_steps()+1))
        self._current_full_states_1 = np.zeros((13+self._num_tendons*2, self._robot_arm_model.get_num_integration_steps()+1))
        self._current_full_states_0[3, :] = 1
        self._time_step = 1e-2
        self._t = 0.0
        self._current_tau = np.zeros(6)
        self._history_states = np.zeros((13+self._num_tendons*2, self._robot_arm_model.get_num_integration_steps()+1, 10000))
        self.wrench_lb = -50
        self.wrench_ub = 50
        self.pos_ub = 5
        self.eta_ub = 1.05
        self.tension_max = 50
        self._differential_kinematics_solver = solver

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

        # Integrate first

        self._integrator_with_boundaries.set('x', self._init_states_0)
        self._integrator_with_boundaries.solve()
        
        self._current_states_0 = self._integrator_with_boundaries.get('x')

        wrench_at_body = -self._robot_arm_model._f_b_pend(self._current_states_0[3:16])

        self._init_states_1 = np.hstack((self._current_states_0[0:7],
                                         np.array(wrench_at_body)[:,0],
                                         self._current_tau[3:6],
                                         np.zeros(3)))
        
        self._integrator_static.set('x', self._init_states_1)
        self._integrator_static.solve()

        self._s_forw_0 = self._integrator_with_boundaries.get('S_forw')
        self._s_forw_1 = self._integrator_static.get('S_forw')

        self._current_states_1 = self._integrator_static.get('x')

        self._dpose_dyu_0 = self._s_forw_0[0:7, 7:13]
        self._dpose_dq_0 = np.hstack((self._s_forw_0[0:7, 13:16], np.zeros((7, 3))))

        self._dpose_dyu_1 = self._s_forw_1[0:7, 7:13]
        self._dpose_dq_1 = np.hstack((np.zeros((7, 3)), self._s_forw_1[0:7, 13:16]))

        self.compute_boundary_Jacobian()

        print("Boundary for arm 1: ", self._robot_arm_model._f_b_pend(self._current_states_0[3:16]))
        print("Boundary for arm 2: ", self._robot_arm_model._f_b(self._current_states_1[3:16]))

        self._db_dyu_pinv_0 = np.linalg.pinv(self._db_dyu_0)      
        self._db_dyu_pinv_1 = np.linalg.pinv(self._db_dyu_1)

        self._B_q_1 = -self._db_dyu_pinv_1@self._db_dq_1
        self._B_q_0 = -self._db_dyu_pinv_0@(self._db_dyu_0 - self._db_dyu_pinv_1@self._db_dq_1)
        # self._J_q_0 = self._dpose_dq_0 + self._dpose_dyu_0@self._B_q_0 + np.linalg.inv(self._s_forw_1[0:7, 0:7])@self._dpose_dyu_1@self._db_dyu_pinv_1@self._db_dq_1
        self._J_q_0 = self._dpose_dq_0 + self._dpose_dyu_0@self._B_q_0
        self._J_q_1 = self._dpose_dq_1 + self._dpose_dyu_1@self._B_q_1 + self._s_forw_1[0:7, 0:7]@self._J_q_0

        # Set lengths!

    def solve_differential_inverse_kinematics(self, task_vel): 

        pass 

    def apply_tension_differential(self, delta_q_tau): 

        self.solve_Jacobians()
        boundary_dot_0 = np.array(self._B_q_0@delta_q_tau*self._time_step)[:, 0]
        boundary_dot_1 = self._B_q_1@delta_q_tau*self._time_step
        pose_dot_1 = np.array(self._J_q_1@delta_q_tau*self._time_step)[:, 0]
        self._current_tau += delta_q_tau*self._time_step
        self._init_wrench += boundary_dot_0
        self._init_wrench_1 += boundary_dot_1
        self._init_states_1[0:7] += pose_dot_1
        self._init_states_0[7:13] = self._init_wrench
        self._init_states_1[7:13] = self._init_wrench_1
        self._init_states_0[13:16] = self._current_tau[0:3]
        self._init_states_1[13:16] = self._current_tau[3:6]
        self._t += self._time_step

    def print_Jacobians(self): 

        pass 

    def print_states(self): 

        pass 

    def solve_full_shape(self): 

        self._current_full_states_0[7:13, 0] = self._init_wrench
        self._current_full_states_0[13: 13+self._num_tendons, 0] = self._current_tau[0:3]

        for i in range(self._robot_arm_model.get_num_integration_steps()):

            x = self._current_full_states_0[:, i]
            self._integrator_static_position_boundary_step.set('x', x)
            self._integrator_static_position_boundary_step.solve()
            self._current_full_states_0[:, i+1] = self._integrator_static_position_boundary_step.get('x')

        self._current_full_states_1[0:7, 0] = x[0:7] 
        self._current_full_states_1[7:13, 0] = np.array(-self._robot_arm_model._f_b_pend(x[3:16]))[:, 0]
        self._current_full_states_1[13: 13+self._num_tendons, 0] = self._current_tau[3:6]

        for i in range(self._robot_arm_model.get_num_integration_steps()):

            x = self._current_full_states_1[:, i]
            self._integrator_static_step.set('x', x)
            self._integrator_static_step.solve()
            self._current_full_states_1[:, i+1] = self._integrator_static_step.get('x')

        self._init_states_0 = self._current_full_states_0[:, 0]
        self._init_states_1 = self._current_full_states_1[:, 0]
        self._init_wrench_1 = self._current_full_states_1[7:13, 0]

    def solve_static(self, init_solution=np.zeros(6)): 

        t = time.time()

        sol = least_squares(self.residuals_func, init_solution, method='lm')
        self._init_wrench = sol.x
        print("Initial wrench: ", sol.x)
        print("Cost: ", sol.cost)

        return sol

    def set_tensions_SS_solver(self, tensions): 

        self._current_tau = tensions

    def residuals_func(self, guess): 

        states_0 = np.hstack((np.zeros(3),
                            np.array([1, 0, 0, 0]),
                            guess,
                            self._current_tau[0:3],
                            np.zeros(3)))

        self._integrator_with_boundaries.set('x', states_0)
        self._integrator_with_boundaries.solve()
        states_0 = self._integrator_with_boundaries.get('x')

        # Do something with states here. 

        wrench_at_body = -self._robot_arm_model._f_b_pend(states_0[3:16])

        states_b = np.hstack((states_0[0:7], 
                              np.array(wrench_at_body)[:,0],
                              self._current_tau[3:6],
                              np.zeros(3)))

        self._integrator_static.set('x', states_b)
        self._integrator_static.solve()
        sol = self._integrator_static.get('x')
        # print('sol: ', sol)

        residuals = self._robot_arm_model._f_b(sol[3:16])
        residuals = np.array(residuals.full())[:, 0]

        return residuals 


    def visualise_robot(self): 

        self.solve_full_shape()

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(self._current_full_states_0[0, :], self._current_full_states_0[1, :], self._current_full_states_0[2, :])
        ax.plot(self._current_full_states_1[0, :], self._current_full_states_1[1, :], self._current_full_states_1[2, :])

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


    def compute_boundary_Jacobian(self):

        self._db_dyu_1 = self._robot_arm_model.f_db_dy(self._current_states_1[3:16])@self._s_forw_1[3:16, 7:13]
        self._db_dq_1 = np.hstack((np.zeros((6, 3)), self._robot_arm_model.f_db_dy(self._current_states_1[3:16])@self._s_forw_1[3:16, 13:16]))

        self._db_dyu_0 = (self._robot_arm_model.f_db_pend_dy(self._current_states_0[0:16])@self._s_forw_0[0:16, 7:13])
        self._db_dq_0 = np.hstack(((self._robot_arm_model.f_db_pend_dy(self._current_states_0[0:16])@self._s_forw_0[0:16, 13:16]), np.zeros((6, 3))))
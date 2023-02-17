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
import osqp
from scipy import sparse

class Quasistatic_Control_Manager: 

    def __init__(self, robot_arm_model): 

        self._robot_arm_model = robot_arm_model
        self._robot1 = Multiple_Shooting_Solver(robot_arm_model)
        self._num_tendons = self._robot_arm_model._tau.shape[0]
        self._integrator_with_boundaries = self._robot_arm_model._create_static_integrator_with_boundaries()
        self._integrator_static_full = self._robot_arm_model._create_static_integrator()
        self._solver_static, self._integrator_static = self._robot1.create_static_solver()
        self._wrench_lb = -50
        self._wrench_ub = 50
        self._tendon_radiuses = self._robot_arm_model._tendon_radiuses_numpy
        self._MAX_ITERATIONS_STATIC = 1000
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
        self._initialise_differential_inverse_solver()

    def set_time_step(self, time_step):

        self._time_step = time_step

    def set_tensions_static_MS_solver(self, tension): 

        lbx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self._wrench_lb*np.ones(6), tension))
        ubx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self._wrench_ub*np.ones(6), tension))        

        self._solver_static.constraints_set(0, 'lbx', lbx_0)
        self._solver_static.constraints_set(0, 'ubx', ubx_0)

    def solve_full_shape(self): 

        self._current_full_states[7:13, 0] = self._init_wrench
        self._current_full_states[13: 13+self._num_tendons, 0] = self._current_tau

        for i in range(self._robot_arm_model.get_num_integration_steps()):

            x = self._current_full_states[:, i]
            self._integrator_static.set('x', x)
            self._integrator_static.solve()
            self._current_full_states[:, i+1] = self._integrator_static.get('x')

    def save_step(self): 

        step_num = int(self._t/self._time_step) - 1
        t, dat = self.get_simulation_data()
        self._history_states[-1, :, step_num] = t
        self._history_states[:-1, :, step_num] = dat 

    def solve_Jacobians(self): 

        self._integrator_with_boundaries.set('x', self._init_sol_boundaries)
        self._integrator_with_boundaries.solve()
        self._s_forw = self._integrator_with_boundaries.get('S_forw')
        self._dpose_dyu = self._s_forw[0:7, 7:13]
        self._dpose_dq = self._s_forw[0:7, 13:16] 
        self._current_states = self._integrator_with_boundaries.get('x')
        R = self.eta_to_Rotation_T_matrix(self._current_states[3:7])
        self.compute_boundary_Jacobian()
        self._db_dyu_pinv = np.linalg.pinv(self._db_dyu)
        self._boundary_conditions = self.compute_boundary_conditions(R, self._current_states[7:10], self._current_states[10:13], self._current_states[13:13+self._num_tendons])
        self._B_q = - self._db_dyu_pinv@self._db_dq
        self._J_q = (self._dpose_dq + self._dpose_dyu@self._B_q)
        self._L_q = self._s_forw[16:19, 13:16] + self._s_forw[16:19, 7:13]@self._B_q 
        self._L_q_pinv = np.linalg.pinv(self._L_q)
        self._J_l = self._J_q@self._L_q_pinv

    def print_Jacobians(self): 

        print('Boundary conditions: ', self._boundary_conditions)
        print("dpose_dyu : ", self._dpose_dyu)
        print("dpose_dq  : ", self._dpose_dq)
        print("db_dyu_pinv: ", self._db_dyu)
        print("db_dq : ", self._db_dq)
        print("J_q: ", self._J_q)
        print("J_l: ", self._J_l)

    def apply_tension_differential(self, delta_q_tau): 

        self.solve_Jacobians()
        boundary_dot = np.array(self._B_q@delta_q_tau*self._time_step)
        self._current_tau += delta_q_tau*self._time_step
        self._init_wrench += boundary_dot[:, 0]
        self._init_sol_boundaries[7:13] = self._init_wrench
        self._init_sol_boundaries[13:13+self._num_tendons] = self._current_tau
        self._lengths_dot = self._L_q@delta_q_tau
        self._t += self._time_step

    def get_simulation_data(self): 

        self.solve_full_shape()

        return self._t, self._current_full_states

    def initialise_static_solver(self, initial_solution): 

        self._solver_static.set(0, 'x', initial_solution)

        subseq_solution = initial_solution

        for i in range(self._robot_arm_model.get_num_integration_steps()): 

            self._integrator_static.set('x', subseq_solution)
            self._integrator_static.solve()
            subseq_solution = self._integrator_static.get('x')
            self._solver_static.set(i+1, 'x', subseq_solution)

        self.INITIALISED_STATIC_SOLVER = 1

    def solve_static(self): 

        t = time.time()
        self._final_sol_viz = np.zeros((3, self._robot_arm_model.get_num_integration_steps()+1))

        if self.INITIALISED_STATIC_SOLVER:  

            for i in range(self._MAX_ITERATIONS_STATIC): 

                self._solver_static.solve()

                if self._solver_static.get_cost() < 1e-7:

                    print("Number of iterations required: ", i+1)
                    print("Total time taken: ", (time.time() - t), 's')
                    print("Time taken per iteration: ", (time.time() - t)/(i+1), "s.")
                    self._init_sol = self._solver_static.get(0, 'x')

                    for k in range(self._robot_arm_model.get_num_integration_steps()+1):

                        self._final_sol_viz[:, k] = self._solver_static.get(k, 'x')[0:3]

                    self._init_sol_boundaries = np.hstack((self._init_sol, 0*np.ones(3)))
                    self._integrator_static_full.set('x', self._init_sol)
                    self._integrator_static_full.solve()
                    self._init_wrench = self._solver_static.get(0, 'x')[7:13]
                    self._current_tau = self._solver_static.get(0, 'x')[13:13+self._num_tendons]
                    print(self._integrator_static_full.get('x'))

                    break

                elif i == self._MAX_ITERATIONS_STATIC - 1 and self._solver_static.get_cost() > 1e-5: 

                    print("DID NOT CONVERGE!")

    def visualise(self): 

        self.solve_full_shape()

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(self._current_full_states[0, :], self._current_full_states[1, :], self._current_full_states[2, :])

        # ax.legend()
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0, 0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def _update(self, i): 

        self._data.set_data(self._history_states[0, :, i], self._history_states[1, :, i])
        self._data.set_3d_properties(self._history_states[2, :, i])

    def animate(self, name): 

        step_num = int(self._t/self._time_step)
        self._history_states = self._history_states[:, :, :step_num]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') 
        animation.writer = animation.writers['ffmpeg']
        # animation.writer.
        plt.ioff()

        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0, 0.2)

        self._data, = ax.plot3D([],[],[])

        ani = animation.FuncAnimation(fig, self._update, frames=range(step_num), interval=self._time_step*1000)
        ani.save(name + '.mp4')

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

    def compute_boundary_conditions(self, R, n, m, tau): 

        R_T = np.transpose(R)
        v = self._e3 + self._Kse_inv@R_T@n

        boundary_conditions = np.zeros(6)

        for i in range(self._num_tendons): 

            tmp = -tau[i]*(R@v)/np.linalg.norm(R@v)
            boundary_conditions[0:3] += tmp
            boundary_conditions[3:6] += self._skew(R@self._tendon_radiuses[i, :])@tmp

        boundary_conditions = np.hstack((n, m)) - boundary_conditions

        return boundary_conditions

    def _skew(self, x): 

        return np.array([[0, -x[2], x[1]],
                        [x[2], 0, -x[0]],
                        [-x[1], x[0], 0]])


    def compute_boundary_Jacobian(self):

        self._db_dyu = self._robot_arm_model.f_db_dy(self._current_states[3:16])@self._s_forw[3:16, 7:13]
        self._db_dq = self._robot_arm_model.f_db_dy(self._current_states[3:16])@self._s_forw[3:16, 13:16]


    def _initialise_differential_inverse_solver(self):

        _lambda = 0.0000001
        self._solver = osqp.OSQP()
        self._qp_A = sparse.csc_matrix(np.vstack((np.eye(3), np.eye(3), np.eye(3))))
        self._qp_P = sparse.csc_matrix(np.zeros((3, 3)))
        self._qp_q = np.zeros(3)
        self._qp_l = np.zeros(9)
        self._qp_u = np.zeros(9)
        self._qp_l_zeros_tau = 1e-5*np.ones(3)
        self._qp_u_max_tau = 20*np.ones(3)
        self._eye_lambda = _lambda*np.eye(3)
        self._solver.setup(P=self._qp_P, q=self._qp_q, A=self._qp_A, l=self._qp_l, u=self._qp_u)

    def DM2Arr(self, dm):
        return np.array(dm.full())

    def solve_differential_inverse_kinematics(self, task_vel): 

        self._qp_q = -2*task_vel@self._J_q[0:2, :]
        self._qp_l[0:3] = (self._qp_l_zeros_tau - self._current_states[13:16])/self._time_step
        self._qp_u[0:3] = (self._qp_u_max_tau - self._current_states[13:16])/self._time_step
        self._qp_l[3:9] = -5
        self._qp_u[3:9] = 5
        self._qp_P.data = self.DM2Arr(np.transpose(self._J_q[0:2, :])@self._J_q[0:2, :] + self._eye_lambda)
        self._solver.update(q=self.DM2Arr(self._qp_q)[0], l=self._qp_l, u=self._qp_u, Px=self._qp_P.todense())
        sol = self._solver.solve()

        return sol.x
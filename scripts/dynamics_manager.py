from casadi import *
import numpy as np

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params

import time

from pyquaternion import Quaternion
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Dynamics_Manager: 

    def __init__(self, robot_arm_model, alpha, time_step): 

        self._robot_arm_model = robot_arm_model
        self.robot1 = Multiple_Shooting_Solver(robot_arm_model)
        self._num_tendons = self._robot_arm_model._tau.shape[0]
        self._solver_dynamic, self._integrator_dynamic = self.robot1.create_dynamic_solver() 
        self._solver_static, self._integrator_static = self.robot1.create_static_solver()
        self._history_terms_i_minus_1 = np.zeros((12, self._robot_arm_model.get_num_integration_steps()+1))
        self._history_terms_i_minus_2 = np.zeros((12, self._robot_arm_model.get_num_integration_steps()+1))
        self._history_terms_i_minus_3 = np.zeros((12, self._robot_arm_model.get_num_integration_steps()+1))
        self._history_terms_i = np.zeros((12, self._robot_arm_model.get_num_integration_steps()+1))
        self._history_R_T = np.zeros((3, 3, self._robot_arm_model.get_num_integration_steps()+1))
        self._history_history = np.zeros((6,self._robot_arm_model.get_num_integration_steps()+1, 3))
        self._e3 = np.zeros((3, self._robot_arm_model.get_num_integration_steps()+1))
        self._e3[2, :] = 1
        self._MAX_ITERATIONS_STATIC = 1000
        self._MAX_ITERATIONS_DYNAMIC = 1000
        self._alpha = alpha
        self._time_step = time_step
        self._wrench_lb = -50
        self._wrench_ub = 50
        self._simulation_counter = 0
        self._Kse = np.array([self._robot_arm_model.get_robot_arm_params().get_Kse()])[0, :, :]
        self._Kbt = np.array([self._robot_arm_model.get_robot_arm_params().get_Kbt()])[0, :, :]
        self._Bse = np.array([self._robot_arm_model.get_robot_arm_params().get_Bse()])[0, :, :]
        self._Bbt = np.array([self._robot_arm_model.get_robot_arm_params().get_Bbt()])[0, :, :]
        self._Kse_inv = np.linalg.inv(self._robot_arm_model.get_robot_arm_params().get_Kse())
        self._Kbt_inv = np.linalg.inv(self._robot_arm_model.get_robot_arm_params().get_Kbt())
        self._sol_history = np.zeros((20+self._robot_arm_model._tau.shape[0], self._robot_arm_model.get_num_integration_steps()+1, 10000))
        self._time_history = 0.
        # self._initialise_animation()
        """TO DO: 
        1. Do update v_u_BDF 
            1a. Need to have a manager for history terms.
            1b. Update history terms."""


        ### For testing only. 

        self._c0 = (1.5 + self._robot_arm_model._robot_arm_params_obj.get_alpha())/(self._robot_arm_model._robot_arm_params_obj.get_time_step()*(1 + self._robot_arm_model._robot_arm_params_obj.get_alpha()))
        self._c1 = -2/self._robot_arm_model._robot_arm_params_obj.get_time_step()
        self._c2 = (0.5 + self._robot_arm_model._robot_arm_params_obj.get_alpha())/(self._robot_arm_model._robot_arm_params_obj.get_time_step()*(1+self._robot_arm_model._robot_arm_params_obj.get_alpha()))
        self._d1 = self._robot_arm_model._robot_arm_params_obj.get_alpha()/(1+self._robot_arm_model._robot_arm_params_obj.get_alpha())

        self._Kse_plus_c0_Bse_inv = np.linalg.inv(self._robot_arm_model.get_robot_arm_params().get_Kse() + self._c0*self._robot_arm_model.get_robot_arm_params().get_Bse())
        self._Kbt_plus_c0_Bbt_inv = np.linalg.inv(self._robot_arm_model.get_robot_arm_params().get_Kbt() + self._c0*self._robot_arm_model.get_robot_arm_params().get_Bbt())
        self._Kse_e3 = self._robot_arm_model.get_robot_arm_params().get_Kse()@self._e3

    def _initialise_animation(self): 

        self.fig = plt.figure() 
        self.ax = self.fig.add_subplots(projection='3d')
        self.xdata, self.ydata, self.zdata = [], [], []
        self.ln, = self.ax.plot([], [], 'ro')
        self._initialise_animation_func()

    def _initialise_animation_func(self): 

        self.ax.set_xlim(0, 2*np.pi)
        self.ax.set_ylim(-1, 1)
        self.ax.set_ylim(-1, 1)
        
        return self.ln,
    
    def update(self, frame): 

        self.ln.set_data()

    def update_v_u_BDF(self): 

        # np.einsum('ijk, ik -> ik', R_hist, v_hist)

        self._history_terms_i_minus_1[0:3, :] = self._Kse_plus_c0_Bse_inv @ (np.einsum('ijk, ik -> jk', self._history_R_T, self._history_terms_i_minus_1[0:3, :]) + self._Kse_e3 - self._Kse@self._history_history[0:3, :, 0])
        # self._history_terms_i_minus_2[0:3, :] = self._Kse_plus_c0_Bse_inv @ (np.einsum('ijk, ik -> ik', self._history_R_T, self._history_terms_i_minus_2[0:3, :]) + self._Kse_e3 - self._Kse@self._history_history[0:3, :, 1])
        # self._history_terms_i_minus_3[0:3, :] = self._Kse_plus_c0_Bse_inv @ (np.einsum('ijk, ik -> ik', self._history_R_T, self._history_terms_i_minus_3[0:3, :]) + self._Kse_e3 - self._Kse@self._history_history[0:3, :, 2])

        self._history_terms_i_minus_1[3:6, :] = self._Kbt_plus_c0_Bbt_inv @ (np.einsum('ijk, ik -> jk', self._history_R_T, self._history_terms_i_minus_1[3:6, :]) - self._Bbt@self._history_history[3:6,:, 0])
        # self._history_terms_i_minus_2[3:6, :] = self._Kbt_plus_c0_Bbt_inv @ (np.einsum('ijk, ik -> ik', self._history_R_T, self._history_terms_i_minus_2[3:6, :])+ self._Bbt@self._history_history[3:6,:, 1])
        # self._history_terms_i_minus_3[3:6, :] = self._Kbt_plus_c0_Bbt_inv @ (np.einsum('ijk, ik -> ik', self._history_R_T, self._history_terms_i_minus_3[3:6, :])+ self._Bbt@self._history_history[3:6,:, 2])

        # self._history_terms_i[6:12] = (self._c0*self._d1+self._c1)*(self._history_terms_i_minus_1[6:12]) \
        #     + (self._c1*self._d1 + self._c2)*(self._history_terms_i_minus_2[6:12]) \
        #     + self._c2*self._d1*(self._history_terms_i_minus_3[6:12])

        self._history_terms_i = (self._c0*self._d1+self._c1)*(self._history_terms_i_minus_1) \
            + (self._c1*self._d1 + self._c2)*(self._history_terms_i_minus_2) \
            + self._c2*self._d1*(self._history_terms_i_minus_3)

        """ Get v and u here."""    

        self._history_history[:, :, 2] = self._history_history[:, :, 1]
        self._history_history[:, :, 1] = self._history_history[:, :, 0]
        self._history_history[:, :, 0] = self._history_terms_i[0:6, :]
        

        for i in range(self._robot_arm_model.get_num_integration_steps()):


            self._solver_dynamic.set(i, 'p', self._history_terms_i[:, i])



    def update_v_u_BDF_static(self): 


        self._history_terms_i_minus_1[0:3, :] = self._e3 + self._Kse_inv@np.einsum('ijk, ik -> jk', self._history_R_T, self._history_terms_i_minus_1[0:3, :])
        self._history_terms_i_minus_2[0:3, :] = self._e3 + self._Kse_inv@np.einsum('ijk, ik -> jk', self._history_R_T, self._history_terms_i_minus_2[0:3, :])
        self._history_terms_i_minus_3[0:3, :] = self._e3 + self._Kse_inv@np.einsum('ijk, ik -> jk', self._history_R_T, self._history_terms_i_minus_3[0:3, :])

        self._history_terms_i_minus_1[3:6, :] = self._Kbt_inv@np.einsum('ijk, ik -> jk', self._history_R_T, self._history_terms_i_minus_1[3:6, :])
        self._history_terms_i_minus_2[3:6, :] = self._Kbt_inv@np.einsum('ijk, ik -> jk', self._history_R_T, self._history_terms_i_minus_2[3:6, :])
        self._history_terms_i_minus_3[3:6, :] = self._Kbt_inv@np.einsum('ijk, ik -> jk', self._history_R_T, self._history_terms_i_minus_3[3:6, :])

        self._history_terms_i = (self._c0*self._d1+self._c1)*(self._history_terms_i_minus_1) \
            + (self._c1*self._d1 + self._c2)*(self._history_terms_i_minus_2) \
            + self._c2*self._d1*(self._history_terms_i_minus_3)

        for i in range(self._robot_arm_model.get_num_integration_steps()):

            self._solver_dynamic.set(i, 'p', self._history_terms_i[:, i])

    def set_tensions_static(self, tension): 

        lbx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self._wrench_lb*np.ones(6), tension))
        ubx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self._wrench_ub*np.ones(6), tension))        

        self._solver_static.constraints_set(0, 'lbx', lbx_0)
        self._solver_static.constraints_set(0, 'ubx', ubx_0)

    def set_tensions_dynamic(self, tension): 

        lbx_0 = np.hstack((self._solver_dynamic.get(0, 'x')[0:7], self._wrench_lb*np.ones(6), self._wrench_lb*np.zeros(6), tension))
        ubx_0 = np.hstack((self._solver_dynamic.get(0, 'x')[0:7], self._wrench_ub*np.ones(6), self._wrench_ub*np.zeros(6), tension))        

        self._solver_dynamic.constraints_set(0, 'lbx', lbx_0)
        self._solver_dynamic.constraints_set(0, 'ubx', ubx_0)     

    def initialise_static_solver(self, initial_solution): 

        self._solver_static.set(0, 'x', initial_solution)

        subseq_solution = initial_solution

        for i in range(self._robot_arm_model.get_num_integration_steps()): 

            self._integrator_static.set('x', subseq_solution)
            self._integrator_static.solve()
            subseq_solution = self._integrator_static.get('x')
            self._solver_static.set(i+1, 'x', subseq_solution)

        self.INITIALISED_STATIC_SOLVER = 1


    def _convert_m_n_to_u_v(self, m, n, eta, step): 

        R_T = self.eta_to_Rotation_T_matrix(eta) 

        u = inv(self._robot_arm_model._robot_arm_params_obj.get_Kbt() + self._c0*self._robot_arm_model._robot_arm_params_obj.get_Bbt())@(np.transpose(R_T)@m - self._robot_arm_model._robot_arm_params_obj.get_Bbt()@self._history_terms_i[3:6, step])
        v = inv(self._robot_arm_model._robot_arm_params_obj.get_Kse() + self._c0*self._robot_arm_model._robot_arm_params_obj.get_Bse())@(np.transpose(R_T)@n + self._robot_arm_model._robot_arm_params_obj.get_Kse()@SX([0, 0, 1]) - self._robot_arm_model._robot_arm_params_obj.get_Bse()@self._history_terms_i[3:6, step])

        u = self._Kbt_inv@R_T@m
        v = np.array([0, 0, 1]) + self._Kse_inv@R_T@n


        return u, v 

    def _initialise_dynamic_solver(self): 

        x = np.hstack((self._solver_static.get(0, 'x')[0:13], np.zeros(6), self._solver_static.get(0, 'x')[13:16]))
        u, v = self._convert_m_n_to_u_v(x[10:13], x[7:10], x[3:7], 0)
        print('c0*u + u_hist = ', self._c0*u + self._history_terms_i[3:6, 0])
        print('c0*v + v_hist = ', self._c0*v + self._history_terms_i[0:3, 0]) 

        self._integrator_dynamic.set('p', self._history_terms_i[:, 0])

        print(x)
        self._integrator_dynamic.set('x', x)

        for i in range(self._robot_arm_model.get_num_integration_steps()): 

            self._integrator_dynamic.set('p', self._history_terms_i[:, i+1])
            self._integrator_dynamic.solve()
            x = self._integrator_dynamic.get('x')
            self._integrator_dynamic.set('x', x)
            solution = np.hstack((self._solver_static.get(i, 'x')[0:13], np.zeros(6), self._solver_static.get(i, 'x')[-3:]))
            self._solver_dynamic.set(i, 'x', solution)
            u, v = self._convert_m_n_to_u_v(x[10:13], x[7:10], x[3:7], i+1)
            print('c0*u + u_hist = ', self._c0*u + self._history_terms_i[3:6, i+1])
            print('c0*v + v_hist = ', self._c0*v + self._history_terms_i[0:3, i+1]) 

        print(self._integrator_dynamic.get('x'))

        self.INITIALISED_DYNAMIC_SOLVER = 1

    def solve_for_static(self):

        t = time.time()
        self._final_sol_viz = np.zeros((3, self._robot_arm_model.get_num_integration_steps()+1))

        if self.INITIALISED_STATIC_SOLVER:  

            for i in range(self._MAX_ITERATIONS_STATIC): 

                self._solver_static.solve()

                if self._solver_static.get_cost() < 1e-5:

                    print("Number of iterations required: ", i+1)
                    print("Total time taken: ", (time.time() - t), 's')
                    print("Time taken per iteration: ", (time.time() - t)/(i+1), "s.")
                    self._init_sol = self._solver_static.get(0, 'x')

                    for k in range(self._robot_arm_model.get_num_integration_steps()+1):

                        self._history_terms_i_minus_1[0:6, k] = self._solver_static.get(k, 'x')[7:13]
                        self._history_terms_i_minus_2[0:6, k] = self._solver_static.get(k, 'x')[7:13]
                        self._history_terms_i_minus_3[0:6, k] = self._solver_static.get(k, 'x')[7:13]
                        self._history_R_T[:, :, k] = self.eta_to_Rotation_T_matrix(self._solver_static.get(k, 'x')[3:7])
                        self._final_sol_viz[:, k] = self._solver_static.get(k, 'x')[0:3]

                    self.update_v_u_BDF_static()
                    self._initialise_dynamic_solver()
                    break

                elif i == self._MAX_ITERATIONS_STATIC - 1 and self._solver_static.get_cost() > 1e-5: 

                    print("DID NOT CONVERGE!")

    def solve_for_dynamic(self): 

        t = time.time()

        for i in range(self._MAX_ITERATIONS_DYNAMIC): 
 
            self._solver_dynamic.solve()

            if self._solver_dynamic.get_cost() < 1e-6 :
            # if self._solver_dynamic.get_cost() < 1e-6  and self._solver_dynamic.get_residuals()[1] < 1e-3:

                for k in range(self._robot_arm_model.get_num_integration_steps()+1):

                    self._history_terms_i_minus_3[0:12, k] = self._history_terms_i_minus_2[0:12, k]
                    self._history_terms_i_minus_2[0:12, k] = self._history_terms_i_minus_1[0:12, k]
                    self._history_terms_i_minus_1[0:12, k] = self._solver_dynamic.get(k, 'x')[7:19]
                    self._history_R_T[:, :, k] = self.eta_to_Rotation_T_matrix(self._solver_static.get(k, 'x')[3:7])
                    self._final_sol_viz[:, k] = self._solver_dynamic.get(k, 'x')[0:3]
                    self._sol_history[0:19+self._num_tendons, k, self._simulation_counter] = self._solver_dynamic.get(k, 'x')
                    self._sol_history[19+self._num_tendons, k, self._simulation_counter] = self._time_history
                    
                print("Position: ",self._final_sol_viz[:, -1])
                print("Velocites: ", self._sol_history[13:19, k, self._simulation_counter])
                print("Number of iterations required: ", i+1)
                print("Total time taken: ", (time.time() - t), 's')
                print("Time taken per iteration: ", (time.time() - t)/(i+1), "s.")
                self._init_sol = self._solver_dynamic.get(0, 'x')
                
                self._time_history += self._time_step
                self._simulation_counter += 1
                self.update_v_u_BDF()

                break

            elif i == self._MAX_ITERATIONS_DYNAMIC - 1 and self._solver_dynamic.get_cost() > 1e-5: 

                print("DID NOT CONVERGE!")
        
    def visualise(self): 

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(self._final_sol_viz[0, :], self._final_sol_viz[1, :], self._final_sol_viz[2, :])

        # ax.legend()
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0, 0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def get_solution_history(self): 

        return self._sol_history

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

        return np.transpose(R)
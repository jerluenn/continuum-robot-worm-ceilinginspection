from casadi import *
import numpy as np

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params

import time

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Dynamics_Manager: 

    def __init__(self, robot_arm_model, alpha, time_step): 

        self._robot_arm_model = robot_arm_model
        self.robot1 = Multiple_Shooting_Solver(robot_arm_model)
        self._num_tendons = self._robot_arm_model._tau.shape[0]
        self._solver_dynamic, self._integrator_dynamic = self.robot1.create_dynamic_solver() 
        self._solver_static, self._integrator_static = self.robot1.create_static_solver()
        # self._history_terms_i_minus_1 = np.zeros((13 + self._num_tendons, self._robot_arm_model.get_num_integration_steps()))
        # self._history_terms_i_minus_2 = np.zeros((13 + self._num_tendons, self._robot_arm_model.get_num_integration_steps()))
        self._history_terms_i = np.zeros((16, self._robot_arm_model.get_num_integration_steps(), 3))
        self._MAX_ITERATIONS_STATIC = 1000
        self._MAX_ITERATIONS_DYNAMIC = 1000
        self._alpha = alpha
        self._time_step = time_step
        self._solve_BDF_coefficients()
        self._wrench_lb = -50
        self._wrench_ub = 50
        self._simulation_counter = 1
        """TO DO: 
        1. Do update v_u_BDF 
            1a. Need to have a manager for history terms.
            1b. Update history terms."""

    def _solve_BDF_coefficients(self):

        self._c0 = (1.5+self._alpha)/(self._time_step*(1+self._alpha))
        self._c1 = -2/self._time_step
        self._c2 = (0.5 + self._alpha)/(self._time_step*(1+self._alpha))
        self._d1 = self._alpha/(1+self._alpha)

    def update_v_u_BDF(self): 

        """Setting matrix must be 16xNUM_INTEGRATION_STEPS:
        Works in eta, n, m, q, om."""

        i_minus_1 = self._index_history%3
        i_minus_2 = (self._index_history - 1)%3
        i_minus_3 = (self._index_history - 2)%3

        history_terms_BDF = self._d1*self._c2*self._history_terms_i[:, :, i_minus_3] + (self._c0*self._d1 + self._c1)*self._history_terms_i[:, :, i_minus_1] + (self._c1*self._d1 + self._c2)*self._history_terms_i[:, :, i_minus_2]

        for i in range(self._robot_arm_model.get_num_integration_steps()):

            self._solver_dynamic.set(i, 'p', history_terms_BDF[:, i])


    def update_v_u_BDF_static(self): 

        """Setting matrix must be 16xNUM_INTEGRATION_STEPS:
        Works in eta, n, m, q, om."""

        self._history_terms_i = (self._c1+self._c2)*self._history_terms_i

        for i in range(self._robot_arm_model.get_num_integration_steps()):

            self._solver_dynamic.set(i, 'p', self._history_terms_i[:, i, 0])

    def set_tensions_static(self, tension): 

        lbx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self._wrench_lb*np.ones(6), tension))
        ubx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self._wrench_ub*np.ones(6), tension))        

        self._solver_static.constraints_set(0, 'lbx', lbx_0)
        self._solver_static.constraints_set(0, 'ubx', ubx_0)

    def set_tensions_dynamic(self, tension): 

        lbx_0 = np.hstack((self._solver_dynamic.get(0, 'x')[0:7], self._wrench_lb*np.ones(6), self._wrench_lb*np.ones(6), tension))
        ubx_0 = np.hstack((self._solver_dynamic.get(0, 'x')[0:7], self._wrench_ub*np.ones(6), self._wrench_ub*np.ones(6), tension))        

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
    
    def _initialise_dynamic_solver(self): 

        for i in range(self._robot_arm_model.get_num_integration_steps()+1): 

            solution = np.hstack((self._solver_static.get(i, 'x')[0:13], np.zeros(6), self._solver_static.get(i, 'x')[-3:]))
            self._solver_dynamic.set(i, 'x', solution)

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

                    for k in range(self._robot_arm_model.get_num_integration_steps()):

                        self._history_terms_i[0:10, k, 0] = self._solver_static.get(k, 'x')[3:13]
                        self._final_sol_viz[:, k+1] = self._solver_static.get(k+1, 'x')[0:3]

                    self.update_v_u_BDF_static()
                    self._initialise_dynamic_solver()
                    break

                elif i == self._MAX_ITERATIONS_STATIC - 1 and self._solver_static.get_cost() > 1e-5: 

                    print("DID NOT CONVERGE!")

    def solve_for_dynamic(self): 

        t = time.time()

        self._index_history = (self._simulation_counter)%3

        for i in range(self._MAX_ITERATIONS_DYNAMIC): 
 
            self._solver_dynamic.solve()

            if self._solver_dynamic.get_cost() < 1e-5:

                print("Number of iterations required: ", i+1)
                print("Total time taken: ", (time.time() - t), 's')
                print("Time taken per iteration: ", (time.time() - t)/(i+1), "s.")
                self._init_sol = self._solver_dynamic.get(0, 'x')

                for k in range(self._robot_arm_model.get_num_integration_steps()):

                    self._history_terms_i[:, k, self._index_history] = self._solver_dynamic.get(k, 'x')[3:19]
                    self._final_sol_viz[:, k+1] = self._solver_dynamic.get(k+1, 'x')[0:3]

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
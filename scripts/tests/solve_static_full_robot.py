import sys
import numpy as np
import time
from casadi import *
from pyquaternion import Quaternion

sys.path.insert(0, "..")

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params
from quasistatic_control_manager_full_robot import Quasistatic_Control_Manager_Full_Robot
from linear_mpc import Linear_MPC

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tendon_radiuses_list = [[0.0175*1.4, 0, 0], [-0.00875*1.4, 0.0151554*1.4, 0], [-0.00875*1.4, -0.0151554*1.4, 0]]
tendon_radiuses = SX(tendon_radiuses_list)
robot_arm_1 = Robot_Arm_Params(0.15, 0.05, -0.5, "1", 0.3)
robot_arm_1.from_solid_rod(0.0005, 60e9, 200e9, 8000)
robot_arm_1.set_gravity_vector('z')
C = np.diag([0.000, 0.000, 0.000])
Bbt = np.diag([1e-4, 1e-4, 1e-4])
Bse = Bbt
# Bse = np.zeros((3,3))
# Bbt = np.zeros((3,3))
robot_arm_1.set_damping_coefficient(C)
robot_arm_1.set_damping_factor(Bbt, Bse)
robot_arm_1.set_tendon_radiuses(tendon_radiuses_list)
robot_arm_model_1 = Robot_Arm_Model(robot_arm_1)

Q_w_p = 1000e3*np.eye(2)
Q_w_t = 1e-1*np.eye(3)
n_states = 2
n_tendons = 3
name = 'single_controller'
R_mat = 1e-2*np.eye(3)
Tf = 0.01
q_max = 20
q_dot_max = 5

diff_inv = Linear_MPC(Q_w_p, Q_w_t, n_states, n_tendons,q_dot_max, q_max, name, R_mat, Tf)
diff_inv_solver, _ = diff_inv.create_inverse_differential_kinematics()

quasi_sim_manager = Quasistatic_Control_Manager_Full_Robot(robot_arm_model_1, diff_inv_solver)

tension = np.array([1.0, 0.0, 0, 0, 0, 0])
initial_guess = np.zeros(6)
quasi_sim_manager.set_tensions_SS_solver(tension)
quasi_sim_manager.solve_static(initial_guess)
quasi_sim_manager.visualise_robot()

iterations = 100

for i in range(iterations): 

    quasi_sim_manager.apply_tension_differential(np.array([1, 0, 0, 0, 0, 0]))
    
quasi_sim_manager.visualise_robot()
print("Robot states at 0: ", quasi_sim_manager._current_states_0)
print("Robot states at 1: ", quasi_sim_manager._current_states_1)
import sys
import numpy as np
import time
from casadi import *
from pyquaternion import Quaternion

sys.path.insert(0, "..")

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params
from quasistatic_control_manager import Quasistatic_Control_Manager

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### SETTING UP SOLVER ###

tendon_radiuses_list = [[0.0175, 0, 0], [-0.00875, 0.0151554, 0], [-0.00875, -0.0151554, 0]]
tendon_radiuses = SX(tendon_radiuses_list)
robot_arm_1 = Robot_Arm_Params(0.15, 0.05, -0.5, "1")
robot_arm_1.from_solid_rod(0.0005, 100e9, 200e9, 8000)
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

initial_solution = np.zeros(19)
initial_solution[3] = 1

init_sol = np.zeros(16)
init_sol[3] = 1

# integrator = robot_arm_model_1._create_static_integrator_with_boundaries()

quasi_sim_manager = Quasistatic_Control_Manager(robot_arm_model_1)
quasi_sim_manager.initialise_static_solver(init_sol)

quasi_sim_manager.initialise_static_solver(init_sol)

quasi_sim_manager.set_tensions_static_MS_solver([0.0, 0.0, 0])
quasi_sim_manager.solve_static()

t0 = time.time()

N = 500
# quasi_sim_manager.set_time_step(1e-3)

for i in range(N): 

    quasi_sim_manager.apply_tension_differential(np.array([0.0, 0.3, 0.0]))
    # quasi_sim_manager.save_step()

for i in range(N): 

    quasi_sim_manager.apply_tension_differential(np.array([0.4, 0.1, 0.0]))
    # quasi_sim_manager.save_step()
    # quasi_sim_manager.apply_length_differential(np.array([0.0001, 0.0, 0.0]))
    
print(quasi_sim_manager.get_simulation_data()[1][13:, :])
print("----------------------------------------")
print(f"Time taken: {(time.time() - t0)/N}")

# quasi_sim_manager.print_Jacobians()
quasi_sim_manager.visualise()

# quasi_sim_manager.animate('test')


import sys
import numpy as np
import time
from casadi import *
from pyquaternion import Quaternion

sys.path.insert(0, "..")

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params
from dynamics_manager import Dynamics_Manager

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NUM_ITERATIONS = 1000

### SETTING UP SOLVER ###

tendon_radiuses = SX([[-0.01, 0.01, 0], [-0.01, -0.01, 0], [0.015, 0, 0]])
robot_arm_1 = Robot_Arm_Params(0.15, 0.05, -0.5, "1")
robot_arm_1.from_solid_rod(0.0005, 100e9, 200e9, 8000)
C = np.diag([0.000, 0.000, 0.000])
Bbt = np.diag([1e-4, 1e-4, 1e-4])
Bse = Bbt
# Bse = np.zeros((3,3))
# Bbt = np.zeros((3,3))
robot_arm_1.set_damping_coefficient(C)
robot_arm_1.set_damping_factor(Bbt, Bse)
robot_arm_1.set_tendon_radiuses(tendon_radiuses)
robot_arm_model_1 = Robot_Arm_Model(robot_arm_1)

initial_solution = np.zeros(16)
initial_solution[3] = 1

### SOLVING ###

sim_manager = Dynamics_Manager(robot_arm_model_1, -0.5, 0.01)
sim_manager.initialise_static_solver(initial_solution)
sim_manager.set_tensions_static([0.0, 0.0, 0])
sim_manager.solve_for_static()
sim_manager.visualise()

sim_manager.set_tensions_dynamic([0.0, 0.0, 0])
sim_manager.solve_for_dynamic()

tension = np.array([0.0, 0.0, 0.0])

for i in range(100): 

    tension[0] += 0.03
    tension[2] += 0.04
    sim_manager.set_tensions_dynamic(tension)
    sim_manager.solve_for_dynamic()

sim_manager.solve_for_dynamic()
sim_manager.visualise()

# pass


import sys
import numpy as np

sys.path.insert(0, "..")

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params

NUM_ITERATIONS = 100

robot_arm_1 = Robot_Arm_Params(0.15, 0.03, -0.5, "1")
robot_arm_1.from_solid_rod(0.001, 100e9, 200e9, 8000)
C = np.diag([0.03, 0.03, 0.03])
Bbt = np.diag([1e-6, 1e-6, 1e-6])
Bse = Bbt
robot_arm_1.set_damping_coefficient(C)
robot_arm_1.set_damping_factor(Bbt, Bse)

robot_arm_model_1 = Robot_Arm_Model(robot_arm_1)
d1 = Multiple_Shooting_Solver(robot_arm_model_1)
solver, integrator = d1.create_static_solver()

yref = np.zeros(16)

solver.cost_set(robot_arm_model_1.get_num_integration_steps(), 'yref', yref)

initial_solution = np.zeros(16)
initial_solution[3] = 1
# initial_solution[7] = 3.69828287e-02
# initial_solution[11] = 0.0027
initial_solution[7] = 0.1
initial_solution[11] = 0.6
solver.set(0, 'x', initial_solution)

subseq_solution = initial_solution

for i in range(robot_arm_model_1.get_num_integration_steps()): 

    integrator.set('x', subseq_solution)
    integrator.solve()
    subseq_solution = integrator.get('x')
    solver.set(i+1, 'x', subseq_solution)

a = 1

wrench_lb = -5
wrench_ub = 5

lb = np.array([-5, -5, -5, -1.05, -1.05, -1.05, -1.05, #pose at start.
            wrench_lb , wrench_lb , wrench_lb , wrench_lb , wrench_lb , wrench_lb, 
            0, 0, 0])

ub = np.array([5, 5, 5, 1.05, 1.05, 1.05, 1.05, #pose at start.
            wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub, 
            0, 0, 0])

for i in range(NUM_ITERATIONS): 

    solver.solve()
    print(solver.get_residuals())
    print(solver.get_cost())


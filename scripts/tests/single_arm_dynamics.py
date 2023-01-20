import sys
import numpy as np
import time

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
initial_solution[7] = 4
initial_solution[11] = -4
solver.set(0, 'x', initial_solution)

subseq_solution = initial_solution

for i in range(robot_arm_model_1.get_num_integration_steps()): 

    integrator.set('x', subseq_solution)
    integrator.solve()
    subseq_solution = integrator.get('x')
    solver.set(i+1, 'x', subseq_solution)

t0 = time.time()

for i in range(NUM_ITERATIONS): 

    solver.solve()

    if solver.get_cost() < 1e-9:

        break

    NUM_ITERATIONS_TAKEN = i

print(f"NUM_ITERATIONS_TAKEN to converge: ", NUM_ITERATIONS_TAKEN)
print(f"Time taken/step: {(time.time() - t0)/NUM_ITERATIONS_TAKEN}")

print("Final sol: ",solver.get(10, 'x'))
print("Init sol: ", solver.get(0, 'x'))


import sys
import numpy as np
import time
from casadi import *

sys.path.insert(0, "..")

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NUM_ITERATIONS = 1000

tendon_radiuses = SX([[-0.01, 0.01, 0], [-0.01, -0.01, 0], [0.015, 0, 0]])
robot_arm_1 = Robot_Arm_Params(0.15, 0.03, -0.5, "1")
robot_arm_1.from_solid_rod(0.0005, 100e9, 200e9, 8000)
C = np.diag([0.03, 0.03, 0.03])
Bbt = np.diag([1e-6, 1e-6, 1e-6])
Bse = Bbt
robot_arm_1.set_damping_coefficient(C)
robot_arm_1.set_damping_factor(Bbt, Bse)
robot_arm_1.set_tendon_radiuses(tendon_radiuses)


robot_arm_model_1 = Robot_Arm_Model(robot_arm_1)
d1 = Multiple_Shooting_Solver(robot_arm_model_1)
solver, integrator = d1.create_static_solver()

# yref = np.zeros(22)
yref = np.zeros(6)

solver.cost_set(robot_arm_model_1.get_num_integration_steps(), 'yref', yref)

initial_solution = np.zeros(16)
initial_solution[3] = 1
initial_solution[9] = -6

solver.set(0, 'x', initial_solution)

subseq_solution = initial_solution

d1.set_tensions(solver, np.array([0, 5, 0]))

for i in range(robot_arm_model_1.get_num_integration_steps()): 

    integrator.set('x', subseq_solution)
    integrator.solve()
    subseq_solution = integrator.get('x')
    solver.set(i+1, 'x', subseq_solution)

t0 = time.time()

for i in range(NUM_ITERATIONS): 

    solver.solve()

    if solver.get_cost() < 1e-4:

        break

    NUM_ITERATIONS_TAKEN = i+1


final_sol = np.zeros((robot_arm_model_1.get_num_integration_steps()+1, 3))

for i in range(robot_arm_model_1.get_num_integration_steps()): 

    final_sol[i+1, 0:3] = solver.get(i+1, 'x')[0:3]

print(f"NUM_ITERATIONS_TAKEN to converge: ", NUM_ITERATIONS_TAKEN)
print(f"Time taken/step: {(time.time() - t0)/NUM_ITERATIONS_TAKEN}")
print(f"Time taken: {(time.time() - t0)}")
print(f"Cost: {solver.get_cost()}")
print("Final sol: ",solver.get(10, 'x'))
print("Init sol: ", solver.get(0, 'x'))

ax = plt.figure().add_subplot(projection='3d')
ax.plot(final_sol[:, 0], final_sol[:, 1], final_sol[:, 2])

ax.legend()
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-0.2, 0.2)
ax.set_zlim(0, 0.2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()




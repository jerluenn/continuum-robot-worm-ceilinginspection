from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *

import numpy as np
from generate_robot_arm_model import Robot_Arm_Model

import time

from matplotlib import pyplot as plt

class Multiple_Shooting_Solver:

    def __init__(self, robot_arm_model): 
    
        self._robot_arm_model = robot_arm_model
        self._boundary_length = self._robot_arm_model.get_boundary_length()
        self._integration_steps = self._robot_arm_model.get_num_integration_steps()
        self.create_static_solver()

    def create_static_solver(self):

        self.ocp = AcadosOcp()
        self.ocp.model = self._robot_arm_model.get_static_robot_arm_model()
        nx = self.ocp.model.x.size()[0]
        nu = self.ocp.model.u.size()[0]
        ny = nx + nu

        self.ocp.dims.N = self._integration_steps
        # self.ocp.cost.cost_type_0 = 'LINEAR_LS'
        # self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        self.ocp.cost.W_e = np.identity(nx)
        self.ocp.cost.W = np.zeros((ny, ny))
        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx_e = np.zeros((nx, nx))
        self.ocp.cost.Vx_e[7:13, 7:13] = np.identity(6)
        self.ocp.cost.yref  = np.zeros((ny, ))
        self.ocp.cost.yref_e = np.zeros((nx))
        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.solver_options.qp_solver_iter_max = 400
        # self.ocp.solver_options.sim_method_num_steps = self.integration_steps
        self.ocp.solver_options.qp_solver_warm_start = 2

        self.ocp.solver_options.levenberg_marquardt = 1.0

        # self.ocp.solver_options.levenberg_marquardt = 1.0

        # self.ocp.solver_options.levenberg_marquardt = 1.0
        self.ocp.solver_options.regularize_method = 'CONVEXIFY'

        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # 
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        self.ocp.solver_options.tf = self._boundary_length

        wrench_lb = -5
        wrench_ub = 5

        self.ocp.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        self.ocp.constraints.lbx_0 = np.array([0, 0, 0, 1, 0, 0, 0, #pose at start.
            wrench_lb , wrench_lb , wrench_lb , wrench_lb , wrench_lb , wrench_lb,
            0, 0, 0])  # tension, alpha, kappa, curvature

        self.ocp.constraints.ubx_0 = np.array([0, 0, 0, 1, 0, 0, 0, #pose at start.
            wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub,
            0, 0, 0]) 

        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        self.ocp.constraints.lbx = np.array([-5, -5, -5, -1.05, -1.05, -1.05, -1.05, #pose at start.
            wrench_lb , wrench_lb , wrench_lb , wrench_lb , wrench_lb , wrench_lb, 
            0, 0, 0])

        self.ocp.constraints.ubx = np.array([5, 5, 5, 1.05, 1.05, 1.05, 1.05, #pose at start.
            wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub , wrench_ub, 
            0, 0, 0])

        self.ocp.constraints.ubu = np.array([0]) 
        self.ocp.constraints.lbu = np.array([0]) 
        self.ocp.constraints.idxbu = np.array([0])

        self.ocp.solver_options.nlp_solver_max_iter = 1

        # AcadosOcpSolver.generate(self.ocp, json_file=f'{self.ocp.model.name}.json')
        # AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
        
        # solver = AcadosOcpSolver.create_cython_solver(json_file=f'{self.ocp.model.name}.json')
        solver = AcadosOcpSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')
        integrator = AcadosSimSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')

        return solver, integrator
import sys
import os
import shutil

from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *

import numpy as np
from generate_robot_arm_parameters import Robot_Arm_Params

class Robot_Arm_Model: 

    def __init__(self, robot_arm_params): 

        self._dir_name = 'c_generated_code_' 
        self._robot_arm_params_obj = robot_arm_params
        self._integration_steps = 10
        self._integration_stages = 4
        self._build_robot_model()


    def _build_robot_model(self): 

        self._mass_distribution = self._robot_arm_params_obj.get_mass_distribution()
        self._Kse = self._robot_arm_params_obj.get_Kse()
        self._Kbt = self._robot_arm_params_obj.get_Kbt()
        self._boundary_length = self._robot_arm_params_obj.get_arm_length()
        self._id = self._robot_arm_params_obj.get_id()
        self._initialise_states()
        self._create_static_integrator()

    def _initialise_states(self):

        # Initialise all ODE states.

        self._p = SX.sym('p', 3)
        self._eta = SX.sym('self._eta', 4) 
        self._n = SX.sym('n', 3)
        self._m = SX.sym('m', 3)
        self._tau = SX.sym('tau', 3)

        self._p_d = SX.sym('p_dot', 3)
        self._eta_d = SX.sym('eta_dot', 4)
        self._n_d = SX.sym('n_dot', 3)
        self._m_d = SX.sym('m_dot', 3)
        self._tau_d = SX.sym('tau_dot', 3)
        self._alpha_d = SX.sym('alpha_dot', 1)

        # Initialise constants

        self._g = SX([9.81, 0, 0])
        self._f_ext = self._mass_distribution * self._g
        self._Kappa = SX.sym('Kappa', 1)

        # Setting R 

        self._R = SX(3,3)
        self._R[0,0] = 2*(self._eta[0]**2 + self._eta[1]**2) - 1
        self._R[0,1] = 2*(self._eta[1]*self._eta[2] - self._eta[0]*self._eta[3])
        self._R[0,2] = 2*(self._eta[1]*self._eta[3] + self._eta[0]*self._eta[2])
        self._R[1,0] = 2*(self._eta[1]*self._eta[2] + self._eta[0]*self._eta[3])
        self._R[1,1] = 2*(self._eta[0]**2 + self._eta[2]**2) - 1
        self._R[1,2] = 2*(self._eta[2]*self._eta[3] - self._eta[0]*self._eta[1])
        self._R[2,0] = 2*(self._eta[1]*self._eta[3] - self._eta[0]*self._eta[2])
        self._R[2,1] = 2*(self._eta[2]*self._eta[3] + self._eta[0]*self._eta[1])
        self._R[2,2] = 2*(self._eta[0]**2 + self._eta[3]**2) - 1

        # Intermediate states

        self._u = inv(self._Kbt)@transpose(reshape(self._R, 3, 3))@self._m
        self._v = inv(self._Kse)@transpose(reshape(self._R, 3, 3))@self._n + SX([0, 0, 1])
        self._k = 0.1


    def _create_static_integrator(self):

        model_name = self._dir_name + 'static_robot_arm' + self._id 

        c = self._k*(1-transpose(self._eta)@self._eta)

        u = SX.sym('u')

        p_dot = reshape(self._R, 3, 3) @ self._v
        eta_dot = vertcat(
            0.5*(-self._u[0]*self._eta[1] - self._u[1]*self._eta[2] - self._u[2]*self._eta[3]),
            0.5*(self._u[0]*self._eta[0] + self._u[2]*self._eta[2] - self._u[1]*self._eta[3]),
            0.5*(self._u[1]*self._eta[0] - self._u[2]*self._eta[1] + self._u[0]*self._eta[3]),
            0.5*(self._u[2]*self._eta[0] + self._u[1]*self._eta[1] - self._u[0]*self._eta[2])
        ) + c * self._eta 
        n_dot = - (self._f_ext) - self.get_external_distributed_forces()
        m_dot = - cross(p_dot, self._n) 
        tau_dot = SX.zeros(self._tau.shape[0])

        x = vertcat(self._p, self._eta, self._n, self._m, self._tau)
        xdot = vertcat(p_dot, eta_dot,
                       n_dot, m_dot, tau_dot)
        x_dot_impl = vertcat(self._p_d, self._eta_d, self._n_d, self._m_d, self._tau_d)
        
        self._static_model = AcadosModel()
        self._static_model.name = model_name
        self._static_model.x = x 
        self._static_model.f_expl_expr = xdot 
        self._static_model.f_impl_expr = xdot - x_dot_impl
        self._static_model.u = u
        self._static_model.z = SX([])
        self._static_model.xdot = x_dot_impl

        sim = AcadosSim()
        sim.model = self._static_model 

        Sf = self._boundary_length

        sim.code_export_directory = model_name
        sim.solver_options.T = Sf
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.num_stages = self._integration_stages
        sim.solver_options.num_steps = self._integration_steps
        # sim.solver_options.sens_forw = Fa_mlse

        acados_integrator = AcadosSimSolver(sim)

        return acados_integrator

    def get_external_distributed_forces(self):

        """TO DO: define p_dot and p_dotdot"""

        p_dot = reshape(self._R, 3, 3) @ self._v

        p_dotdot = reshape(self._R, 3, 3) @ skew(self._u) @ self._v

        for i in range(self._tau.shape[0]):

            f_t = - (self._tau[i]) * (skew(p_dot)@skew(p_dot))@p_dotdot / (norm_2(p_dot)**3)

        return f_t 

    def get_robot_arm_params(self): 

        return self._robot_arm_params_obj

    def get_static_robot_arm_model(self): 

        return self._static_model

    def get_boundary_length(self):

        return self._boundary_length

    def set_num_integration_stages(self, stages):

        self._integration_stages = stages

    def set_num_integrator_steps(self, steps): 

        self._integration_steps = steps

    def get_num_integration_stages(self):

        return self._integration_stages

    def get_num_integration_steps(self):

        return self._integration_steps
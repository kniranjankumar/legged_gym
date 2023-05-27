from legged_gym import LEGGED_GYM_ROOT_DIR
import os
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.envs.a1.a1_config import A1FlatCfg, A1RoughCfgPPO
from scipy.stats import truncnorm
import numpy as np
import torch
            
class TwoLegBalanceRobot(LeggedRobot):
    def __init__(self,*args,**kwargs):
        self.num_actors = 1
        self.relative_cube_pos = None
        super(TwoLegBalanceRobot, self).__init__(*args,**kwargs)

        
    # def get_privileged_obs(self):
    #     ## return box location
    #     return super(TwoLegBalanceRobot, self).get_privileged_obs()
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # create some wrapper tensors for different slices
        self.rigid_body_states = gymtorch.wrap_tensor(body_states)
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.all_root_states.view(self.num_envs, self.num_actors,13)[:,0,:]
        # self.cube_states = self.all_root_states.view(self.num_envs, self.num_actors,13)[:,1,:]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)#[:,0]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.foot_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:,self.feet_indices,:]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
    # def _create_envs(self):
    #     asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    #     asset_root = os.path.dirname(asset_path)
    #     asset_file = os.path.basename(asset_path)

    #     asset_options = gymapi.AssetOptions()
    #     asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
    #     asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
    #     asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
    #     asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
    #     asset_options.fix_base_link = self.cfg.asset.fix_base_link
    #     asset_options.density = self.cfg.asset.density
    #     asset_options.angular_damping = self.cfg.asset.angular_damping
    #     asset_options.linear_damping = self.cfg.asset.linear_damping
    #     asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
    #     asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
    #     asset_options.armature = self.cfg.asset.armature
    #     asset_options.thickness = self.cfg.asset.thickness
    #     asset_options.disable_gravity = self.cfg.asset.disable_gravity

    #     robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
    #     self.num_dof = self.gym.get_asset_dof_count(robot_asset)
    #     self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
    #     dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
    #     rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

    #     # save body names from the asset
    #     body_names = self.gym.get_asset_rigid_body_names(robot_asset)
    #     self.dof_names = self.gym.get_asset_dof_names(robot_asset)
    #     self.num_bodies = len(body_names)
    #     self.num_dofs = len(self.dof_names)
    #     feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
    #     penalized_contact_names = []
    #     for name in self.cfg.asset.penalize_contacts_on:
    #         penalized_contact_names.extend([s for s in body_names if name in s])
    #     termination_contact_names = []
    #     for name in self.cfg.asset.terminate_after_contacts_on:
    #         termination_contact_names.extend([s for s in body_names if name in s])

    #     asset_root = "../../IsaacGym_Preview_3_Package/isaacgym/assets/"
    #     asset_file = "urdf/cube.urdf"
    #     cube_asset_options = gymapi.AssetOptions()
    #     # cube_asset_options.density = 0.01
        
    #     # cube_asset = self.gym.load_asset(self.sim, asset_root, asset_file, cube_asset_options)
    #     # print("loaded 📦 asset")
    #     # cube_pose = gymapi.Transform()
    #     # cube_pose.r = gymapi.Quat(0, 0, 0, 1)
    #     # cube_pose.p = gymapi.Vec3(2, 0, 2)



    #     base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel #+ [2,0,2]+ [0,0,0,1] + [0,0,0] + [0,0,0]
    #     self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
    #     start_pose = gymapi.Transform()
    #     start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        

    #     self._get_env_origins()
    #     env_lower = gymapi.Vec3(0., 0., 0.)
    #     env_upper = gymapi.Vec3(0., 0., 0.)
    #     # cube_offset = gymapi.Vec3(0.38, 0.1, 0.2)
        
    #     self.actor_handles = []
    #     self.envs = []
    #     self.cube_handles = []
    #     self.all_env_ids = torch.tensor([i for i in range(self.num_envs)],device=self.device)
    #     for i in range(self.num_envs):
    #         # create env instance
    #         env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
    #         pos = self.env_origins[i].clone()
    #         # cube_pose.p = gymapi.Vec3(*pos) + cube_offset
    #         # pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
    #         pos[2] = 0.3
    #         start_pose.p = gymapi.Vec3(*pos)
    #         rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
    #         self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
    #         actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
    #         # self.gym.set_asset_rigid_shape_properties(cube_asset, rigid_shape_props)
            
    #         # cube_handle = self.gym.create_actor(env_handle, cube_asset, cube_pose, "cube", i, 0)
    #         # self.gym.set_actor_scale(env_handle, cube_handle, 4)
    #         dof_props = self._process_dof_props(dof_props_asset, i)
    #         self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
    #         body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
    #         body_props = self._process_rigid_body_props(body_props, i)
    #         self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
    #         self.envs.append(env_handle)
    #         self.actor_handles.append(actor_handle)
    #         # self.cube_handles.append(cube_handle)

    #     self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
    #     for i in range(len(feet_names)):
    #         self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

    #     self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
    #     for i in range(len(penalized_contact_names)):
    #         self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

    #     self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
    #     for i in range(len(termination_contact_names)):
    #         self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    # def reset_idx(self, env_ids):
    #     for env_id in env_ids:
    #         # print(self.actor_handles)
    #         body_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], 0)
    #         # print([(body_prop.com.x, body_prop.com.y, body_prop.com.z) for body_prop in body_props])
    #         # body_props[18].mass = np.random.uniform(0,1)*0.+0.001
    #         # body_props[17].mass = (0.1- body_props[18].mass)*0+0.001
    #         # print(self.gym.set_actor_rigid_body_properties(self.envs[env_id], self.actor_handles[env_id], body_props,recomputeInertia=True))
    #         body_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], 0)
    #         # print([body_prop.mass for body_prop in body_props])
    #         # print(self.gym.get_actor_rigid_body_names(self.envs[env_id], self.actor_handles[env_id]))
    #     # self.gym.simulate(self.sim) 
    #     super().reset_idx(env_ids)

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if env_id==0:
            self.mass_perturbation = self.sample(self.cfg.domain_rand.added_mass_range[0], 
                                                       self.cfg.domain_rand.added_mass_range_stddev[0], 
                                                       [-1.0,1.0], 
                                                       self.num_envs)
            self.com_shift_mass = self.sample(self.cfg.domain_rand.com_shift_mass_range[0], 
                                                       self.cfg.domain_rand.com_shift_mass_range_stddev[0], 
                                                       [0.1,2], 
                                                       self.num_envs)
            # print(self.com_shift_mass, self.cfg.domain_rand.com_shift_mass_range[0], 
            #                                            self.cfg.domain_rand.com_shift_mass_range_stddev[0], 
            #                                            [0.1,2])
        if self.cfg.domain_rand.randomize_base_mass:
            # rng = self.cfg.domain_rand.added_mass_range
            # props[0].mass += np.random.uniform(rng[0], rng[1])
            # props[0].mass = 4.713+ np.clip(np.random.normal(self.cfg.domain_rand.added_mass_range[0], 
            #                  self.cfg.domain_rand.added_mass_range_stddev[0]),-1,1)
            props[0].mass = self.mass_perturbation[env_id] + 4.713
            # props[0].mass = props[0].mass, -1, 1)
        if self.cfg.domain_rand.randomize_com:
            # props[18].mass = np.random.uniform(self.cfg.domain_rand.added_mass_range[0],
            #                                    self.cfg.domain_rand.added_mass_range[1])
            # print("com_shift_mass", self.com_shift_mass[env_id])
            props[18].mass = self.com_shift_mass[env_id] 
            props[17].mass = 2.1- props[18].mass
            # print(props[18].mass, props[17].mass, props[0].mass)
        return props
    
    def sample(self, loc, scale, bounds, size):
        """ Sample from chosen distribution"""
        if self.cfg.domain_rand.distribution == "uniform":
            return self.sample_uniform(bounds, size)
        elif self.cfg.domain_rand.distribution == "truncated_normal":
            return self.sample_truncated_normal(loc, scale, bounds, size)
        else:
            raise ValueError("Unknown distribution: {}".format(self.cfg.domain_rand.distribution))
        
    def sample_truncated_normal(self, loc, scale, bounds, size):
        """ Sample truncated normal distribution"""
        myclip_a, myclip_b = bounds
        # print(bounds,loc,scale)
        a, b = abs(myclip_a - loc) / scale, abs(myclip_b - loc) / scale
        # print(a,b, loc,scale,size)
        r = truncnorm.rvs(-a, b, loc, scale, size=size)
        return r
    
    def sample_uniform(self,bounds, size):
        """ Sample truncated normal distribution"""
        r = np.random.uniform(bounds[0], bounds[1], size=size)
        return r
    
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
        #     if env_id==0:
        #         # prepare friction randomization
        #         friction_range = self.cfg.domain_rand.friction_range
        #         num_buckets = 64
        #         bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
        #         friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
        #         self.friction_coeffs = friction_buckets[bucket_ids]

        #     for s in range(len(props)):
        #         props[s].friction = self.friction_coeffs[env_id]
            # friction = np.clip(np.random.normal(self.cfg.domain_rand.friction_range[0],
            #                                self.cfg.domain_rand.friction_range_stddev[0]), 0.5,1.25)
            if env_id==0:
                self.friction_coeffs = self.sample(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range_stddev[0], [0.5,1.25], self.num_envs)
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props
    
    def _reset_root_states(self, env_ids):
            """ Resets ROOT states position and velocities of selected environmments
                Sets base position based on the curriculum
                Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
            Args:
                env_ids (List[int]): Environemnt ids
            """
            # if 0 in env_ids:
                # print("########################################################################")
            # robot_bodies = ['base', 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot', 
            #                         'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 
            #                         'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot', 
            #                         'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot', 
            #                         'back_mass', 'front_mass']

            # print(body_props[0].com)
            if self.custom_origins:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            else:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # base velocities
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            # env_ids_int32_cube = env_ids.to(dtype=torch.int32)*2+1
            # env_ids_int32 = torch.flatten(torch.stack((env_ids.to(dtype=torch.int32), env_ids.to(dtype=torch.int32)*2+1),1))
            # self.cube_states[env_ids] = torch.tensor([0.38, 0.1, 0.2]+[0.]*3+[1.]+[0.]*6, device=self.root_states.device)
            # x, y = self.generate_target_location(len(env_ids))
            # self.cube_states[env_ids,0] = x
            # self.cube_states[env_ids,1] = y
            # self.cube_states[env_ids, :3] += self.env_origins[env_ids]
            
            # cube_state = torch.stack([torch.tensor([0.,0.,0.]+[0.]*3+[1.]+[0.]*6, device=self.root_states.device)]*self.num_envs,0)
            # cube_state[:,:3] = self.root_states[:,:3]+torch.tensor([[1,0,0]],device=self.device) 
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            # for env_id in env_ids:
            #     body_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.actor_handles[env_id])
            #     # print(body_props)
            #     # print(self.gym.get_actor_rigid_body_index(self.envs[env_id], self.object_handles[env_id],"link_2"))
            #     # print(self.gym.get_actor_rigid_body_names(self.envs[env_id], self.actor_handles[env_id]))
                
            #     # body_props[-1].mass = np.random.uniform(0,1)*0.1
            #     # body_props[-2].mass = np.random.uniform(0,1)*0.1
            #     self.gym.set_actor_rigid_body_properties(self.envs[env_id], self.actor_handles[env_id], body_props)
                # body_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.object_handles[env_id])
                # masses = [body.mass for body in body_props]


    # def generate_target_location(self, num_candidates):
    #     radius = 3.
    #     random_angle = torch.rand(num_candidates, device=self.device)*np.pi*0 #- np.pi/2*0
    #     x = radius * torch.sin(random_angle)
    #     y = radius * torch.cos(random_angle)
    #     return x, y
        
    # def _reset_dofs(self, env_ids):
    #         """ Resets DOF position and velocities of selected environmments
    #         Positions are randomly selected within 0.5:1.5 x default positions.
    #         Velocities are set to zero.

    #         Args:
    #             env_ids (List[int]): Environemnt ids
    #         """
    #         self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
    #         self.dof_vel[env_ids] = 0.

    #         env_ids_int32 = env_ids.to(dtype=torch.int32)* self.num_actors
    #         self.gym.set_dof_state_tensor_indexed(self.sim,
    #                                             gymtorch.unwrap_tensor(self.dof_state),
    #                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def reset_dr_params(self):
        default_rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[0], self.actor_handles[0])
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.envs[i]
            actor_handle = self.actor_handles[i]
            # print("resetting friction")
            rigid_shape_props = self._process_rigid_shape_props(default_rigid_shape_props, i)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
        # env_ids = self.all_env_ids
        # reset robot states
        # with torch.inference_mode():
        #     self.reset()

    def compute_observations(self):
        """ Computes observations
        """
        # relative_cube_pos[:,3] = 0
        # print(self.relative_cube_pos[0])
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,                        #3
                                    self.base_ang_vel  * self.obs_scales.ang_vel,                       #3
                                    self.projected_gravity,                                             #3
                                    # torch.zeros_like(relative_cube_pos),                              #3
                                    # self.relative_cube_pos[:,:2],                                       #3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,    #12
                                    self.dof_vel * self.obs_scales.dof_vel,                             #12
                                    self.actions                                                        #12
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # print(torch.isnan(self.obs_buf).any())
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def post_physics_step(self):
        
        # cube_pos = self.cube_states[:,:3]
        # cube_orientation = self.cube_states[:,3:7]
        agent_pos = self.root_states[:,:3]
        agent_orientation = self.root_states[:,3:7]
        q,t = tf_inverse(agent_orientation, agent_pos)
        # self.relative_cube_pos = tf_apply(q,t,cube_pos)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # print(cube_pos[0], self.relative_cube_pos[0], t[0])
        super(TwoLegBalanceRobot, self).post_physics_step()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        super().check_termination()
        orientation_limit = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)[0] > 0.2
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # error = torch.square(base_height - self.cfg.rewards.base_height_target)
        self.reset_buf |= orientation_limit
        # self.reset_buf |= error > 0.1
    
        
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states.contiguous()))
    
    # def _reward_target_reach(self,):
    #     target_distance = torch.clip(torch.norm(self.relative_cube_pos[:,:2], dim=1), 0)
    #     rew = torch.exp(-target_distance)
    #     # print(rew[0])
    #     return rew
    
    
    def _reward_foot_contact(self):
        # Penalty if wrong feet touch the ground
        weight = 0.5
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_not(torch.logical_or(contact, self.last_contacts))
        # print((contact_filt[0,1]+ contact_filt[0,2] - contact_filt[0,0] - contact_filt[0,3]))
        return (torch.logical_and(contact_filt[:,1], contact_filt[:,2])).double()*weight #+ (torch.logical_not(torch.logical_or(contact_filt[:,0], contact_filt[:,3]))).double()*weight
    
    def _reward_foot_lift(self):
        # reward for distance from ground
        weight = 0.5
        foot_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:,self.feet_indices,:]
        FR_foot_z = foot_states[:,1,2]
        RL_foot_z = foot_states[:,2,2]
        # body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # names = self.gym.get_actor_rigid_body_names(self.envs[0],0)
        # # foot indices 4, 8, 12, 17
        # print(names)
        # print(gymtorch.wrap_tensor(body_states)[self.feet_indices,:])
        # print(f"FL: {FL_foot_z[0]}, RR: {RR_foot_z[0]}")
        # return torch.min(torch.vstack((FR_foot_z,RL_foot_z)),1)[0].double()*weight
        return torch.minimum(FR_foot_z,RL_foot_z).double()*weight
    
    def _reward_action_magnitude(self):
        return torch.norm(self.actions, p=1, dim=1)
    
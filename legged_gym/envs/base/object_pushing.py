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
            
class PushingRobot(LeggedRobot):
    def __init__(self,*args,**kwargs):
        self.num_actors = 3
        self.relative_cube_pos = None
        super(PushingRobot, self).__init__(*args,**kwargs)

        
    def get_privileged_obs(self):
        ## return box location
        return super(PushingRobot, self).get_privileged_obs()
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.all_root_states.view(self.num_envs, self.num_actors,13)[:,0,:]
        self.puck_states = self.all_root_states.view(self.num_envs, self.num_actors,13)[:,1,:]
        self.target_states = self.all_root_states.view(self.num_envs, self.num_actors,13)[:,2,:]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)#[:,0]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

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
        
    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        asset_root = "../../IsaacGym_Preview_3_Package/isaacgym/assets/"
        asset_file = "urdf/cube_big.urdf"
        puck_asset_file = "urdf/puck.urdf"
        cube_asset_options = gymapi.AssetOptions()
        puck_asset_options = gymapi.AssetOptions()
        # puck_asset_options.density = 0.05
        
        cube_asset_options.density = 0.01
        
        cube_asset = self.gym.load_asset(self.sim, asset_root, asset_file, cube_asset_options)
        puck_asset = self.gym.load_asset(self.sim, asset_root, puck_asset_file, puck_asset_options)
        puck_rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(puck_asset)
        print("loaded 📦 asset")
        cube_pose = gymapi.Transform()
        cube_pose.r = gymapi.Quat(0, 0, 0, 1)
        # cube_pose.p = gymapi.Vec3(2, 0, 2)



        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel #+ [2,0,2]+ [0,0,0,1] + [0,0,0] + [0,0,0]
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        cube_offset = gymapi.Vec3(0.5, 0.1, 0.2)
        
        self.actor_handles = []
        self.envs = []
        self.puck_handles = []

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            cube_pose.p = gymapi.Vec3(*pos) + cube_offset
            # pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            pos[2] = 0.3
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

            # puck_rigid_shape_props = self._process_puck_rigid_body_props(puck_rigid_shape_props_asset, i)
            # self.gym.set_asset_rigid_shape_properties(puck_asset, puck_rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            # self.gym.set_asset_rigid_shape_properties(cube_asset, rigid_shape_props)

            puck_handle = self.gym.create_actor(env_handle, puck_asset, cube_pose, "puck", i, 0) # manipulate this cube
            puck_body_props = self.gym.get_actor_rigid_body_properties(env_handle, puck_handle)
            puck_body_props = self._process_puck_rigid_body_props(puck_body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, puck_handle, puck_body_props, recomputeInertia=True)
            
            cube_handle2 = self.gym.create_actor(env_handle, cube_asset, cube_pose, "cube", self.num_envs+1, 0) #target marker
            self.gym.set_rigid_body_color(env_handle, cube_handle2, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1,0,0))
            # self.gym.set_actor_scale(env_handle, cube_handle, 4)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.puck_handles.append(puck_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])


    def _reset_root_states(self, env_ids):
            """ Resets ROOT states position and velocities of selected environmments
                Sets base position based on the curriculum
                Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
            Args:
                env_ids (List[int]): Environemnt ids
            """
            # if 0 in env_ids:
                # print("########################################################################")
            if self.custom_origins:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            else:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # base velocities
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
            env_ids_int32_robot = env_ids.to(dtype=torch.int32)*2
            env_ids_int32_cube = env_ids.to(dtype=torch.int32)*2+1
            # env_ids_int32 = torch.flatten(torch.stack((env_ids.to(dtype=torch.int32)*self.num_actors, env_ids.to(dtype=torch.int32)*2+1),1))
            actor_ids = torch.flatten(torch.linspace(0, self.num_actors*self.num_envs-1,self.num_envs*self.num_actors,device=self.device).reshape(self.num_envs,self.num_actors)[env_ids])
            actor_ids_int32 = actor_ids.to(dtype=torch.int32)
            
            self.puck_states[env_ids] = torch.tensor([0.6, 0.2, 0.085]+[0.]*3+[1.]+[0.]*6, device=self.root_states.device)
            self.puck_states[env_ids,:2] += self.env_origins[env_ids,:2]
            x, y = self.generate_target_location(len(env_ids))
            self.target_states[env_ids,0] = x
            self.target_states[env_ids,1] = y
            self.target_states[env_ids, :3] += self.env_origins[env_ids]
            
            # cube_state = torch.stack([torch.tensor([0.,0.,0.]+[0.]*3+[1.]+[0.]*6, device=self.root_states.device)]*self.num_envs,0)
            # cube_state[:,:3] = self.root_states[:,:3]+torch.tensor([[1,0,0]],device=self.device) 
            # all_bodies_root_state = torch.reshape(torch.cat((self.root_states,self.puck_states),1), (-1,13))
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.all_root_states),
                                                        gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))

            # for env_id in env_ids:
            #     body_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.puck_handles[env_id])
            #     # print(body_props)
            #     # print(self.gym.get_actor_rigid_body_index(self.envs[env_id], self.object_handles[env_id],"link_2"))
            #     # print(self.gym.get_actor_rigid_body_names(self.envs[env_id], self.object_handles[env_id]))
            #     body_props = self._process_puck_rigid_body_props(body_props, env_id)
            #     # body_props[0].mass = np.random.uniform(0,1)+1
            #     self.gym.set_actor_rigid_body_properties(self.envs[env_id], self.puck_handles[env_id], body_props, recomputeInertia=True)
            #     # body_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.object_handles[env_id])
            #     # masses = [body.mass for body in body_props]

    def generate_target_location(self, num_candidates):
        radius = 2.
        random_angle = torch.rand(num_candidates, device=self.device)*np.pi/2*0 +np.pi/2 #- np.pi/2
        x = radius * torch.sin(random_angle)
        y = radius * torch.cos(random_angle)
        return x, y
        
    def _reset_dofs(self, env_ids):
            """ Resets DOF position and velocities of selected environmments
            Positions are randomly selected within 0.5:1.5 x default positions.
            Velocities are set to zero.

            Args:
                env_ids (List[int]): Environemnt ids
            """
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
            self.dof_vel[env_ids] = 0.

            env_ids_int32 = env_ids.to(dtype=torch.int32)* self.num_actors
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def reset_dr_params(self):

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.envs[i]
            puck_handle = self.puck_handles[i]
            # print("resetting friction")
            # rigid_shape_props = self._process_rigid_shape_props(default_rigid_shape_props, i)
            # self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, puck_handle)
            body_props = self._process_puck_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, puck_handle, body_props, recomputeInertia=True)
            
    def _process_puck_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        # props[0].mass = 1#+np.random.uniform(0,10) # 0.1
        if env_id==0:
            self.puck_mass = self.sample(self.cfg.domain_rand.puck_mass_sampled[0], 
                                                       self.cfg.domain_rand.puck_mass_sampled[1], 
                                                       self.cfg.domain_rand.puck_mass_range, 
                                                       self.num_envs)
        if self.cfg.domain_rand.randomize_puck_mass:
            props[0].mass = self.puck_mass[env_id]
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
    
    def compute_observations(self):
        """ Computes observations
        """
        # relative_cube_pos[:,3] = 0
        # print(self.relative_cube_pos[0])
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,                        #3
                                    self.base_ang_vel  * self.obs_scales.ang_vel,                       #3
                                    self.projected_gravity,                                             #3
                                    # torch.zeros_like(relative_cube_pos),                              #3
                                    self.agent_relative_cube_pos[:,:2],                                 #2 
                                    self.agent_relative_target_pos[:,:2],                               #2    
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,    #12
                                    self.dof_vel * self.obs_scales.dof_vel,                             #12
                                    self.actions                                                        #12
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def post_physics_step(self):
        self.agent_relative_cube_pos = self.get_relative_translation([self.root_states[:,3:7], self.root_states[:,:3]],
                                                               [self.puck_states[:,3:7], self.puck_states[:,:3]])
        # self.agent_relative_cube_pos[:,0] -= 0.3
        self.agent_relative_target_pos = self.get_relative_translation([self.root_states[:,3:7], self.root_states[:,:3]],
                                                               [self.target_states[:,3:7], self.target_states[:,:3]])
        self.cube_relative_target_pos = self.get_relative_translation([self.puck_states[:,3:7], self.puck_states[:,:3]],
                                                               [self.target_states[:,3:7], self.target_states[:,:3]])
        # print(cube_pos[0], self.relative_cube_pos[0], t[0])
        super(PushingRobot, self).post_physics_step()
        
    # def check_termination(self):
    #     """ Check if environments need to be reset
    #     """
    #     self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
    #     self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
    #     self.reset_buf |= self.time_out_buf
    
    def get_relative_translation(self, transform1, transform2):
        """_summary_

        Args:
            transform1 (List): Containing orientation first and translation second
            transform2 (List): Containing orientation first and translation second
        """
        q,t = tf_inverse(transform1[0], transform1[1])
        return tf_apply(q,t,transform2[1])
        
    def _push_robots(self):
        """ 
            Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.all_root_states.contiguous()))
    
    def _reward_object_target_dist(self,):
        
        target_distance = torch.clip(torch.norm(self.cube_relative_target_pos[:,:2], dim=1), 0)
        rew = torch.exp(-target_distance/2)
        # print(rew[0])
        return rew
    
    def _reward_object_robot_dist(self,):
        
        target_distance = torch.clip(torch.norm(self.agent_relative_cube_pos[:,:2], dim=1), 0)
        rew = torch.exp(-target_distance/3)
        # print(rew[0])
        return rew
 
    def _reward_robot_target_dist(self,):
        
        target_distance = torch.clip(torch.norm(self.agent_relative_target_pos[:,:2], dim=1),0,2)
        rew = torch.exp(target_distance/2)
        # print(rew[0])
        return rew
        
    def reset_idx(self, env_ids):        
        if len(env_ids) != 0:
            # success = (self.root_states[env_ids, 0]>(self.env_origins[env_ids,0]+4)).type(torch.float32)
            if hasattr(self, 'agent_relative_target_pos'):
                # success = torch.logical_or(torch.norm(self.agent_relative_target_pos[env_ids,:2], dim=1) < 0.5, self.episode_length_buf[env_ids] < 50)
                # print(self.agent_relative_door_pos[torch.logical_not(success),:2])
                # success = self.success[env_ids]
                success = torch.norm(self.cube_relative_target_pos[env_ids,:2], dim=1) < 0.5
                not_messy_init = self.episode_length_buf[env_ids] > 50
                not_messy_init = not_messy_init | success
                success = torch.masked_select(success, not_messy_init)
                # print("failed", env_ids, success)      
                
                # failed_envs = torch.stack([torch.masked_select(self.target_states[:,0],self.success),
                #                               torch.masked_select(self.target_states[:,1],self.success)], dim=1)
                # if failed_envs.size(0)>0:
                #     self.failed_envs = failed_envs
                # print(self.failed_envs.size())
            else:
                success = torch.zeros_like(env_ids, device=self.device).type(torch.float32)
            # print(success)
        super().reset_idx(env_ids)
        if len(env_ids) != 0:
            self.extras["episode"]["success"] = success

    # def check_termination(self):
    #     super().check_termination()
    #     # self.reset_buf |= torch.norm(self.agent_relative_cube_pos[:,:2], dim=1) > 2.0
    #     target_distance = torch.clip(torch.norm(self.agent_relative_target_pos[:,:2], dim=1),0,2)
    #     self.reset_buf |= target_distance < 1.0
        
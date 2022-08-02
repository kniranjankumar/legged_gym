from legged_gym import LEGGED_GYM_ROOT_DIR
import os
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.envs.a1.a1_config import A1FlatCfg, A1RoughCfgPPO
import numpy as np
import torch
            
class InteractiveRobot(LeggedRobot):
    def __init__(self,*args,**kwargs):
        self.igibson_asset_root = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/resources/"
        self.asset_paths = ["scenes/8903/two_room_house.urdf",
                            "objects/lack_table/lack.urdf",
                            # "trash_can/11259/11259.urdf",
                            # "trash_can/102254/102254.urdf",
                            # "office_chair/179/179.urdf",
                            # "office_chair/723/723.urdf",
                            # "office_chair/2490/2490.urdf"
                            ]
        self.asset_offsets = [[2.0,0.0,0.93],
                              [0,0,0],
                              [-2,2,1],
                              [4,4,2],
                              [4,-4,2],
                              [-4,4,2]
                              ]
        self.cube_offset = [-1., -0.13, 0.5]
        
        self.asset_orientations = [[ 0, 0, 0.7071068, 0.7071068 ],
                                    [0,0,0,1],
                                    [0,0,0,1],
                                    [0,0,0,1],
                                    [0,0,0,1],
                                    [0,0,0,1]]
        self.actor_dofs = [12]
        self.num_actors = 1 + 1 + len(self.asset_paths)
        
        # kwargs["num_velocity_iterations"] = 1
        super(InteractiveRobot, self).__init__(*args,**kwargs)

        
    def get_privileged_obs(self):
        ## return box location
        return super(InteractiveRobot, self).get_privileged_obs()
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        self.object_states  = []
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.all_root_states.view(self.num_envs, self.num_actors,13)[:,0,:]
        self.object_states_tensor = self.all_root_states.view(self.num_envs, self.num_actors,13)[:,1:,:]
        self.target_states = self.all_root_states.view(self.num_envs, self.num_actors,13)[:,-1,:]
        for i in range(self.num_actors-1):
            self.object_states.append(self.object_states_tensor)

        self.all_dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state = self.all_dof_states.view(self.num_envs, sum(self.actor_dofs),2)[:,:self.num_dof,:]
        self.objects_dof_states = self.all_dof_states.view(self.num_envs, sum(self.actor_dofs),2)[:,self.num_dof:,:]
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

        #load assets
        target_asset = "urdf/cube_big.urdf"    
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.fix_base_link = True
        
        cube_asset = self.gym.load_asset(self.sim, "../../IsaacGym_Preview_3_Package/isaacgym/assets/", target_asset, cube_asset_options)
        
        
        igibson_assets = []
        igibson_asset_options = gymapi.AssetOptions()
        igibson_asset_options.density = 10.0
        igibson_asset_options.fix_base_link = True
        igibson_asset_options.vhacd_enabled = True
        # igibson_asset_options.vhacd_params.max_convex_hulls = 10
        for asset_path in self.asset_paths:
            folder, urdf_name = os.path.split(asset_path)
            path = os.path.join(self.igibson_asset_root,folder)
            igibson_assets.append(self.gym.load_asset(self.sim, path, urdf_name, igibson_asset_options))
            print("loaded ", asset_path)
            num_revolute_joints = sum([self.gym.get_asset_joint_type(igibson_assets[-1],joint_id).name == "JOINT_REVOLUTE" for joint_id in range(self.gym.get_asset_joint_count(igibson_assets[-1]))])
            self.actor_dofs.append(num_revolute_joints)
            

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel #+ [2,0,2]+ [0,0,0,1] + [0,0,0] + [0,0,0]
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        cube_pose = gymapi.Transform()
        cube_pose.r = gymapi.Quat(0, 0, 0, 1)
        # start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)

        cube_init_state = [torch.tensor(self.cube_offset+[0.]*3+[1.]+[0]+[0.]*5, device=self.device)]
        imported_asset_states = [torch.tensor(self.asset_offsets[i]+self.asset_orientations[i]+[0.]*6, device=self.device) for i in range(len(self.asset_paths))]
        imported_asset_states += cube_init_state
        self.objects_init_states = torch.stack(imported_asset_states, 0)
        
        self.actor_handles = []
        self.envs = []
        self.object_handles = []

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[0] += torch_rand_float(-1., 1., (1,1) ,device=self.device).squeeze(1)[0]
            
            cube_pose.p = gymapi.Vec3(*pos) + gymapi.Vec3(*self.cube_offset)
            # pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            # pos[2] = 0.3
            start_pose.p = gymapi.Vec3(*pos) + gymapi.Vec3(*self.base_init_state[:3])
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name+str(i), i, 0)
            # self.gym.set_asset_rigid_shape_properties(cube_asset, rigid_shape_props)            
            # self.gym.set_actor_scale(env_handle, cube_handle, 4)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            
            # self.object_handles.append(cube_handle)
            for asset, location, orientation in zip(igibson_assets,self.asset_offsets, self.asset_orientations):
                asset_start_pose = gymapi.Transform()
                asset_start_pose.p = gymapi.Vec3(*location) + gymapi.Vec3(*self.env_origins[i].clone())
                asset_start_pose.r = gymapi.Quat(*orientation)
                object_handle = self.gym.create_actor(env_handle, asset, asset_start_pose, "door"+str(i), i, 1)
                self.object_handles.append(object_handle)
                props = self.gym.get_actor_dof_properties(env_handle, object_handle)
                props["driveMode"].fill(gymapi.DOF_MODE_NONE)
                props["stiffness"].fill(5.0)
                props["damping"].fill(0.0)
                props["friction"].fill(0.5)
                self.gym.set_actor_dof_properties(env_handle, object_handle, props)
            cube_handle = self.gym.create_actor(env_handle, cube_asset, cube_pose, "cube",self.num_envs+1, 0)
            self.gym.set_rigid_body_color(env_handle, cube_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1,0,0))
                  
        self.actors_with_dofs = [i for i, dof in enumerate(self.actor_dofs) if dof>0]
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
            # if 1 in env_ids:
                # print("########################################################################")
            # base position
            # print(self.root_states[env_ids, 0]>(self.env_origins[env_ids,0]+3))
            if self.custom_origins:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            else:
                self.root_states[env_ids] = self.base_init_state
                self.root_states[env_ids, :3] += self.env_origins[env_ids]
                robot_x, robot_y = self.generate_target_location(len(env_ids))
                # self.root_states[env_ids,0] += robot_x
                # self.root_states[env_ids,1] += robot_y
                self.root_states[env_ids,0] += torch_rand_float(-3., 1., (len(env_ids),1), device=self.device)[:,0] #randomize distance from door
                self.root_states[env_ids,1] += torch_rand_float(-2., 2., (len(env_ids),1), device=self.device)[:,0] #randomize distance from door
                orientation_z = torch_rand_float(-np.pi, np.pi, (len(env_ids),1), device=self.device)[:,0]
                self.root_states[env_ids,3:7] = quat_from_euler_xyz(torch.zeros_like(orientation_z), torch.zeros_like(orientation_z),orientation_z)
            # base velocities
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
            self.object_states_tensor[env_ids,:,:] = self.objects_init_states
            self.object_states_tensor[env_ids,:,:3] += self.env_origins[env_ids].unsqueeze(1)
            self.target_states[env_ids] = torch.tensor([0.38, 0.1, 0.2]+[0.]*3+[1.]+[0.]*6, device=self.root_states.device)
            x, y = self.generate_target_location(len(env_ids))
            self.target_states[env_ids,0] = x
            self.target_states[env_ids,1] = y
            self.target_states[env_ids, :3] += self.env_origins[env_ids]

            actor_ids = torch.flatten(torch.linspace(0, self.num_actors*self.num_envs-1,self.num_envs*self.num_actors,device=self.device).reshape(self.num_envs,self.num_actors)[env_ids])
            actor_ids_int32 = actor_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.all_root_states),
                                                        gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))

    def _reset_dofs(self, env_ids):
            """ Resets DOF position and velocities of selected environmments
            Positions are randomly selected within 0.5:1.5 x default positions.
            Velocities are set to zero.

            Args:
                env_ids (List[int]): Environemnt ids
            """
            self.objects_dof_states[env_ids,:,1] = 0.0
            self.objects_dof_states[env_ids,:,0] = 0.0
             
            # self.objects_dof_states *= 0.
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
            self.dof_vel[env_ids] = 0.
            # actor_ids = torch.flatten(torch.linspace(0, 2*sum(self.actor_dofs)*self.num_envs-1,2*self.num_envs*sum(self.actor_dofs),device=self.device).reshape(self.num_envs,sum(self.actor_dofs),2)[env_ids])
            actors_in_env_ids = torch.linspace(0, self.num_actors*self.num_envs-1,self.num_envs*self.num_actors,device=self.device).reshape(self.num_envs,self.num_actors)[env_ids]
            actor_ids = torch.flatten(torch.index_select(actors_in_env_ids, 1,torch.tensor(self.actors_with_dofs,device=self.device)))
            env_ids_int32 = actor_ids.to(dtype=torch.int32)#* self.num_actors
            # env_ids_int32 = env_ids.to(dtype=torch.int32)*self.num_actors
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.all_dof_states),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.all_actor_torques = torch.nn.functional.pad(self.torques,(0,sum(self.actor_dofs)-12),"constant",1)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.all_actor_torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def compute_observations(self):
        """ Computes observations
        """
        # print(torch.abs(self.objects_dof_states[:,0,0]))
        # print(self.base_lin_vel[0].tolist())
        
        
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,                        #3
                                    self.base_ang_vel  * self.obs_scales.ang_vel,                       #3
                                    self.projected_gravity,                                             #3
                                    self.agent_relative_target_pos[:,:2],                                 #2
                                    self.agent_relative_door_pos[:,:2],                                 #2
                                    torch.abs(self.objects_dof_states[:,0,0]).view(-1,1),               #1  
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,    #12
                                    self.dof_vel * self.obs_scales.dof_vel,                             #12
                                    self.actions,                                                        #12
                                    torch.clamp(self.root_states[:,:2]-self.env_origins[:,:2],torch.tensor([-3,-3],device=self.root_states.device),torch.tensor([3,8],device=self.root_states.device))/10                         #12
                                    ),dim=-1)
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def generate_radial_target_location(self, num_candidates):
        radius = torch.rand(num_candidates, device=self.device)*3.
        # radius = 3.
        random_angle = torch.rand(num_candidates, device=self.device)*np.pi #- np.pi/2
        x = radius * torch.sin(random_angle)
        y = radius * torch.cos(random_angle)
        return x, y
    
    def generate_target_location(self, num_candidates):
        x = torch.rand(num_candidates, device=self.device)*9-3
        y = torch.rand(num_candidates, device=self.device)*5-2.5
        # sample door position more to help robot exit the room
        # if torch.rand(1)<0.99:
        #     x = x*0+2
        #     y = y*0
        return x,y    
    
    def get_relative_translation(self, transform1, transform2):
        """_summary_

        Args:
            transform1 (List): Containing orientation first and translation second
            transform2 (List): Containing orientation first and translation second
        """
        q,t = tf_inverse(transform1[0], transform1[1])
        return tf_apply(q,t,transform2[1])
    
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states.contiguous()))
    
    def post_physics_step(self):
        self.agent_relative_door_pos = self.get_relative_translation([self.root_states[:,3:7], self.root_states[:,:3]],
                                                               [self.object_states_tensor[:,0,3:7], self.object_states_tensor[:,0,:3]])
        self.agent_relative_target_pos = self.get_relative_translation([self.root_states[:,3:7], self.root_states[:,:3]],
                                                               [self.target_states[:,3:7], self.target_states[:,:3]])
        super(InteractiveRobot, self).post_physics_step()
    
    def get_relative_translation(self, transform1, transform2):
        """_summary_

        Args:
            transform1 (List): Containing orientation first and translation second
            transform2 (List): Containing orientation first and translation second
        """
        q,t = tf_inverse(transform1[0], transform1[1])
        return tf_apply(q,t,transform2[1])

    def reset_idx(self, env_ids):        
        if len(env_ids) != 0:
            # success = (self.root_states[env_ids, 0]>(self.env_origins[env_ids,0]+4)).type(torch.float32)
            if hasattr(self, 'agent_relative_target_pos'):
                success = torch.norm(self.agent_relative_target_pos[:,:2], dim=1) < 0.1
            else:
                success = torch.zeros_like(env_ids, device=self.device).type(torch.float32)
            # print(success)
        super().reset_idx(env_ids)
        if len(env_ids) != 0:
            self.extras["episode"]["success"] = success

    def get_robot_position(self):
        return self.root_states[:,:2]-self.env_origins[:,:-1]

    def _reward_door_angle(self,):
        # past_door_bool = ((self.root_states[:,:2]-self.object_states_tensor[:,0,:2]).view(-1,2) > 0.8).type(self.objects_dof_states.type)
        return torch.abs(self.objects_dof_states[:,0,0])
    
    def _reward_cross_door(self,):
        crossed_bool = self.root_states[:, 0]>(self.env_origins[:,0]+4)
        return crossed_bool.type(torch.float64)     
    
    def _reward_robot_target_dist(self,):
        target_distance = torch.clip(torch.norm( self.agent_relative_target_pos[:,:2], dim=1), 0)
        target_room = (self.target_states[:,0]-self.env_origins[:,0])>2
        robot_room = (self.root_states[:,0]-self.env_origins[:,0])>2
        same_room_mask = (target_room==robot_room).type(torch.float64)
        door_distance = torch.clip(torch.norm( self.agent_relative_door_pos[:,:2], dim=1), 0)
        factor = 0.5
        rew = torch.exp(-target_distance)*same_room_mask + torch.exp(-door_distance) *(1-same_room_mask)*factor
        # print(rew[0])
        return rew   
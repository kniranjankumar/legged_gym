# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .base.object_pushing import PushingRobot
from .base.door_opening_robot import DoorOpeningRobot
from .base.door_opening_robotv2 import DoorOpeningRobotv2
from .base.target_reaching import TargetReachingRobot
from .base.straight_walking_robot import StraightWalkingRobot
from .base.standing_robot import StandingRobot
from .base.turning_robot import TurningRobot
from .base.visual_explorer import VisualRobot
from .base.crouching_robot import CrouchingRobot
from .base.interactive_target_reach import  InteractiveRobot
from .base.interactive_target_reachv2 import  InteractiveRobotv2
from .base.two_leg_balance_robot import TwoLegBalanceRobot
from .base.ballu_walk import WalkingBallu
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import * #A1RoughCfg, A1RoughCfgPPO, A1FlatCfg, A1TargetReachCfg, A1FlatCfgPPO, A1MultiSkillCfgPPO, A1MultiSkillReachCfgPPO
from .ballu.ballu_config import *
import os
from legged_gym.utils.task_registry import task_registry

task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )


task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "a1_flat", LeggedRobot, A1FlatCfg(), A1FlatCfgPPO() )

task_registry.register( "a1_straight_walker", StraightWalkingRobot, A1FlatCfg(), A1FlatCfgPPO() )
task_registry.register( "turning", TurningRobot, A1TurnCfg(), A1TurnCfgPPO())
task_registry.register( "dooropen", DoorOpeningRobot, A1DoorOpeningCfg(), A1DoorOpeningCfgPPO())
task_registry.register( "dooropenv2", DoorOpeningRobotv2, A1DoorOpeningv2Cfg(), A1DoorOpeningv2CfgPPO())
task_registry.register( "reachtarget", TargetReachingRobot, A1TargetReachCfg(), A1MultiSkillReachCfgPPO())
task_registry.register( "pushobject", PushingRobot, A1TargetObjectPushCfg(), A1MultiSkillObjectPushCfgPPO())
task_registry.register( "standing", StandingRobot, A1StandingCfg(), A1StandingCfgPPO())
task_registry.register( "visualrobot", VisualRobot, A1DoorOpeningv2Cfg(), A1DoorOpeningv2CfgPPO())
task_registry.register( "crouching", CrouchingRobot, A1CrouchingCfg(), A1CrouchingCfgPPO())

task_registry.register("interactive_targetreach", InteractiveRobot, InteractiveTargetReachCfg(), InteractiveTargetReachCfgPPO())
task_registry.register("interactive_targetreachv2", InteractiveRobotv2, InteractiveTargetReachv2Cfg(), InteractiveTargetReachv2CfgPPO())
task_registry.register("interactive_targetreachv3", InteractiveRobot, InteractiveTargetReachv3Cfg(), InteractiveTargetReachv3CfgPPO())

task_registry.register("two_leg_balance", TwoLegBalanceRobot, TwoLegBalanceCfg(), TwoLegBalanceCfgPPO())

task_registry.register("bayrn_blockpush", PushingRobot, A1BayrnPushCfg(), A1BayrnPushCfgPPO())


task_registry.register( "ballu", WalkingBallu, BalluCfg(), BalluCfgPPO() )
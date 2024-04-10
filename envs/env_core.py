import random
import time

import numpy as np
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)
import argparse
from metadrive.constants import HELP_MESSAGE
from metadrive.policy.idm_policy import ManualControllableIDMPolicy

envs = dict(
    roundabout=MultiAgentRoundaboutEnv,
    intersection=MultiAgentIntersectionEnv,
    tollgate=MultiAgentTollgateEnv,
    bottleneck=MultiAgentBottleneckEnv,
    parkinglot=MultiAgentParkingLotEnv,
    pgma=MultiAgentMetaDrive
)

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self,args,config):
        self.args=args
        env_cls_name = args.env
        self.env= envs[env_cls_name](
            config
         )
        self.agent_num = self.args.num_agents
        self.obs_dim = list(self.env.observation_space.values())[0].shape[0]
        self.action_dim = list(self.env.action_space.values())[0].shape[0]
        #[Box(-inf, inf, (2,), float32), Box(-inf, inf, (2,), float32)]
        self.action_space = list(self.env.action_space.values())
        self.observation_space=list(self.env.observation_space.values())



    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        state = self.env.reset()
        sub_agent_obs=list(state[0].values())

        """
        [array([0.49429161, 0.11767061, 0.08365448, 0.80148316, 0.00954728,
       0.17152566, 0.90665392, 0.16911162, 0.29806652, 0.25131156,
       0.99582807, 0.61365704, 0.82655726, 0.16925658]), 
       array([0.98333378, 0.96975691, 0.56973207, 0.91186993, 0.20662234,
       0.69301029, 0.93051759, 0.79534628, 0.06723421, 0.25382497,
       0.19824206, 0.65885847, 0.62897183, 0.28868158])]
        """
        return sub_agent_obs

    def step(self, actions):
        # print(actions)
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        sub_agent_obs,reward,done,truncateds,info = self.env.step({agent_id: action for agent_id,action in zip(self.env.agents.keys(),actions)})
        sub_agent_obs=list(sub_agent_obs.values())
        sub_agent_reward=list(reward.values())
        sub_agent_done = list(done.values())[:-1]
        sub_agent_info = list(info.values())

        new_agent_processed = False  # Flag to track if new agent's data is already used
        for agent_index, agent_done in enumerate(sub_agent_done):
            if agent_done:
                if len(sub_agent_done)>self.agent_num :
                    # if len(sub_agent_done)!=self.agent_num:
                    sub_agent_obs[agent_index] = sub_agent_obs[-1]
                    sub_agent_obs = np.delete(sub_agent_obs, -1, axis=0)
                    sub_agent_reward = np.delete(sub_agent_reward, -1)
                    sub_agent_done = np.delete(sub_agent_done, -1)
                    sub_agent_info = np.delete(sub_agent_info, -1)
                    new_agent_processed = True  # Mark that new agent's data has been used
                elif new_agent_processed and len(sub_agent_done)==self.agent_num:
                    # If another agent is done and the new agent's data is already used, skip processing
                    continue
                else:
                    # If the agent is done but hasn't reached its destination, reset the environment
                    # ref = my_env.reset()
                    break  # Assuming the entire environment is reset
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def close(self):
        self.env.close()


if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="straight", choices=list(envs.keys()))
    parser.add_argument("--top_down", action="store_true",default=True)
    parser.add_argument("--num_agents", type=int,default=5)
    args = parser.parse_args()
    config=dict(
        random_spawn_lane_index=False,
        start_seed=0,
        random_traffic=False,
        use_render=True,
        crash_done=True,
        sensors=dict(rgb_camera=(RGBCamera, 512, 256)),
        # start_seed=random.randint(0, 1000),
        show_coordinates=True,
        allow_respawn=True,
        delay_done=0,
        image_observation=False,
        # interface_panel=["rgb_camera", "dashboard"],

        # agent_policy=ManualControllableIDMPolicy,
        num_agents=args.num_agents,
        vehicle_config=dict(
            lidar=dict(
                add_others_navi=False,
                num_others=4,
                distance=50,
                num_lasers=30,
            ),
            side_detector=dict(num_lasers=30),
            lane_line_detector=dict(num_lasers=12),
        )
    )
    my_env=EnvCore(args,config)

    my_env.reset()
    action=[np.zeros(2)+[0,1] for i in range(args.num_agents)]


    def check_arrival(done, info):
        """
        Check if any agent has arrived at its destination.

        :param done: List of boolean values indicating if an episode is done for each agent.
        :param info: List of dictionaries containing information about each agent.
        :return: The index of the agent that has arrived at its destination, or None if none have.
        """
        for index, (d, agent_info) in enumerate(zip(done, info)):
            if d and agent_info.get('arrive_dest', False):
                return index
        return None


    def adjust_env_outputs(ref, r, done, info):
        """
        Adjust the environment outputs to remove agents that have reached their destination.

        :param ref: List of observations for each agent.
        :param r: List of rewards for each agent.
        :param done: List of done flags for each agent.
        :param info: List of info dictionaries for each agent.
        :return: Adjusted ref, r, done, info with agents that have reached destination removed.
        """
        # Filter out agents that have reached their destination
        active_agents_indices = [i for i, agent_info in enumerate(info) if not agent_info.get('arrive_dest', False)]

        # Adjust the outputs to only include active agents
        adjusted_ref = [ref[i] for i in active_agents_indices]
        adjusted_r = [r[i] for i in active_agents_indices]
        adjusted_done = [done[i] for i in active_agents_indices]
        adjusted_info = [info[i] for i in active_agents_indices]

        return adjusted_ref, adjusted_r, adjusted_done, adjusted_info


    while True:
        ref,r,done,info=my_env.step(action)
        print(done, len(ref), len(r), len(info))
        # Handle multi-agent environments
        new_agent_processed = False  # Flag to track if new agent's data is already used
        for agent_index, agent_done in enumerate(done):
            if agent_done:
                if info[agent_index].get('arrive_dest') and not new_agent_processed:
                    ref[agent_index] = ref[-1]
                    ref = np.delete(ref, -1, axis=0)
                    r = np.delete(r, -1)
                    done = np.delete(done, -1)
                    info = np.delete(info, -1)
                    new_agent_processed = True  # Mark that new agent's data has been used
                elif new_agent_processed:
                    # If another agent is done and the new agent's data is already used, skip processing
                    continue
                else:
                    # If the agent is done but hasn't reached its destination, reset the environment
                    ref = my_env.reset()
                    break  # Assuming the entire environment is reset

        print(done, len(ref), len(r), len(info))
        #'crash_vehicle': False, 'crash_object': False, 'crash_building': False,
        # 'out_of_road': False, 'arrive_dest': False, 'max_step': False, 'env_seed': 883, 'crash': False,
        # print(done,len(ref),len(r),len(info),info[0]["arrive_dest"],info[1]["arrive_dest"])
        print("_________________________________________________")

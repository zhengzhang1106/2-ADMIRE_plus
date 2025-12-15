import gym
from gym.spaces import Box, Dict
from gym.spaces import MultiDiscrete
import numpy as np
import Database
import RWA
import Service
import AuGraph


class AuGraphEnv(gym.Env):
    """
        自定义环境必须继承自gym.Env，并实现reset和step方法
        在方法__init__中，必须带第二个参数，用于传递envconfig
    """
    lightpath_cumulate = 0  # 当前光路数
    au_edge_list = []   # 存储RWA结果

    # 初始化
    def __init__(self, env_config):
        # # 5 个动作位（分别对应 5 类边）。每个位可选的绝对权重档位：
        # self.edge_types = ["GrmE", "LPE", "TxE", "WLE", "MuxE"]
        # self.action_levels = np.array([0, 5, 50, 200, 500, 1000], dtype=np.float32)
        # self.action_space = MultiDiscrete([len(self.action_levels)] * len(self.edge_types))

        # self.action_levels = {
        #     "GrmE": np.array([0, 10, 20, 40, 1000], dtype=np.float32),
        #     "LPE": np.array([0, 1, 2, 5], dtype=np.float32),
        #     "TxE": np.array([20, 50, 100, 200, 300], dtype=np.float32),
        #     "WLE": np.array([0, 5, 10, 20, 1000], dtype=np.float32),
        #     "MuxE": np.array([0, 1, 5, 10], dtype=np.float32),
        # }
        # self.action_space = MultiDiscrete([len(self.action_levels[t]) for t in self.edge_types])
        self.action_space = Box(low=np.zeros(5, dtype=np.float32), high=np.ones(5, dtype=np.float32), dtype=np.float32)
        self.observation_space = Dict({
            # 各边间的剩余容量
            'phylink': Box(low=-1*np.ones([2 *Database.link_number, Database.wavelength_number * Database.time]),
                           high=Database.wavelength_capacity*np.ones([2 * Database.link_number, Database.wavelength_number * Database.time]), dtype=np.float32),
            # 业务id,[0]
            'request_index': Box(low=np.array([0]), high=np.array([Database.job_number-1]), shape=(1,), dtype=np.int32),
            # 业务源、目的节点
            'request_src': Box(low=np.array([0]), high=np.array([Database.node_number-1]), shape=(1,), dtype=np.int32),
            'request_dest': Box(low=np.array([0]), high=np.array([Database.node_number-1]), shape=(1,), dtype=np.int32),
            # 业务流量
            'request_traffic': Box(low=np.zeros(Database.time), high=Database.wavelength_capacity * np.ones(Database.time), dtype=np.float32)
        })

        self.u = [0, 0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7]
        self.v = [1, 2, 4, 3, 3, 5, 6, 5, 7, 8, 8, 8]
        # 生成双向：把 (u,v) 与 (v,u) 拼起来得到 24 条
        self.u_dir = self.u + self.v
        self.v_dir = self.v + self.u
        # self.init()
        self.reset()

    # 还原环境
    def reset(self):
        """
        每完成一个episode,环境重新初始化
        :return: 返回环境初始化状态
        """
        print("reset")
        Service.generate_service(0, Database.time)  # 产生业务
        Database.clear(Database.links_physical)  # 清空物理链路
        AuGraph.links_virtual_list.clear()      # 清空虚拟链路
        index = 0
        links = self.linkmsg(Database.links_physical)
        src, dest, traffic = self.find_req_info(index)
        self.observation = {
            'phylink': links,
            'request_index': [index],
            'request_src': [src],
            'request_dest': [dest],
            'request_traffic': traffic
        }
        self.done = False
        self.step_num = 0
        AuGraphEnv.lightpath_cumulate = 0
        AuGraphEnv.job_success = 0
        AuGraphEnv.au_edge_list.clear()
        return self.observation

    # 步骤，更新环境，设置奖励
    # 所选动作作用于环境后环境返回的奖励和下一步状态，并判断是否达到终止状态
    # :param action:输入是动作的序号
    # :return:输出是：下一步状态，立即回报，是否终止，调试项
    def step(self, action) -> tuple:
        self.step_num += 1

        # idx = np.asarray(action, dtype=np.int64)
        # action_t = np.array([self.action_levels[i] for i in idx], dtype=np.float32)
        # action_t = np.array([self.action_levels[t][i] for t, i in zip(self.edge_types, idx)], dtype=np.float32)
        action_t = (np.asarray(action, dtype=np.float32) * 1000.0)

        request_index_current = self.observation['request_index'][0]  # 当前业务索引
        # 需要将动作参数（辅助图权重）传入辅助图初始化，然后进行路由，计算物理链路剩余带宽，作为奖励
        if request_index_current == 0:
            AuGraph.au_graph_init(action_t)  # 初始化权重
            flag, lightpath_num, au_edge_collection = RWA.route_wave_assign(action_t,request_index_current)  # flag表示是否选路成功，wave_used表示使用的波长（用于计算奖励）
        else:
            AuGraph.update_au_graph_weight(action_t)  # 用动作更新权重
            flag, lightpath_num, au_edge_collection = RWA.route_wave_assign(action_t, request_index_current)

        if flag:
            if request_index_current == Database.job_number - 1:  # 所有业务都部署完成，结束此次迭代
                self.done = True
                request_index = request_index_current
            else:
                request_index = request_index_current + 1  # 业务往后一个

            request_src, request_dest, request_traffic = self.find_req_info(request_index)
            phylinks = self.linkmsg(Database.links_physical)
            self.observation = {
                'phylink': phylinks,
                'request_index': [request_index],
                'request_src': [request_src],
                'request_dest': [request_dest],
                'request_traffic': request_traffic
            }
            reward = lightpath_num * (-1)
            AuGraphEnv.lightpath_cumulate += lightpath_num
            print('id', request_index_current, 'weight', action_t, "lightpath_cum", AuGraphEnv.lightpath_cumulate,
                  "lightpath_cur", lightpath_num, "reward", reward)

        else:  # 选路失败不更新物理网络状态和业务
            if request_index_current == Database.job_number - 1:  # 所有业务都部署完成，结束此次迭代
                self.done = True
                request_index = request_index_current
            else:
                request_index = request_index_current + 1

            request_src, request_dest, request_traffic = self.find_req_info(request_index)
            phylinks = self.linkmsg(Database.links_physical)
            self.observation = {
                'phylink': phylinks,
                'request_index': [request_index],
                'request_src': [request_src],
                'request_dest': [request_dest],
                'request_traffic': request_traffic
            }
            reward = -10
            print('id', request_index_current, 'weight', action_t, "lightpath_cum", AuGraphEnv.lightpath_cumulate, "lightpath_cur", lightpath_num, "reward", reward)

        AuGraphEnv.au_edge_list = au_edge_collection
        return self.observation, reward, self.done, {}  # 最后的return全返回了，如果没到最左边或者最右边，self.done=False


    def render(self, mode='human'):
        pass

    def find_req_info(self, index):
        src = Service.service_list[index]['src']
        dest = Service.service_list[index]['dest']
        traffic = Service.service_list[index]['traffic']
        # print('id',Service.service_list[index]['id'],' src',src,' ,dest',dest,' traffic',traffic[0])
        return src, dest, traffic

    # def linkmsg(self, links_physical):
    #     linkmsg = []
    #     row, col = Database.graph_connect.shape
    #     for i in range(row):
    #         for j in range(col):
    #             if Database.graph_connect[i][j] == 1:
    #                 linkmsg.append(links_physical[:, i, j].tolist())
    #     return linkmsg   # 正常应该是(24, 72)维的列表

    def linkmsg(self, links_physical):
        E = len(self.u_dir)  # 24
        F = Database.wavelength_number * Database.time  # 72
        arr = np.empty((E, F), dtype=np.float32)
        for k, (i, j) in enumerate(zip(self.u_dir, self.v_dir)):
            arr[k, :] = links_physical[:, i, j].astype(np.float32)  # 72 维
        return arr  # shape: (24, 72)


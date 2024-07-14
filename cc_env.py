import numpy as np


class Charge_Env:
    def __init__(
        self,
        bus_num: int,
        bus_vol: int,
        bus_cost_step: float,
        bus_cost_e: float,
        charge_step: float,
        charge_num: int,
        min_e: float,
        total_step: int,
        e_price_list: "list[float]",
        wait_num_list: "list[int]",
        punish_waiting: float,
        min_e_reward: float,
        charge_price_punish: float,
        free_charge_punish: float,
        people_remain_rate: float = 0.5,
    ) -> None:
        """_summary_

        Args:
            bus_num (int): 公交车数量
            bus_vol (int): 公交车容量
            bus_cost_step (float): 公交车行驶完一圈所消耗的步数
            bus_cost_e (float): 公交车行驶完一圈所消耗的电量
            charge_step (float): 充满电所需要的步数
            charge_num (int): 充电桩的数量
            min_e (float): 公交车所允许的最小电量
            total_step (int): 环境演绎的最大步数
            e_price_list (list[float]): 电价-时间表
            wait_num_list (list[int]): 人流-时间表
            punish_waiting (float): 每有一个人等待所施加的惩罚
            min_e_reward (float): 当电量不够时执行充电操作所给予的奖励
            charge_price_punish (float): 消耗电费所给予的惩罚系数
            free_charge_punish (float): 当电量不够时不执行充电操作所给予的惩罚
            people_remain_rate (float, optional): 等待人群的自然衰减率 Defaults to 0.5.
        """
        self.bus_num = bus_num
        self.bus_vol = bus_vol

        self.bus_cost_step = bus_cost_step
        self.bus_v = 1 / bus_cost_step

        self.bus_cost_e = bus_cost_e
        self.cost_e_speed = bus_cost_e / bus_cost_step

        self.charge_step = charge_step
        self.charge_speed = 1 / charge_step

        self.min_e = min_e
        self.charge_num = charge_num
        self.total_step = total_step
        self.e_price_list = e_price_list
        self.wait_num_list = wait_num_list

        self.state_dim = 2 * bus_num
        self.state = np.zeros((self.state_dim,))

        self.punish_waiting = punish_waiting
        self.min_e_reward = min_e_reward
        self.charge_price_punish = charge_price_punish
        self.free_charge_punish = free_charge_punish
        self.remain_rate = people_remain_rate

        self.now_step = 0
        self.waiting_person = 0
        self.reset()

    def reset(self):
        self.state = np.zeros((self.state_dim,))
        for i in range(self.bus_num):
            self.state[2 * i + 1] = 1
        self.now_step = 0
        self.waiting_person = 0
        return self.state.copy()

    def step(self, action_list: "list[int]"):
        reward_list = []
        comming_bus_num = 0
        now_charge_numn = 0
        for i in range(self.bus_num):
            rwd = 0
            bus_state = self.state[2 * i : 2 * (i + 1)].copy()
            bus_action = action_list[i]

            if bus_state[0] < 1e-3:
                if bus_action == 0:
                    if bus_state[1] < self.min_e:
                        rwd -= self.free_charge_punish
                    else:
                        self.state[2 * i] = 1 - self.bus_v
                        self.state[2 * i + 1] -= self.bus_cost_e
                        comming_bus_num += 1

                elif bus_action == 1:

                    if now_charge_numn < self.charge_num:
                        self.state[2 * i + 1] = min(
                            self.state[2 * i + 1] + self.charge_speed, 1
                        )
                        rwd -= (
                            self.charge_price_punish * self.e_price_list[self.now_step]
                        )
                        if bus_state[1] < self.min_e:
                            rwd += self.min_e_reward

                        now_charge_numn += 1
            else:
                self.state[2 * i] -= self.bus_v
                self.state[2 * i + 1] -= self.bus_cost_e
            reward_list.append(rwd)

        self.waiting_person = (
            self.wait_num_list[self.now_step] + self.waiting_person * self.remain_rate
        )
        self.waiting_person = max(
            0, self.waiting_person - comming_bus_num * self.bus_vol
        )
        wait_punish = self.waiting_person * self.punish_waiting / self.bus_num

        for i in range(self.bus_num):
            reward_list[i] -= wait_punish

        self.now_step += 1
        next_state = self.state.copy()
        return next_state, reward_list, self.now_step >= self.total_step

    @staticmethod
    def loadtxt(path: str):
        return np.loadtxt(path).tolist()

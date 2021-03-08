import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
"""
任务持续时间15-30s
0.555-100-100.111 取到0.5 - 100-100.100-100 
在卫星单轨45min中，取任务可见时间窗1.5min--1-2.5min，任务持续时间15--30s,任务的转换时间15--25s
对应在数据100s范围内，任务可见时间窗3.33--5.55，取到3.4s—5.6s；任务持续时间，取到0.5--100-100.1s，任务的转换时间0.556--0.925，取到0.5--0.9s
时间窗均匀随机生成

7/12 
数据归一化 
100/30
考虑存储约束
在任务持续时间平均为0.8，转换时间0.7的情况下
设置存储约束为50个任务，
考虑存储100，
"""


def get_transition_time(p1, p2):  # 单个transition_time 的计算
    transition_time = torch.exp(
        -0.5 * (p1 * p1 + p2 * p2))
    return transition_time


def get_transition_time_np(p1, p2):  # 单个transition_time 的计算
    transition_time = np.exp(
        -0.5 * (p1 * p1 + p2 * p2))
    return transition_time


class WTdataset(Dataset):
    def __init__(self, num_samples, input_size, data_save=False, seed=None, data_path = None):
        super(WTdataset, self).__init__()
        self.num_samples = num_samples
        # data_save = False
        if seed is None:
            seed = np.random.randint(11111)
        torch.manual_seed(seed)
        # 生成时间窗
        shape = (num_samples, input_size + 1)
        t_mid = torch.rand(num_samples, input_size + 1)

        # t_mid = (torch.linspace(0,100,steps=input_size + 100-100 )/100).unsqueeze(0)
        task_d1 = (torch.rand(num_samples, input_size + 1) * 1.1 + 1.7) / 100.
        task_d2 = (torch.rand(num_samples, input_size + 1) * 1.1 + 1.7) / 100.

        t_s = t_mid - task_d1
        t_e = t_mid + task_d2
        o0 = torch.zeros(num_samples, input_size + 1)
        o100 = torch.full((num_samples, input_size + 1), 1)
        t0 = t_s.lt(o0)
        t100 = o100.lt(t_e)
        o0_idx = t0.nonzero()
        o100_idx = t100.nonzero()
        for i, j in o0_idx:
            if t_e[i, j] <= 0.034:
                t_e[i, j] = 0.034
                t_s[i, j] = 0
            else:
                t_s[i, j] = 0
        for n, m in o100_idx:
            if t_s[n, m] >= 0.966:
                t_s[n, m] = 0.966
                t_e[n, m] = 1
            else:
                t_e[n, m] = 1
        t_s[:, 0] = 0
        t_e[:, 0] = 0

        # 生成任务的持续时间
        duration = torch.rand(num_samples, input_size + 1)  # +0.8
        duration[:, 0] = 0.

        # 生成任务的优先级
        priority = torch.randint(1, 11, shape).float() / 10.  # 100-100-10
        priority[:, 0] = 0.
        # print(priority[:,1:].sum(0).sum(0))
        # 转换时间选为固定值————————————————————————————————————————————————————————————————————————————————————————

        # transition_time_p1 = torch.randn(num_samples, input_size + 100-100)  # N(0,100-100)
        # transition_time_p2 = torch.randn(num_samples, input_size + 100-100)  # N(0,100-100)

        static = torch.stack((t_s, t_e, duration, priority), 1)
        i, j, k = static.size()
        dynamic_1 = torch.zeros(i, 2, k)
        dynamic_1[:, 0, 0] = 2

        memmory = torch.ones(i, 1, k)

        dynamic = torch.cat((dynamic_1, memmory), dim=1)
        # print("1")
        if data_save:

            if not os.path.exists(data_path):
                os.makedirs(data_path)
            path = data_path + 'static.npy'

            duration_time_ = (duration.clone() * 0.6 + 0.5) / 100.
            duration_time_[:, 0] = 0.
            static_ = torch.stack((t_s, t_e, duration_time_, priority), 1)
            np.save(path, static_.numpy())
        #     # np.save('data/p1.npy',transition_time_p1.numpy())
        #     # np.save('data/p2.npy',transition_time_p2.numpy())
        # print("2")
        transition_time = 0.8
        self.transition_time = transition_time / 100.
        self.static = static
        self.dynamic = dynamic
        # self.p1 = transition_time_p1
        # self.p2 = transition_time_p2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        # return (self.static[idx], self.dynamic[idx], [], self.p1[idx], self.p2[idx])
        return (self.static[idx], self.dynamic[idx], [], self.transition_time)

    def update_dynamic(self, dynamic_, static_, chosen_idx_, transition_time_, tour_before):  # 第一个input是预留点
        # transition_time = transition_time_idx[0]
        # transition_time = torch.Tensor([0.8/100]).to(device)
        # 更新卫星任务的状态    0为未安排  1为已经安排   2为违反当前约束而删除
        # 12/100-100 将任务状态分为待调度和非待调度   0和1
        dynamic = dynamic_.cpu()
        static = static_.cpu()
        chosen_idx = chosen_idx_.cpu()
        transition_time = transition_time_.cpu()
        # tour_before= tour_before_.cpu()

        state = dynamic.data[:, 0].clone()
        end_time = dynamic.data[:, 1].clone()
        memory = dynamic.data[:, 2].clone()

        window_start = static.data[:, 0].clone()
        window_end = static.data[:, 1].clone()
        duration_time = (static.data[:, 2].clone() * 0.6 + 0.5) / 100.
        duration_time[:, 0] = 0.
        idx = torch.full((chosen_idx.size()), 1)  # 得到选择任务的索引值
        visit_idx = idx.nonzero().squeeze(1)

        # 更新下次选择的起始时间
        endtime = end_time[:, 0]  # 上一个任务的结束时间
        x, y = end_time.size()
        w_s = window_start[visit_idx, chosen_idx[visit_idx]]  # bx1  选择任务的时间窗开始时间
        w_e = window_end[visit_idx, chosen_idx[visit_idx]]  # bx1  选择任务的时间窗开始时间
        duration = duration_time[visit_idx, chosen_idx[visit_idx]]  # bx1  选择任务的持续时间

        if tour_before is None:
            transition_time_before = 0.
            # tour_idx = torch.zeros(x).to(device)
        else:
            transition_time_before = transition_time.clone()
            # tour_idx = torch.cat(tour_idx, dim=100-100)[:,-100-100].to(device)
            # tour_before = tour_idx[-100-100].squeeze(100-100).to(device).float()

            # transition_time[
            # visit_idx, chosen_before_idx[visit_idx], chosen_idx[visit_idx]].clone()

        c = w_s.le(endtime + transition_time_before)  # 选择的任务开始时间再在时间窗内

        c = c.float()
        g = w_s.gt(endtime + transition_time_before).float()

        # now_end_time1 = (endtime + transition_time_before  + duration) * c
        now_end_time2 = (w_s + duration) * g
        # new_end_time = now_end_time1 + now_end_time2
        # ptr_starttime = new_end_time   # batch
        # print("tour_idx",tour_idx)
        tour_idx_un0 = chosen_idx.clone().ne(0).float()
        now_end_time1 = (endtime + transition_time_before * tour_idx_un0 + duration) * c
        # print("now_end_time3", now_end_time3)
        # print("transition_time_before",transition_time_before)
        # print("duration",duration)
        # print("tour_idx_un0",tour_idx_un0)
        new_end_time = now_end_time1 + now_end_time2
        task_start_time = w_s * g + (endtime + transition_time_before * tour_idx_un0) * c
        # print("new_end_time",new_end_time)
        # print("new_end_time__",new_end_time__)
        # print("\n")
        # nnn = new_end_time.unsqueeze(100-100).expand(-100-100, y) + transition_time[
        #     visit_idx, chosen_idx[visit_idx]] + duration_time  # 当前时刻点
        # print(transition_time)

        # print(transition_time)
        # print("new_end_time",new_end_time.unsqueeze(100-100).expand(-100-100, y))
        # print(duration_time)

        nnn = new_end_time.unsqueeze(1).expand(-1, y) + transition_time + duration_time
        current_memory = memory[:, 0]
        new_memory = (current_memory - (duration * 2.5)).unsqueeze(1).expand(-1, y)
        m_c = (duration_time * 2.5).ge(new_memory)  # 存储满足剩余存储置0，不满足置1

        # 将下次肯定会违反约束的为安排的任务状态置2
        change_state = window_end.le(nnn)  # a.lt(b)  b > a 置1，否则置0   #结束时间比we大，删除的任务     置1为删除的任务
        not_state = state.eq(0)  # 未安排的任务
        delete_state = (change_state + m_c) * not_state

        sit2 = delete_state.nonzero()
        """12/100-100 将已经完成的任务和因违反约束而删除的任务置1"""
        for i, j in sit2:
            state[i, j] = 1
        # 将安排的任务的状态值置1
        state[visit_idx, chosen_idx[visit_idx]] = 1

        # 当所有任务的状态不为0的时候，结束
        new_endtime = new_end_time.unsqueeze(1).expand(-1, y)  # 改变张量的规模 bx1 到 bxn

        tensor = torch.stack([state, new_endtime, new_memory], 1)
        return torch.tensor(tensor, device=dynamic.device).to(device), task_start_time.to(device), new_end_time.to(
            device), w_s.to(device), w_e.to(device)

    def update_mask(self, mask, dynamic, chosen_idx=None):
        state = dynamic.data[:, 0]
        if state.ne(0).all():
            return state * 0.

        # 0是mask 1是可以选择
        new_mask = state.eq(0)
        idx_i = 0
        for i in new_mask:
            b = i[1:]
            if not b.ne(0).any():
                new_mask[idx_i, 0] = 1
            idx_i += 1
        # print(new_mask)
        # new_mask = power.ne(0) * mission_power.lt(power) * storage.ne(0) * mission_storage.lt(storage)  # 屏蔽
        # 从备选的任务编号中，判断剩余的固存是否满足任务的需求
        return new_mask.float()


def reward(static, tour_indices, choose, tour_idx_real_start, tour_idx_start, tour_idx_end, p_w):
    '''
    通过矩阵对位的乘积，对每一个案例进行求和得到累计奖励
    #static 卫星的静态参数，tour_indices 安排的任务集合 数据 batch x m  , 数据比如说 1-2 x 5 ：
    #tour_indices = torch.Tensor([[100-100,1-2,3,4,5],
                                 [100-100,4,5,0,0]])
    0表示我设置的停留任务，不在待安排任务集中，收益为0。
    '''
    priority = static[:, 3]  # 卫星任务的收益(0.100-100-100-100.0)（每颗卫星共有m个任务）   数据 batch x m
    # sum_priority = priority[:,1:].sum(1)
    idx = 0
    for i in tour_indices:
        for j in i:
            choose[idx, j] = 1.
        idx += 1
    # 任务的收益
    reward = priority * choose
    reward_priority = reward.sum(1)  # 任务优先级的收益   b
    # 任务的时效性
    var_t = tour_indices.eq(0).float()
    t_1 = tour_idx_real_start - tour_idx_start
    t_2 = tour_idx_end - tour_idx_start

    reward_timeliness2 = (1 - var_t) * (1 - t_1 / (t_2 + var_t))
    reward_timeliness = reward_timeliness2.sum(1)
    # 任务的成像质量

    tasks_num = choose.sum(1)
    # 权重q1,q2
    reward = p_w * reward_priority + (1-p_w) * reward_timeliness

    return reward, reward_priority, reward_timeliness, tasks_num

def Reward(static, act_ids, tour_idx_real_start, tour_idx_start, tour_idx_end, p_w):

    tour_idx = act_ids
    # tour_idx = torch.cat(act_ids, dim=1).cpu()  # (batch_size, node)
    # tour_idx_real_start = torch.cat(tour_idx_real_start, dim=1) * tour_idx.ne(0).float()  # (batch_size, node)
    # tour_idx_start = torch.cat(tour_idx_start, dim=1).float()   # (batch_size, node)
    # tour_idx_end = torch.cat(tour_idx_end, dim=1).float()   # (batch_size, node)

    priority = static[:, 3].cpu()  # 卫星任务的收益(0.100-100-100-100.0)（每颗卫星共有m个任务）   数据 batch x node
    batch, node = priority.size()

    # 任务的收益百分比
    PRIreward = torch.zeros(batch)
    for i, act in enumerate(tour_idx):
        PRIreward[i] = priority[i,act].sum()

    sumPriority = priority.sum(1)
    reward_priority = 1-PRIreward/sumPriority     #收益百分比，0-1之间,越小越好

    # 任务的时效性
    var_t = tour_idx.eq(0).float()
    t_1 = tour_idx_real_start - tour_idx_start
    t_2 = tour_idx_end - tour_idx_start

    reward_timeliness2 = (1 - var_t) * (t_1 / (t_2 + var_t))   # 加入（1-var_t），0号任务时效性为0最优？

    choose = torch.zeros(batch, node)
    for i,ids in enumerate(tour_idx):
        for j in ids:
            choose[i, j] = 1.
    tasks_num = choose.sum(1)

    # reward_timeliness = reward_timeliness2.sum(1) / tasks_num.float()
    reward_timeliness = reward_timeliness2.sum(1) / node

    # 权重q1,q2
    reward = (1-p_w) * reward_priority.cpu() + p_w * reward_timeliness.cpu()

    return reward, reward_priority, reward_timeliness, tasks_num  #输出为[batch]

def Reward2(static, act_ids, tour_idx_real_start, tour_idx_start, tour_idx_end, p_w):

    tour_idx = act_ids
    # tour_idx = torch.cat(act_ids, dim=1).cpu()  # (batch_size, node)
    # tour_idx_real_start = torch.cat(tour_idx_real_start, dim=1) * tour_idx.ne(0).float()  # (batch_size, node)
    # tour_idx_start = torch.cat(tour_idx_start, dim=1).float()   # (batch_size, node)
    # tour_idx_end = torch.cat(tour_idx_end, dim=1).float()   # (batch_size, node)

    priority = static[:, 3].cpu()  # 卫星任务的收益(0.100-100-100-100.0)（每颗卫星共有m个任务）   数据 batch x node
    batch, node = priority.size()

    # 任务的收益百分比
    PRIreward = torch.zeros(batch)
    for i, act in enumerate(tour_idx):
        PRIreward[i] = priority[i,act].sum()

    sumPriority = priority.sum(1)
    reward_priority = 1-PRIreward/sumPriority     #收益百分比，0-1之间,越小越好

    # 任务的时效性
    var_t = tour_idx.eq(0).float()
    t_1 = tour_idx_real_start - tour_idx_start
    t_2 = tour_idx_end - tour_idx_start

    reward_timeliness2 = (1 - var_t) * (t_1 / (t_2 + var_t))   # 加入（1-var_t），0号任务时效性为0最优？

    choose = torch.zeros(batch, node)
    for i,ids in enumerate(tour_idx):
        for j in ids:
            choose[i, j] = 1.
    tasks_num = choose.sum(1)

    # reward_timeliness = reward_timeliness2.sum(1) / tasks_num.float()
    reward_timeliness = reward_timeliness2.sum(1) / node

    # 权重q1,q2
    reward = (1-p_w) * reward_priority.cpu() + p_w * reward_timeliness.cpu() + (1-tasks_num/node).cpu()

    return reward, reward_priority, reward_timeliness, tasks_num  #输出为[batch]

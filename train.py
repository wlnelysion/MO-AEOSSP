import os
import time
import xlwt
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
# from model import Encoder, DRL4SSP
# import layers
from simple_ass import Encoder_Embedding, DRL4SSP
from tasks.sat import WTdataset
#from tasks.sat import reward, Reward, Reward2
from tasks.sat import Reward as Reward

# import heuristic

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_transition_time(p1, p2):  # 单个transition_time 的计算
    transition_time = torch.exp(
        -0.5 * (p1 * p1 + p2 * p2))
    return transition_time


class StateCritic(nn.Module):
    # 定义Critic网络
    # 评价网络的复杂度
    """
    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder_Embedding(static_size, hidden_size)  # fliter2 x 1114  128个
        self.dynamic_encoder = Encoder_Embedding(dynamic_size, hidden_size)  # fliter1 x 1114  128个
        self.p_encoder = Encoder_Embedding(2, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)  # fliter256 x 1114  20个
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)  # fliter20 x 1114   20个
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)  # fliter20 x 1114    1个

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        static_hidden = self.static_encoder(static)  # 128维向量
        dynamic_hidden = self.dynamic_encoder(dynamic)  # 128维向量

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)  # 横向连接到一起      [s d]

        output = F.relu(self.fc1(hidden))  # relu（20维向量
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output


def validate(data_loader, actor, reward_fn, p_w, test):
    """Used to monitor progress on a validation set & optionally plot solution."""
    # print("validate")
    # print("p_w: validate:", p_w)
    # torch.cuda.empty_cache()
    actor.eval()

    rewards, rewards_priority, rewards_timeliness = [], [], []
    tasks_num = []
    time_ = 0.
    # reward_max = 0.
    # reward_min = 1000.
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0, transition_time_idx = batch
        transition_time = transition_time_idx[0]
        num_samples, j, input_size = static.size()

        choose = torch.zeros(num_samples, input_size).to(device)
        static = static.to(device)
        dynamic = dynamic.to(device)

        x0 = x0.to(device) if len(x0) > 0 else None

        time1 = time.time()
        with torch.no_grad():
            tour_indices, _, tour_idx_real_start, tour_idx_start, tour_idx_end = actor.forward(
                static, dynamic, transition_time)

        # reward = reward_fn(static, tour_indices, choose).mean().item()
        time2 = time.time()
        time_ += time2 - time1
        # print("validate")
        reward, reward_priority, reward_timeliness, task_num = reward_fn(static, tour_indices,
                                                                         tour_idx_real_start,
                                                                         tour_idx_start,
                                                                         tour_idx_end, p_w)
        if not test:
            reward = reward.mean().item()
            reward_priority=reward_priority.mean().item()
            reward_timeliness = reward_timeliness.mean().item()
        rewards.append(reward)
        rewards_priority.append(reward_priority)
        rewards_timeliness.append(reward_timeliness)
        tasks_num.append(task_num)

    if test:
        return rewards, rewards_priority, rewards_timeliness, time_, tasks_num
    else:
        return np.mean(rewards), np.mean(rewards_priority), np.mean(rewards_timeliness), time_


# 训练网络
def train(model_path, actor, critic, priority_weight, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,
          **kwargs):
    """Constructs the main actor & critic netw orks, and performs all training."""

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    actor_scheduler = lr_scheduler.MultiStepLR(actor_optim, range(5000, 5000 * 1000, 5000), gamma=float(0.96))
    print('优化器构造完毕')

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_params = None
    best_reward = 1

    #根据权重列表循环训练并保存
    print('开始按权重个数循环训练模型')
    for i, p_w in enumerate(priority_weight):

        model_save_dir = model_path + str(int(p_w*1000))

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        if i==0 :
            epoch_num = 5
        else:
            epoch_num = 2

        print('--------------------------------权重为：', p_w, '训练开始------------------------------------------------')
        #每个模型训练10个epoch
        for epoch in range(epoch_num):

            #torch.cuda.empty_cache()

            actor.train()
            critic.train()

            times, losses, rewards, critic_rewards, tasks_nums = [], [], [], [], []
            reward_prioritys, reward_timelinesss = [], []

            epoch_start = time.time()
            start = epoch_start

            for batch_idx, batch in enumerate(train_loader):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标

                static, dynamic, x0, transition_time_idx = batch  # datasfile":('...','...','...')

                transition_time = transition_time_idx[0]

                num_samples, j, input_size = static.size()
                choose = torch.zeros(num_samples, input_size).to(device)
                static = static.to(device)
                dynamic = dynamic.to(device)

                tour_indices, tour_logp, tour_idx_real_start, tour_idx_start, tour_idx_end = actor(
                    static, dynamic, transition_time)

                # Sum the log probabilities for each city in the tour
                # reward = reward_fn(static, tour_indices)
                reward, reward_priority, reward_timeliness, tasks_num = reward_fn(static, tour_indices,
                                                                                  tour_idx_real_start,
                                                                                  tour_idx_start,
                                                                                  tour_idx_end, p_w)

                # Query the critic for an estimate of the reward
                critic_est = critic(static, dynamic).view(-1)

                reward = reward.to(device)
                advantage = reward-critic_est

                actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
                critic_loss = torch.mean(advantage ** 2)

                actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optim.step()

                actor_scheduler.step()

                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optim.step()

                critic_rewards.append(torch.mean(critic_est.detach()).item())
                rewards.append(torch.mean(reward.detach()).item())
                tasks_nums.append(torch.mean(tasks_num.detach()).item())
                losses.append(torch.mean(actor_loss.detach()).item())

                reward_prioritys.append(torch.mean(reward_priority.detach()).item())
                reward_timelinesss.append(torch.mean(reward_timeliness.detach()).item())
                # print(len(losses))
                # print("losses")
                if (batch_idx + 1) % 100 == 0:
                    end = time.time()
                    times.append(end - start)
                    start = end

                    mean_loss = np.mean(losses[-100:])
                    mean_reward = np.mean(rewards[-100:])
                    mean_task_num = np.mean(tasks_nums[-100:])

                    mean_pri = np.mean(reward_prioritys[-100:])
                    mean_time = np.mean(reward_timelinesss[-100:])

                    print('  Batch %d/%d, reward: %2.3f, pri: %2.3f, timeline: %2.3f, loss: %2.4f, task_num: %d, time: %2.4fs' %
                          (batch_idx, len(train_loader), mean_reward, mean_pri, mean_time, mean_loss, mean_task_num,
                           times[-1]))

            mean_loss = np.mean(losses)
            mean_reward = np.mean(rewards)

            mean_valid, _1, _2, t_ = validate(valid_loader, actor, reward_fn, p_w, False)

            model_name = str(epoch) + 'actor.pt'
            # 保存Actor网络的参数
            save_path = os.path.join(model_save_dir, model_name)
            torch.save(actor.state_dict(), save_path)

            print('%2.4f Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs ' \
                  '(%2.4fs / 100 batches)\n' % \
                  (epoch, mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
                   np.mean(times)))


        print('--------------------------------权重为：', p_w, '训练完毕------------------------------------------------')

# 对模型的训练
def train_sat(args, priority_weight, model_path, load_path, data_path):


    # Determines the maximum amount of load for a vehicle based on num nodes

    STATIC_SIZE = 4
    DYNAMIC_SIZE = 3

    #构造训练数据
    train_data = WTdataset(args.train_size,
                           args.num_nodes,
                           False,
                           args.seed)

    #构造验证数据
    test_data = WTdataset(args.test_size,
                           args.num_nodes,
                           True,
                           args.seed + 1,
                           data_path)
    print('训练、测试数据构造完毕')

    #构造A网络
    actor = DRL4SSP(args.num_nodes + 1, STATIC_SIZE, args.hidden_size, DYNAMIC_SIZE, args.num_layers, args.n_head,
                    args.n_layers, args.k_dim,
                    args.v_dim, args.const_local, train_data.update_dynamic, train_data.update_mask, args.dropout,
                    ).to(device)

    #构造C网络
    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)
    print('A、C网络初始化完毕')

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = test_data
    kwargs['reward_fn'] = Reward
    kwargs['render_fn'] = None  # sat.render

    # 中断后恢复训练节点
    if load_path:
        path = os.path.join(load_path, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))
        path = os.path.join(load_path, 'critic.pt')
        actor.load_state_dict(torch.load(path, device))
        print('A、C断点回复完毕')

    train(model_path, actor, critic, priority_weight, **kwargs)  #所有模型存储完毕


# 对模型的训练
def test_sat(sheet, args, priority_weight, model_path, data_path):
    # Determines the maximum amount of load for a vehicle based on num nodes

    STATIC_SIZE = 4
    DYNAMIC_SIZE = 3


    # 构造验证数据
    # test_data = WTdataset(10,
    #                       args.num_nodes,
    #                       True,
    #                       args.seed + 1,
    #                       data_path)

    test_data = WTdataset(args.test_size,
                          args.test_nodes,
                          True,
                          args.seed + 1,
                          data_path)

    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    print('测试集读取完毕')

    # 构造A网络
    actor = DRL4SSP(args.num_nodes + 1, STATIC_SIZE, args.hidden_size, DYNAMIC_SIZE, args.num_layers, args.n_head,
                    args.n_layers, args.k_dim,
                    args.v_dim, args.const_local, test_data.update_dynamic, test_data.update_mask, args.dropout,
                    ).to(device)


    model_list = []
    for file in os.listdir(model_path):
        file = os.path.join(model_path, file)
        for model in os.listdir(file):
            if model.endswith('actor.pt'):
                file_path = os.path.join(file, model)
                model_list.append(file_path)

    print(model_list[0])

    #读取不同权重模型参数
    # for i, p_w in enumerate(priority_weight):

        # if i == 11:
        #     break
    for i, m in enumerate(model_list):
        print(m)
        #p_w = int(m.split('/')[-1].split('\\')[0])/1000
        p_w = int(m.split('/')[-2]) / 1000

        print('--------------------------------权重为：', p_w, '测试开始------------------------------------------------')

        # model_save_dir = model_path + str(int(p_w*1000))
        # path = os.path.join(model_save_dir, 'actor.pt')
        actor.load_state_dict(torch.load(m, device))

        out, out_priority, out_timeless, time1, tasks_num = validate(test_loader, actor, Reward, p_w, args.test)

        #每个模型的测试结果写入xls
        for j in range(out[0].size(0)):
            # if j /10 ==0:
            #     out_put_name = out_put_name_ + str(int(j/10))
            sheet.write(0, 0 + 8 * j, 'instance' + str(int(j)))
            sheet.write(0, 1 + 8 * j, '加权目标值')
            sheet.write(0, 2 + 8 * j, '任务优先级')
            sheet.write(0, 3 + 8 * j, '时效性')
            sheet.write(0, 4 + 8 * j, '时间')
            sheet.write(0, 5 + 8 * j, '完成任务数量')
            sheet.write(0, 6 + 8 * j, '模型路径')

            sheet.write(i + 1, 0 + 8 * j, 'priority_weight' + str(int(p_w*1000)))
            sheet.write(i + 1, 1 + 8 * j, float(out[0].cpu().numpy()[j]))
            sheet.write(i + 1, 2 + 8 * j, float(out_priority[0].cpu().numpy()[j]))
            sheet.write(i + 1, 3 + 8 * j, float(out_timeless[0].cpu().numpy()[j]))
            sheet.write(i + 1, 4 + 8 * j, time1)
            sheet.write(i + 1, 5 + 8 * j, float(tasks_num[0].cpu().numpy()[j]))
            sheet.write(i + 1, 6 + 8 * j, '0531-2-test/'+m)

        print('--------------------------------权重为：', p_w, '测试完毕------------------------------------------------')
    # print('RL-AEOSS Average rewards: ', out)
    # print('RL-AEOSS Average priority rewards: ', out_priority)
    # print('RL-AEOSS Average timeless rewards: ', out_timeless)
    # print('RL-AEOSS Total times: ', time1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=66, type=int)

    parser.add_argument('--n_head', default=8, type=int)
    parser.add_argument('--n_layers', default=3, type=int)
    parser.add_argument('--k_dim', default=256, type=int)
    parser.add_argument('--v_dim', default=256, type=int)
    parser.add_argument('--const_local', default=32, type=int)
    parser.add_argument('--task', default='sat')

    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--max_grad_norm', default=3., type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
#改造后
    parser.add_argument('--train_size', type=int, default=200000)
    parser.add_argument('--test_size', type=int, default=30)
    parser.add_argument('--nodes', dest='num_nodes', type=int, default=50)
    parser.add_argument('--test_nodes', type=int, default=200)

    parser.add_argument('--test', action='store_true', default=True)
    parser.add_argument('--load_path', type=str, default=None)

    parser.add_argument('--num_model', type=int, default=100)
    parser.add_argument('--model_save_path', type=str, default='model/')
    parser.add_argument('--data_save_path', type=str, default='data/')

    args = parser.parse_args()

    #测试入口
    if args.test:
        workbook = xlwt.Workbook()
        out_put_name = "0618-modle0608-50-data0608-200"
        sheet = workbook.add_sheet(out_put_name, cell_overwrite_ok=True)


        #计算所有多目标权重
        priority_weight = []
        for i in range(0, 1001, int(1001/args.num_model)):
            priority_weight.append(i/1000)

        num_model = len(priority_weight)
        print('模型个数=从0到1000取权重个数=', num_model)

        model_path = args.model_save_path + str(args.num_nodes) + '/' + str(num_model) + '/'
        data_path = args.data_save_path + str(args.test_size) + '/' + str(args.test_nodes) + '/'

        test_sat(sheet, args, priority_weight, model_path, data_path)
        print('测试完毕')

        workbook.save(out_put_name + '.xls')
        print('输出结果完毕')

    else:     #训练入口

        #计算所有多目标权重
        priority_weight = []
        for i in range(0, 1001, int(1001/args.num_model)):
            priority_weight.append(i/1000)

        num_model = len(priority_weight)
        print('模型个数=从0到1000取权重个数=', num_model)

        model_path = args.model_save_path + str(args.num_nodes) + '/' + str(num_model) + '/'
        data_path = args.data_save_path + str(args.test_size) + '/' + str(args.num_nodes) + '/'
        train_sat(args, priority_weight, model_path, args.load_path, data_path)

        print('训练完毕')


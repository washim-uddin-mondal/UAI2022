import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.categorical import Categorical
import os
import copy


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super(Actor, self).__init__()
        self.state_size = 2*state_size  # one-hot state + mean-distribution
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(self.state_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, state, state_dist):
        state_joined = torch.cat([state, state_dist])
        output = F.relu(self.linear1(state_joined))
        output = F.relu(self.linear2(output))
        output = F.softmax(self.linear3(output), dim=-1)
        return output


def train(args):
    actor = Actor(args.num_states, args.num_actions, args.hidden_size)
    NumActParam = 2*args.num_states * args.hidden_size + args.hidden_size + args.hidden_size**2 + args.hidden_size + args.hidden_size*args.num_actions + args.num_actions
    optimizer = optim.Adam(list(actor.parameters()))

    # Floating point representation of states
    states_float = torch.tensor(range(0, args.num_states)).float()

    for j in range(args.J):

        if j % args.update_interval == 0:
            args.logger.info(f'training iteration: {j}')

        w = torch.zeros(NumActParam)
        w_avg = torch.zeros(NumActParam)

        for _ in range(args.L):

            # Initial state distribution
            curr_state_dist = torch.ones(args.num_states) / args.num_states
            curr_state = Categorical(curr_state_dist).sample().long()

            """ ------------ Sampling (x, mu, u) ------------ """
            FLAG = False
            while not FLAG:
                if torch.rand(1) > args.gamma:
                    FLAG = True
                """ --------- Update Subroutine -------------- """

                """ ------------ Current State ------------------- """
                curr_state_one_hot = torch.zeros(args.num_states)
                curr_state_one_hot[curr_state] = 1

                """ ------------- Mean of Current State Distribution ------------- """
                curr_state_dist_mean = torch.dot(states_float, curr_state_dist)

                """ ------------- Current Action ------------------ """
                policy = Categorical(actor(curr_state_one_hot, curr_state_dist))
                curr_action = policy.sample().long()

                """ ------------- Next State --------------- """
                fraction = 1 - (curr_state_dist_mean/(args.num_states-1))
                if curr_action == 0:
                    next_state = curr_state
                else:
                    chi = torch.rand(1)
                    next_state = curr_state + (chi * fraction * (args.num_states - 1 - curr_state)).long()
                    next_state_one_hot = torch.zeros(args.num_states)
                    next_state_one_hot[next_state] = 1

                """ -------------- Next State Distribution ------------- """

                next_state_dist = torch.zeros(args.num_states)
                for state_t in range(0, args.num_states):
                    one_hot_state_t = torch.zeros(args.num_states)
                    one_hot_state_t[state_t] = 1

                    for action_t in range(0, args.num_actions):
                        dist_vec = torch.zeros(args.num_states)
                        if action_t == 0:
                            dist_vec[state_t] = 1
                        else:
                            prob_mass = 1/(fraction * (args.num_states - 1 - state_t))
                            total_prob = torch.tensor(1.0)
                            state_t_plus_1 = state_t
                            while total_prob > 0 and state_t_plus_1 < args.num_states:
                                dist_vec[state_t_plus_1] = torch.minimum(prob_mass, total_prob)
                                total_prob -= torch.minimum(prob_mass, total_prob)
                                state_t_plus_1 += 1

                        prob = actor(one_hot_state_t, curr_state_dist)[action_t] * curr_state_dist[state_t]
                        next_state_dist += dist_vec * prob

                """ --------------------- Update ------------------ """
                curr_state = copy.copy(next_state)
                curr_state_dist = copy.copy(next_state_dist)

            """ ------------ Sampling Advantage Functions ---------- """
            FLAG = False
            SumRewards = torch.tensor([0.])

            while not FLAG:
                if torch.rand(1) > args.gamma:
                    FLAG = True
                """ --------- Update Subroutine -------------- """

                """ ------------ Current State ------------------- """
                curr_state_one_hot = torch.zeros(args.num_states)
                curr_state_one_hot[curr_state] = 1

                """ ------------- Mean of Current State Distribution ------------- """
                curr_state_dist_mean = torch.dot(states_float, curr_state_dist)

                """ ------------- Current Action ------------------ """
                policy = Categorical(actor(curr_state_one_hot, curr_state_dist))
                curr_action = policy.sample().long()

                """ ------------- Next State --------------- """
                fraction = 1 - (curr_state_dist_mean/(args.num_states-1))
                if curr_action == 0:
                    next_state = curr_state
                else:
                    chi = torch.rand(1)
                    next_state = curr_state + (chi * fraction * (args.num_states - 1 - curr_state)).long()
                    next_state_one_hot = torch.zeros(args.num_states)
                    next_state_one_hot[next_state] = 1

                """ -------------- Next State Distribution ------------- """

                next_state_dist = torch.zeros(args.num_states)
                for state_t in range(0, args.num_states):
                    one_hot_state_t = torch.zeros(args.num_states)
                    one_hot_state_t[state_t] = 1

                    for action_t in range(0, args.num_actions):
                        dist_vec = torch.zeros(args.num_states)
                        if action_t == 0:
                            dist_vec[state_t] = 1
                        else:
                            prob_mass = 1/(fraction * (args.num_states - 1 - state_t))
                            total_prob = torch.tensor(1.0)
                            state_t_plus_1 = state_t
                            while total_prob > 0 and state_t_plus_1 < args.num_states:
                                dist_vec[state_t_plus_1] = torch.minimum(prob_mass, total_prob)
                                total_prob -= torch.minimum(prob_mass, total_prob)
                                state_t_plus_1 += 1

                        prob = actor(one_hot_state_t, curr_state_dist)[action_t] * curr_state_dist[state_t]
                        next_state_dist += dist_vec * prob

                """ -------------- SumRewards Update ---------- """
                SumRewards += args.alpha_r * curr_state - args.beta_r * (curr_state_dist_mean ** args.sigma) - args.lambda_r * curr_action

                """ --------------------- Update ------------------ """
                curr_state = copy.copy(next_state)
                curr_state_dist = copy.copy(next_state_dist)

            Value_R = 0
            Q_R = 0

            if torch.rand(1) < 0.5:
                Value_R = SumRewards
            else:
                Q_R = SumRewards

            Advantage_R = 2*(Q_R-Value_R)

            # Gradient Update for the Sub-Problem
            log_prob = policy.log_prob(curr_action)
            optimizer.zero_grad()
            log_prob.backward()

            phi_grads = []
            for f in actor.parameters():
                phi_grads.append(f.grad.view(-1))
            phi_grads = torch.cat(phi_grads)

            h_grads = (torch.dot(w, phi_grads)-Advantage_R)*phi_grads

            w = w - args.alpha * h_grads
            w_avg += w/args.L

        count = 0
        for phi in actor.parameters():
            phi.data -= (args.eta/(1-args.gamma))*w_avg[count]
            count += 1

    if not os.path.exists('Models'):
        os.mkdir('Models')
    torch.save(actor.state_dict(), f'Models/Actor{args.sigma}.pkl')


def evaluateMFC(args):
    actor = Actor(args.num_states, args.num_actions)

    if not os.path.exists(f'Models/Actor{args.sigma}.pkl'):
        raise ValueError('Model does not exist.')
    actor.load_state_dict(torch.load(f'Models/Actor{args.sigma}.pkl'))

    # Initial state distribution
    curr_state_dist = torch.ones(args.num_states)/args.num_states

    # Floating point representation of states
    states_float = torch.tensor(range(0, args.num_states)).float()

    ValueRewardMFC = 0
    curr_gamma = 1

    for iter_count in range(args.run_eval):
        curr_state_dist_mean = torch.dot(states_float, curr_state_dist)
        fraction = 1 - (curr_state_dist_mean / (args.num_states - 1))

        next_state_dist = torch.zeros(args.num_states)
        curr_average_reward = 0

        for state_t in range(0, args.num_states):
            one_hot_state_t = torch.zeros(args.num_states)
            one_hot_state_t[state_t] = 1

            for action_t in range(0, args.num_actions):
                dist_vec = torch.zeros(args.num_states)
                if action_t == 0:
                    dist_vec[state_t] = 1
                else:
                    prob_mass = 1 / (fraction * (args.num_states - 1 - state_t))
                    total_prob = torch.tensor(1.0)
                    state_t_plus_1 = state_t
                    while total_prob > 0 and state_t_plus_1 < args.num_states:
                        dist_vec[state_t_plus_1] = torch.minimum(prob_mass, total_prob)
                        total_prob -= torch.minimum(prob_mass, total_prob)
                        state_t_plus_1 += 1

                prob = actor(one_hot_state_t, curr_state_dist)[action_t] * curr_state_dist[state_t]
                next_state_dist += dist_vec * prob

                reward = args.alpha_r * state_t - args.beta_r * (curr_state_dist_mean ** args.sigma) - args.lambda_r * action_t
                curr_average_reward += reward * prob

        ValueRewardMFC += curr_gamma * args.gamma * curr_average_reward
        curr_gamma *= args.gamma

        curr_state_dist = copy.copy(next_state_dist)

    return ValueRewardMFC


def evaluateMARL(args, N):
    actor = Actor(args.num_states, args.num_actions)

    if not os.path.exists(f'Models/Actor{args.sigma}.pkl'):
        raise ValueError('Model does not exist.')
    actor.load_state_dict(torch.load(f'Models/Actor{args.sigma}.pkl'))

    # Initial state distribution
    init_state_dist = torch.ones(args.num_states)/args.num_states

    # Current Joint State
    curr_joint_state = Categorical(init_state_dist).sample([N]).long()
    next_joint_state = torch.zeros(N).long()

    # Floating point representation of states
    states_float = torch.tensor(range(0, args.num_states)).float()

    # Doubly Stochastic Interaction Matrix
    W = torch.zeros([N, N])
    for i in range(0, N):
        if i + args.K > N:
            W[i, i:] = 1 / args.K
            W[i, :args.K - N + i] = 1 / args.K
        else:
            W[i, i: i + args.K] = 1 / args.K

    ValueRewardMARL = 0
    curr_gamma = 1

    for iter_count in range(args.run_eval):
        curr_average_reward = 0

        curr_joint_state_one_hot = torch.zeros([N, args.num_states])
        curr_joint_state_one_hot[range(0, N), curr_joint_state] = 1

        curr_state_dist = torch.matmul(W, curr_joint_state_one_hot)

        for agent_index in range(0, N):
            agent_state = curr_joint_state[agent_index]
            agent_state_one_hot = curr_joint_state_one_hot[agent_index, :]
            agent_state_dist = curr_state_dist[agent_index, :]
            agent_state_dist_mean = torch.dot(states_float, agent_state_dist)
            agent_action = Categorical(actor(agent_state_one_hot, agent_state_dist)).sample()

            agent_reward = args.alpha_r * agent_state - args.beta_r * (agent_state_dist_mean ** args.sigma) - args.lambda_r * agent_action
            curr_average_reward += agent_reward/N

            # Next State for the agent
            if agent_action == 1:
                chi = torch.rand(1)
                fraction = 1 - (agent_state_dist_mean/(args.num_states-1))
                next_joint_state[agent_index] = curr_joint_state[agent_index] + (chi*fraction*(args.num_states - 1 - curr_joint_state[agent_index])).long()
            else:
                next_joint_state[agent_index] = curr_joint_state[agent_index]

        ValueRewardMARL += curr_gamma*args.gamma*curr_average_reward
        curr_gamma *= args.gamma

        """ ----------- State Update -------------------- """
        curr_joint_state = copy.copy(next_joint_state)

    return ValueRewardMARL

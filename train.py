import gym

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from Model import Policy
from PytorchEnvWrapper import PytorchWrapper
from Util import discountRewards, runPolicy

def train(params, cb = lambda loss, reward_mean, reward_sum: None):
    writer = SummaryWriter(params.log_dir)
    writer.add_text('params', str(params), 0)

    env = PytorchWrapper(gym.make(params.env))
    
    env.seed(params.seed)
    torch.manual_seed(params.seed)

    model = Policy()
    old_model = Policy()
    old_model.load_state_dict(model.state_dict())
    old_model.eval()
    opt = optim.Adam(model.parameters(), lr=params.lr)

    def learn(expBuffer, i):
        action_log_probs, action_log_probs_old, rewards, prewards = list(map(torch.stack, zip(*expBuffer)))
        drewards = discountRewards(rewards, params.gamma)
        drewards_norm = (drewards - drewards.mean()) / (drewards.std() + 1e-8)
        advantages = drewards_norm - prewards

        r_theta = action_log_probs / action_log_probs_old
        clip_r_theta = torch.clamp(r_theta, 1 - params.epsilon, 1 + params.epsilon)
        policy_loss = torch.min(r_theta * advantages, clip_r_theta * advantages).mean()
        
        critic_loss = F.smooth_l1_loss(prewards, drewards_norm, reduction='sum')
        # the paper suggest this lost instead: torch.mean(advantages ** 2); 
        # smooth_l1 version reduce the loss spikes
        # an entropy loss may be added, but its an overkill for simple problems

        loss = policy_loss + params.critic_r * critic_loss
        old_model.load_state_dict(model.state_dict())
        old_model.eval()

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        writer.add_scalar('reward/avg', rewards.mean(), i)
        writer.add_scalar('reward/sum', rewards.sum(), i)
        writer.add_scalar('loss/policy', policy_loss, i)
        writer.add_scalar('loss/critic', critic_loss, i)
        writer.add_scalar('loss/loss', loss, i)
        cb(loss, rewards.mean(), rewards.sum())

    def main():
        for i in range(params.episode):
            obs = env.reset()
            expBuffer = []
            runningReward = 10.0

            for step in range(params.max_step):
                action_probs, preward = model(obs)
                action_probs_old, _ = old_model(obs)
        
                dist = Categorical(action_probs)
                dist_old = Categorical(action_probs_old)
                action = dist.sample()

                obs, reward, done, _ = env.step(action)
                
                expBuffer.append((
                    dist.log_prob(action),
                    dist_old.log_prob(action),
                    reward,
                    preward))

                if done:
                    break

            learn(expBuffer, i)
            runningReward = runningReward * 0.1 + torch.stack(list(zip(*expBuffer))[2]).sum() * 0.9
            print('Episode {}\tLast length: {:5d}\tRunning Rwd: {:3d}'.format(i, step, int(runningReward)))

            if runningReward > 199:
                break

        torch.save(model.state_dict(), params.log_dir + '/finalStateDict')
        print(runningReward.item(), i)
        return runningReward.item() / i
        
    return main()
    
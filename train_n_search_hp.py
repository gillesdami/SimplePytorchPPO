import argparse

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

from train import train

parser = argparse.ArgumentParser(description='hp research for the model')
parser.add_argument('--episode', type=int, default=1000)
parser.add_argument('--env', default='CartPole-v0')
parser.add_argument('--search-resume', default=None)
params = parser.parse_args()

i = 0
def fn_to_optimize(**opt_params):
    global i
    i += 1

    print(opt_params)

    return train(argparse.Namespace(**{
        'episode': params.episode,
        'gamma': opt_params['gamma'].item(),
        'lr': 10**opt_params['lr'].item(),
        'env': params.env,
        'seed': 42,
        'max_step': 10000,
        'log_dir': 'logs/search/'+str(i),
        'critic_r': opt_params['critic_r'].item(),
        'epsilon': opt_params['epsilon'].item()
    }))
    
hyper_params_bounds = {
    'gamma': (0, 1),
    'lr': (-6, -3),
    'critic_r': (0, 10),
    'epsilon': (0.1, 0.3)
}

opt = BayesianOptimization(
    f=fn_to_optimize,
    pbounds=hyper_params_bounds,
    random_state=42,
    verbose=2
)

if params.search_resume:
    load_logs(opt, logs=[params.search_resume+"/logs.json"])

opt.subscribe(Events.OPTMIZATION_STEP, JSONLogger(path="logs/logs.json"))
opt.maximize(init_points=16, n_iter=64)
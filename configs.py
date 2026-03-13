def bouncing_ball_config():
    return {
        'env_id': 'BouncingBall-v0',
        'model_path': './saved_models/bb_ppo_with_replay_250.zip',
        'model_dir': 'bouncing_ball',
        'n_timesteps': 250_000,
        'bounds': [(0,15),(-15,15)],
        'resolution': 400,
        'n_runs': 100,
        'initial_preds': [(0,4),(1,0),(1,-4)],
        'pad_to_size': 401,
        'mark_terminal': False,
        'rounds': 2,
        'n_dims': 2,
        'n_acts': 2
    }

def random_walk_config():
    return {
        'env_id': 'RandomWalk-v0',
        'model_path': './saved_models/rw-ppo-50_000.zip',
        'model_dir': 'random_walk',
        'n_timesteps': 50_000,
        'bounds': [(0,1.2),(0,1.2)],
        'resolution': 1000,
        'n_runs': 1000,
        'initial_preds': [(0,1), (1,1)],
        'pad_to_size': 15,
        'mark_terminal': True,
        'rounds': 15,
        'n_dims': 2,
        'n_acts': 2
    }

def load_config(name):
    if name == 'bouncing_ball':
        return bouncing_ball_config()
    elif name == 'random_walk':
        return random_walk_config()
    else:
        raise ValueError(f"Unknown config name: {name}")

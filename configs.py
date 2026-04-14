def bouncing_ball_config():
    return {
        'env_id': 'BouncingBall-v0',
        'model_name': 'Bouncing Ball',
        'model_path': './saved_models/bb_ppo_with_replay_250.zip',
        'model_dir': 'bouncing_ball',
        'n_timesteps': 250_000,
        'bounds': [(0,15),(-15,15)],
        'resolution': 400,
        'n_runs': 100,
        'initial_preds': [(0,4),(1,0),(1,-4)],
        'pad_to_size': 401,
        'mark_terminal': False,
        'estimation_runs': 25,
        'rounds': 2,
        'n_dims': 2,
        'n_acts': 2,
        'het_thresh': 0.1,
        'laplace': 0.0
    }

def random_walk_config():
    return {
        'env_id': 'RandomWalk-v0',
        'model_name': 'Random Walk',
        'model_path': './saved_models/rw-ppo-50_000.zip',
        'model_dir': 'random_walk',
        'n_timesteps': 50_000,
        'bounds': [(0,1.2),(0,1.2)],
        'resolution': 1000,
        'n_runs': 1000,
        'initial_preds': [(0,1), (1,1)],
        'pad_to_size': 15,
        'mark_terminal': True,
        'estimation_runs': 100,
        'rounds': 15,
        'n_dims': 2,
        'n_acts': 2,
        'laplace': 0.5
    }

def cruise_control_config():
    return {
        'env_id': 'CruiseControl-v0',
        'model_name': 'Cruise Control',
        'model_path': './saved_models/cc_ppo.zip',
        'model_dir': 'cruise_control',
        'n_timesteps': 250_000,
        # state: (v_ego, v_front, distance)
        'bounds': [(-10, 20), (-8, 20), (-20, 220)],
        'resolution': 200,
        'n_runs': 100,
        # splits on all 3 dims: unsafe distance, close following, reversing ego, reversing front
        'initial_preds': [(2, 0), (2, 30), (0, 0), (1, 0)],
        'pad_to_size': 121,
        'mark_terminal': False,
        'estimation_runs': 50,
        'rounds': 3,
        'n_dims': 3,
        'n_acts': 3,
        'laplace': 0.1
    }


def load_config(name):
    if name == 'bouncing_ball':
        return bouncing_ball_config()
    elif name == 'random_walk':
        return random_walk_config()
    elif name == 'cruise_control':
        return cruise_control_config()
    else:
        raise ValueError(f"Unknown config name: {name}")

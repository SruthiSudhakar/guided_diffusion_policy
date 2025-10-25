import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill
import pdb
from termcolor import colored
import time

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])

def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result

def aggregate(data, method='max'):
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        if n_obs_steps > 2:
            print('MORE THAN 2 OBSERVATIONS')
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.grasps = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.grasps = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def hallucinate_step(self, action):
        """
        actions: (n_action_steps,) + action_shape

        Executes actions in the environment to see what would happen,
        but saves internal state so it can be restored later.
        """
        # Save the wrapper's internal state before hallucination
        if not hasattr(self, '_saved_obs'):
            # Store the saved state as instance variables to be restored later
            self._saved_obs = self.obs.copy()
            self._saved_reward = self.reward.copy()
            self._saved_done = self.done.copy()
            self._saved_grasps = self.grasps.copy()
            # Deep copy the defaultdict and its deques
            self._saved_info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))
            for key, val in self.info.items():
                self._saved_info[key] = val.copy()

        temp_observations = []
        temp_processed_obs = []
        i=0
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            # start=time.time()
            observation, processed_obs, reward, done, info = self.env.env.hallucinate_step(act)
            # end=time.time()
            # print(colored(f'{i} step time: {end-start}','red'))
            i+=1
            temp_observations.append(observation)
            temp_processed_obs.append(processed_obs)
        # start=time.time()
        # obs = self.env.env.env.reset_to({'states': current_state})
        # end=time.time()
        # print(colored(f'reset time: {end-start}','red'))
        return temp_observations, temp_processed_obs

    def reset_after_hallucination(self, reset_to_state):
        # Reset the physics state
        # start=time.time()
        obs = self.env.env.env.reset_to({'states': reset_to_state})
        # end=time.time()
        # print(colored(f'reset time: {end-start}','red'))

        # Restore the wrapper's internal state buffers
        if hasattr(self, '_saved_obs'):
            self.obs = self._saved_obs
            self.reward = self._saved_reward
            self.done = self._saved_done
            self.grasps = self._saved_grasps
            self.info = self._saved_info

            # Clear the saved state
            del self._saved_obs
            del self._saved_reward
            del self._saved_done
            del self._saved_grasps
            del self._saved_info

        return obs

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, reward, done, info = super().step(act)

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self.grasps.append(self.env.is_grasping())
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
        
    def is_grasping(self):
        return self.grasps

import numpy as np
import multiprocessing as mp


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                reward, terminated, info = env.step(data)
                obs = env.get_obs()
                state = env.get_state()
                if terminated:
                    obs, state = env.reset()
                remote.send((obs, state, reward, terminated, info))
            elif cmd == 'reset':
                obs, state = env.reset()
                remote.send((obs, state))
            elif cmd == 'get_obs_size':
                remote.send(env.get_obs_size())
            elif cmd == 'get_state_size':
                remote.send(env.get_state_size())
            elif cmd == 'get_total_actions':
                remote.send(env.get_total_actions())
            elif cmd == 'get_num_of_agents':
                remote.send(env.get_num_of_agents())
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        print('Worker KeyboardInterrupt')
    finally:
        remote.close()


class SubprocVecEnv(object):
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = []
        for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns):
            p = mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            p.daemon = True
            p.start()
            self.ps.append(p)
            work_remote.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, state, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(state), np.stack(rews), np.stack(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, state = zip(*results)
        return np.stack(obs), np.stack(state)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def get_obs_size(self):
        self.remotes[0].send(('get_obs_size', None))
        return self.remotes[0].recv()
        
    def get_state_size(self):
        self.remotes[0].send(('get_state_size', None))
        return self.remotes[0].recv()
        
    def get_total_actions(self):
        self.remotes[0].send(('get_total_actions', None))
        return self.remotes[0].recv()
        
    def get_num_of_agents(self):
        self.remotes[0].send(('get_num_of_agents', None))
        return self.remotes[0].recv()

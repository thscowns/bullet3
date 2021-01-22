import time
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
print("parentdir=", parentdir)
import json
from pybullet_envs.deep_mimic.learning.rl_world import RLWorld
from pybullet_envs.deep_mimic.learning.ppo_agent import PPOAgent

import pybullet_data
from pybullet_utils.arg_parser import ArgParser
from pybullet_utils.logger import Logger
from pybullet_envs.deep_mimic.env.pybullet_deep_mimic_env import PyBulletDeepMimicEnv
import sys
import random
import numpy as np
import parmap
update_timestep = 1. / 240.
animating = True
step = False
total_reward = 0
steps = 0
rewards = []
eps = 0
import pandas as pd
import matplotlib.pyplot as plt


def update_world(world, time_elapsed):
    timeStep = update_timestep
    world.update(timeStep)
    reward = world.env.calc_reward(agent_id=0)
    global total_reward
    total_reward += reward
    global steps
    global eps
    steps += 1

    # print("reward=",reward)
    # print("steps=",steps)
    end_episode = world.env.is_episode_end()
    if (end_episode or steps >= 1000):
        # print("total_reward=",total_reward)
        rewards.append(total_reward)
        print("total_reward=", total_reward)
        total_reward = 0
        eps += 1
        steps = 0
        world.end_episode()
        world.reset()
    return


def build_arg_parser(args):
    arg_parser = ArgParser()
    arg_parser.load_args(args)

    arg_file = arg_parser.parse_string('arg_file', '')
    if arg_file == '':
        arg_file = "run_humanoid3d_walk_args.txt"
    if (arg_file != ''):
        path = pybullet_data.getDataPath() + "/args/" + arg_file
        succ = arg_parser.load_file(path)
        Logger.print2(arg_file)
        assert succ, Logger.print2('Failed to load args from: ' + arg_file)
    return arg_parser


args = sys.argv[1:]


def build_world(args, enable_draw, model_path="~/deepmimic_output", i=0):
    arg_parser = build_arg_parser(args)
    print("enable_draw=", enable_draw)
    env = PyBulletDeepMimicEnv(arg_parser, enable_draw)
    world = RLWorld(env, arg_parser)
    # world.env.set_playback_speed(playback_speed)

    motion_file = arg_parser.parse_string("motion_file")
    print("motion_file=", motion_file)
    bodies = arg_parser.parse_ints("fall_contact_bodies")
    print("bodies=", bodies)
    int_output_path = arg_parser.parse_string("int_output_path")
    print("int_output_path=", int_output_path)
    agent_files = pybullet_data.getDataPath() + "/" + arg_parser.parse_string("agent_files")

    AGENT_TYPE_KEY = "AgentType"

    print("agent_file=", agent_files)
    with open(agent_files) as data_file:
        json_data = json.load(data_file)
        print("json_data=", json_data)
        assert AGENT_TYPE_KEY in json_data
        agent_type = json_data[AGENT_TYPE_KEY]
        print("agent_type=", agent_type)
        agent = PPOAgent(world, id, json_data)

        agent.set_enable_training(False)

        world.load_agents(path=model_path, i=i)
        world.reset()
    return world


def main(i, path="/home/thscowns/deepmimic_output/projected_walk/"):
    # f = open("~/deepmimic_output/")
    world = build_world(args, False, model_path=path, i=i)
    rewards = []
    total_reward = 0
    eps = 0
    steps = 0
    while (world.env._pybullet_client.isConnected() and eps <10):

        timeStep = update_timestep
        # time.sleep(timeStep)
        keys = world.env.getKeyboardEvents()
        animating = True
        if world.env.isKeyTriggered(keys, ' '):
            animating = not animating
        if world.env.isKeyTriggered(keys, 'i'):
            step = True
        if (animating or step):
            world.update(timeStep)
            reward = world.env.calc_reward(agent_id=0)
            steps += 1
            step = False
            end_episode = world.env.is_episode_end()
            total_reward += reward
            if (end_episode or steps >= 1000):
                # print("total_reward=",total_reward)
                rewards.append(total_reward)
                print("total_reward=", total_reward)
                total_reward = 0
                eps += 1
                steps = 0
                world.end_episode()
                world.reset()


    rewards = np.array(rewards)
    # world.load_agents(path=path, i=i * 100)
    mean = np.mean(rewards)
    std = np.std(rewards)
    print("mean=", mean)
    print("std=", std)
    return mean, std


def multi(list, path):
    return [main(i * 100, path) for i in list]


def square(l, p):
    return [(x, x + p) for x in l]


def plot(x, data, file_path):
    fig, ax = plt.subplots()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Return")
    for i in data.keys():
        mean = data[i][0]
        std = data[i][1]
        ax.plot(x * 100, mean, label=i)

        plt.fill_between(x * 100, mean-std, mean+std, alpha=.1)
    plt.legend()
    plt.savefig(file_path)


if __name__ == '__main__':
    num_cores = 16
    data = list(range(0, 140))
    splited_data = np.array_split(data, num_cores)
    splited_data = [x.tolist() for x in splited_data]
    # paths = [3, 7, 10]
    paths = ["attached_out_walk/", "projected_walk/", "aligned_walk/"]
    res = {}
    for p in paths:
        # path = "/home/thscowns/deepmimic_output/attached_out_walk/"
        result = parmap.map(multi, splited_data, "/home/thscowns/deepmimic_output/" + p, pm_pbar=True, pm_processes=num_cores)
        '''result = parmap.map(square, splited_data, p, pm_pbar=True,
                            pm_processes=num_cores)'''
        print(result)
        # results = np.array(result).flatten()

        # print("results=", results)
        mean = []
        std = []
        for x in result:
            for y in x:
                mean.append(y[0])
                std.append(y[1])
        print("mean=", mean)
        print("std=", std)
        res[p] = (np.array(mean), np.array(std))
    plot(np.array(data), res, "deepmimic_output/rewards_log_walk.png")
# main(2*100)

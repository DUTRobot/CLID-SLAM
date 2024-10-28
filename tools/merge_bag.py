#!/usr/bin/env python

import os

import yaml
from rosbag import Bag


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def merge(config):
    input_folder = config["input_folder"]
    output_folder = config["output_folder"]
    outbag_name = config["outbag_name"]
    input_bags = os.listdir(input_folder)
    input_bags.sort()  # 根据文件名进行排序

    if config["verbose"]:
        print("Writing bag file: " + outbag_name)

    with Bag(os.path.join(output_folder, outbag_name), 'w') as o:
        for ifile in input_bags:
            if config["verbose"]:
                print("> Reading bag file: " + ifile)
            with Bag(os.path.join(input_folder, ifile), 'r') as ib:
                for topic, msg, t in ib:
                    o.write(topic, msg, t)


if __name__ == "__main__":
    config = load_config('./tools/config/merge.yaml')
    merge(config)

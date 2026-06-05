# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time

import paddle

levels = {0: 'ERROR', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}
log_level = 2

_file_handle = None


def setup_file_logger(log_dir, config_path):
    """Setup file logging. Creates log dir and opens a timestamped log file.

    Args:
        log_dir (str): Directory for log files.
        config_path (str): Path to the YAML config file (used for naming).

    Returns:
        str: Path to the created log file, or None if not on rank 0.
    """
    global _file_handle
    if paddle.distributed.ParallelEnv().local_rank != 0:
        return None
    os.makedirs(log_dir, exist_ok=True)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, "{}_{}.log".format(config_name, timestamp))
    _file_handle = open(log_path, 'w', encoding='utf-8')
    return log_path


def log(level=2, message=""):
    if paddle.distributed.ParallelEnv().local_rank == 0:
        current_time = time.time()
        time_array = time.localtime(current_time)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
        if log_level >= level:
            line = "{} [{}]\t{}".format(current_time, levels[level], message)
            print(line.encode("utf-8").decode("latin1"))
            sys.stdout.flush()
            if _file_handle is not None:
                _file_handle.write(line + '\n')
                _file_handle.flush()


def debug(message=""):
    log(level=3, message=message)


def info(message=""):
    log(level=2, message=message)


def warning(message=""):
    log(level=1, message=message)


def error(message=""):
    log(level=0, message=message)

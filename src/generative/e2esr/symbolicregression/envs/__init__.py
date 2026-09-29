# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# All rights reserved.
#
# This source code is licensed under the license found in the
# licenses/e2esr.txt file in the root directory of this repository.
# Modified for repository integration: license location and portable imports.

from logging import getLogger

# from .generators import operators_conv, Node
from .environment import FunctionEnvironment

logger = getLogger()


ENVS = {
    "functions": FunctionEnvironment,
}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)

    # tasks
    tasks = [x for x in params.tasks.split(",") if len(x) > 0]
    assert len(tasks) == len(set(tasks)) > 0
    assert all(task in env.TRAINING_TASKS for task in tasks)
    params.tasks = tasks
    logger.info(f'Training tasks: {", ".join(tasks)}')

    return env

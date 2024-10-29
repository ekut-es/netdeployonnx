#
# Copyright (c) 2024 netdeployonnx contributors.
#
# This file is part of netdeployonx.
# See https://github.com/ekut-es/netdeployonnx for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import abc


class Scheduler(abc.ABC):
    @abc.abstractclassmethod
    def select_predecessors_scheduled(
        cls, grid, unscheduled_nodes, dependency_free_func: callable = None
    ):
        raise NotImplementedError("Scheduler::select_predecessors_scheduled")

    @abc.abstractclassmethod
    def schedule(cls, grid, node, previous=None):
        raise NotImplementedError("Scheduler::schedule")

    @classmethod
    def dependency_free(cls, node, grid, dependency_free_func: callable = None):
        # check all inputs
        # return true when all inputs are already in the grid
        return all(
            input in grid or (dependency_free_func and dependency_free_func(input))
            for input in node.input
        )


class ASAPScheduler(Scheduler):
    @classmethod
    def select_predecessors_scheduled(
        cls, grid, unscheduled_nodes, dependency_free_func: callable = None
    ):
        node = None
        for node in unscheduled_nodes:
            if cls.dependency_free(
                node, grid, dependency_free_func=dependency_free_func
            ):
                break
        else:
            # we could find out why we could not schedule any node
            for node in unscheduled_nodes:
                reasons = {
                    f"{input} in grid": (lambda: input in grid) for input in node.input
                }
                if dependency_free_func:
                    reasons.update(
                        {
                            f"{input} {dependency_free_func(input)}": (
                                lambda: dependency_free_func(input)
                            )
                            for input in node.input
                        }
                    )
                for reason, check in reasons.items():
                    if check():
                        break
                # print(f"{node.name} => {reason}")
            raise Exception("no free node found")
        # print("selected node", node.name)
        unscheduled_nodes.remove(node)
        return node, unscheduled_nodes

    @classmethod
    def schedule(cls, grid, node, previous=None):
        x, y = previous or (0, 0)

        # find a free slot
        while grid[x, y]:
            x = x
            y = y + 1  # just one below

        return x, y

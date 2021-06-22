import copy
import uuid

import networkx as nx
from dask import visualize


class Workflow:
    def __init__(self, tasks=None):
        self.tasks = nx.DiGraph()
        if tasks:
            self.add_tasks(tasks, connect=False)

    def add_tasks(
        self, other, connect=False, output_nodes=None, as_single_element=True, arg_index=0
    ):
        """Keep all nodes and edges, connects output from first workflow to input in second if connect=True
        (assumes 1:M, M:1 or 1:1 connections)"""
        if output_nodes:
            wf1_out_tasks = output_nodes
        else:
            wf1_out_tasks = self.get_output()
        if not isinstance(other, Workflow):
            wf2_in_tasks = []
            if isinstance(other, list):
                self.tasks.add_nodes_from(other)
                wf2_in_tasks.extend(other)
            else:
                self.tasks.add_node(other)
                wf2_in_tasks.append(other)
        else:
            wf2_in_tasks = other.get_input()
            self.tasks = nx.compose(self.tasks, other.tasks)

        if not connect:
            return
        else:
            # TODO: assert this is consistent with task input
            workflow_connection = self.find_workflow_connections(wf1_out_tasks, wf2_in_tasks)
            for wf2_in_task in wf2_in_tasks:
                if as_single_element and len(wf1_out_tasks) == 1:
                    wf1_out_task = wf1_out_tasks[0]
                else:
                    wf1_out_task = wf1_out_tasks
                if not wf2_in_task.has_input():
                    wf2_in_task.task_input = (wf1_out_task,)
                else:
                    wf2_in_task_input = list(wf2_in_task.task_input)
                    wf2_in_task_input.insert(arg_index, wf1_out_task)
                    wf2_in_task.task_input = tuple(wf2_in_task_input)
            self.connect_tasks(workflow_connection)

    @staticmethod
    def find_workflow_connections(wf1_out_tasks, wf2_in_tasks):
        if len(wf1_out_tasks) > 1 and len(wf2_in_tasks) > 1:
            raise ValueError('Having N:M connections currently not supported')
        elif len(wf1_out_tasks) > 1:
            wf2_in_task = wf2_in_tasks[0]
            workflow_connection = {wf1_out_task: wf2_in_task for wf1_out_task in wf1_out_tasks}
        elif len(wf2_in_tasks) > 1:
            wf1_out_task = wf1_out_tasks[0]
            workflow_connection = {wf1_out_task: wf2_in_task for wf2_in_task in wf2_in_tasks}
        else:
            wf1_out_task, wf2_in_task = wf1_out_tasks[0], wf2_in_tasks[0]
            workflow_connection = {wf1_out_task: wf2_in_task}
        return workflow_connection

    def connect_tasks(self, connect_dict):
        """Connects task with dict: {from: to}"""
        edges = []
        for key, value in connect_dict.items():
            if key == value:
                raise ValueError(f'Cannot connect {key.name} to itself')
            if isinstance(value, list):
                key_edges = [(key, v) for v in value]
                edges.extend(key_edges)
            else:
                edges.append((key, value))
        self.tasks.add_edges_from(edges)

    def as_dict(self):
        as_dict = dict()
        for task in nx.dfs_tree(self.tasks):
            key = task.task_id
            input_list = [self.id_convert(t) for t in task.task_input]
            value = (task.function, *input_list)
            as_dict[key] = value
        return as_dict

    @staticmethod
    def id_convert(task):
        # TODO: Convert recursively
        if isinstance(task, Task):
            return task.task_id
        elif isinstance(task, list):
            return [t.task_id if isinstance(t, Task) else t for t in task]
        else:
            return task

    def get_output(self):
        return [node for node in self.tasks.nodes if self.tasks.out_degree(node) == 0]

    def get_input(self):
        return [node for node in self.tasks.nodes if self.tasks.in_degree(node) == 0]

    def get_upstream_tasks(self, task):
        edges = list(nx.edge_dfs(self.tasks, task, orientation='reverse'))
        return [node for node, _, _ in edges]

    def force_new_task_ids(self):
        for task in self.tasks.nodes:
            task.force_new_id()

    def copy(self, new_ids=True):
        wf_copy = copy.deepcopy(self)
        if new_ids:
            wf_copy.force_new_task_ids()
        return wf_copy

    def plot_dask(self, filename):
        visualize(self.as_dict(), filename=filename, collapse_outputs=True)

    # def plot_graph(self, filename):
    #     nx.draw(self.tasks, with_labels=True, font_weight='bold')
    #     plt.savefig(filename)

    def __str__(self):
        return '\n'.join([str(task) for task in self.tasks])


class Task:
    def __init__(self, name, function, *task_input, final_task=False):
        self.name = name
        self.function = function
        self.task_input = task_input
        if not final_task:
            self.task_id = f'{name}-{uuid.uuid4()}'
        else:
            self.task_id = 'results'

    def has_input(self):
        return len(self.task_input) > 0

    def force_new_id(self):
        self.task_id = f'{self.name}-{uuid.uuid4()}'

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

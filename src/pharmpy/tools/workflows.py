import uuid


class Workflow:
    def __init__(self, tasks=None):
        self.tasks = []
        if tasks:
            self.add_tasks(tasks)

    def add_tasks(self, tasks):
        if isinstance(tasks, list):
            self.tasks.extend(tasks)
        else:
            self.tasks.append(tasks)

    def as_dict(self):
        as_dict = dict()
        for task in self.tasks:
            if isinstance(task.task_input, Task):
                value = (task.function, task.task_input.task_id)
            else:
                value = (task.function, *task.task_input)
            as_dict[task.task_id] = value
        return as_dict

    def get_task(self, name):
        for task in self.tasks:
            if task.name == name:
                return task
        return None

    def merge_workflows(self, other):
        self.add_tasks(other.tasks)

    def __str__(self):
        return '\n'.join([str(task) for task in self.tasks])


class Task:
    def __init__(self, name, function, task_input, final_task=False, condition=None):
        self.name = name
        self.function = function
        self.task_input = self.format_input(task_input)
        if not final_task:
            self.task_id = f'{name}_{uuid.uuid1()}'
        else:
            self.task_id = 'results'
        self.condition = condition

    @staticmethod
    def format_input(task_input):
        input_formatted = []
        if isinstance(task_input, str):
            task_input = [task_input]
        try:
            for elem in task_input:
                if isinstance(elem, Task):
                    input_formatted.append(elem.task_id)
                else:
                    input_formatted.append(elem)
        except TypeError:
            input_formatted = [task_input]
        return input_formatted

    def add_condition(self, other):
        if self.condition and not isinstance(self.condition, list):
            self.condition = list(self.condition)
        self.condition.append(other)

    def __str__(self):
        input_str = ', '.join([str(type(elem)) for elem in self.task_input])
        return f'{self.name}: {self.function}({input_str})'

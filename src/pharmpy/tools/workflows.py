class Workflow:
    def __init__(self, tasks=None, models=None):
        self.tasks = []
        if tasks:
            self.add_tasks(tasks)
        self.models = models
        self.infiles = []
        self.outfiles = []

    def add_tasks(self, tasks):
        if isinstance(tasks, list):
            self.tasks.extend(tasks)
        else:
            self.tasks.append(tasks)

    def as_dict(self):
        as_dict = dict()
        for task in self.tasks:
            if isinstance(task.task_input, tuple):
                value = (task.function, *task.task_input)
            else:
                value = (task.function, task.task_input)
            as_dict[task.name] = value
        return as_dict

    def get_task(self, name):
        for task in self.tasks:
            if task.name == name:
                return task
        return None

    def add_infiles(self, source, destination='.'):
        self.infiles.append((source, destination))

    def add_outfiles(self, outfiles):
        self.outfiles.append(outfiles)


class Task:
    def __init__(self, name, function, task_input, condition=None):
        self.name = name
        self.function = function
        self.task_input = task_input
        self.condition = condition

    def add_condition(self, other):
        if self.condition and not isinstance(self.condition, list):
            self.condition = list(self.condition)
        self.condition.append(other)

    def __str__(self):
        return f'{self.name}: {self.function}({self.task_input})'

class Workflow:
    def __init__(self, tasks=None):
        self.tasks = []
        if tasks:
            self.add_tasks(tasks)
        self.infiles = []
        self.outfiles = {}

    def add_tasks(self, tasks):
        if isinstance(tasks, list):
            self.tasks.extend(tasks)
        else:
            self.tasks.append(tasks)

    def add_infiles(self, infile_new):
        if isinstance(infile_new, list):
            self.infiles.extend(infile_new)
        elif isinstance(infile_new, tuple):
            self.infiles.append(infile_new)
        else:
            self.infiles.append((infile_new, '.'))

    def add_outfiles(self, outfiles):
        # TODO: cover cases when same model have added outfiles
        self.outfiles = {**self.outfiles, **outfiles}

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

    def merge_workflows(self, other):
        self.add_tasks(other.tasks)
        self.add_infiles(other.infiles)
        self.add_outfiles(other.outfiles)

    def __str__(self):
        tasks_str = '\n'.join([str(task) for task in self.tasks])
        infiles_str = '\n'.join([f'{source},{destination}' for source, destination in self.infiles])
        outfiles_str = '\n'.join(
            [f'{model.name},{outfile}' for model, outfile in self.outfiles.items()]
        )
        return f'Tasks:\n{tasks_str}\nInfiles:\n{infiles_str}\nOutfiles:\n{outfiles_str}'


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

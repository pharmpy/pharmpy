import sys
from typing import TypeVar

from pharmpy.internals.fs.cwd import chdir
from pharmpy.internals.fs.tmp import TemporaryDirectory

from ..workflow import Workflow, WorkflowBuilder, insert_context
from .baseclass import Dispatcher

T = TypeVar('T')


class LocalSerialDispatcher(Dispatcher):
    def run(self, workflow: Workflow[T], context) -> T:
        with TemporaryDirectory() as tempdirname, chdir(tempdirname):
            context.log_info(f"Dispatching workflow in {context}")
            res = self._run(workflow)
            context.log_info("End dispatch")
        return res  # pyright: ignore [reportGeneralTypeIssues]

    @staticmethod
    def _run(workflow):
        if len(workflow.input_tasks) == 1:
            start_task = workflow.input_tasks[0]
            queue = list(workflow.traverse('bfs', start_task))
        else:
            # FIXME: Assumes workflows in two layers (e.g. several modelfit)
            queue = workflow.input_tasks + [workflow.output_tasks[0]]

        res_cache, visited = dict(), list()

        unvisited_children = {task: len(workflow.get_successors(task)) for task in workflow.tasks}
        while queue:
            task = queue.pop(0)
            parent_tasks = workflow.get_predecessors(task)
            parent_res = [res_cache[parent] for parent in parent_tasks if parent in res_cache]
            if len(parent_res) != len(parent_tasks):
                queue.append(task)
                continue
            visited.append(task)
            for parent in parent_tasks:
                unvisited_children[parent] -= 1
                if unvisited_children[parent] == 0:
                    del res_cache[parent]
            task_res = task.function(*task.task_input, *parent_res)
            res_cache[task] = task_res

        assert len(workflow.output_tasks) == 1
        final_task = workflow.output_tasks[0]
        res = res_cache[final_task]

        return res

    def call_workflow(self, wf: Workflow[T], unique_name, ctx) -> T:
        wb = WorkflowBuilder(wf)
        insert_context(wb, ctx)
        wf = Workflow(wb)

        res = self._run(wf)

        return res

    def abort_workflow(self) -> None:
        sys.exit()

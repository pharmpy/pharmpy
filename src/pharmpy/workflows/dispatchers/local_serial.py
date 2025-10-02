import os
from typing import NoReturn, Optional, TypeVar

from pharmpy.internals.fs.cwd import chdir
from pharmpy.internals.fs.tmp import TemporaryDirectory

from ..workflow import Workflow, WorkflowBuilder, insert_context
from .baseclass import AbortWorkflowException, Dispatcher, SigHandler
from .slurm_helpers import get_slurm_nodedict, get_slurm_nodename, is_running_on_slurm

T = TypeVar('T')


class LocalSerialDispatcher(Dispatcher):
    def run(self, workflow: Workflow[T], context) -> Optional[T]:
        with TemporaryDirectory() as tempdirname, chdir(tempdirname):
            context.log_info(f"Dispatching workflow with local_serial dispatcher in {context}")
            with SigHandler(context):
                try:
                    res = self._run(workflow)
                except AbortWorkflowException:
                    res = None
            context.log_info("End dispatch")
        return res

    @staticmethod
    def _run(workflow: Workflow[T]) -> T:
        res_cache = dict()
        unvisited_children = {task: len(workflow.get_successors(task)) for task in workflow.tasks}
        for task in workflow.sort('topological'):
            parent_tasks = workflow.get_predecessors(task)
            parent_res = [res_cache[parent] for parent in parent_tasks if parent in res_cache]
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

    def call_workflow(self, wf: Workflow[T], unique_name: str, context) -> T:
        wb = WorkflowBuilder(wf)
        insert_context(wb, context)
        wf = Workflow(wb)

        res = self._run(wf)

        return res

    def abort_workflow(self) -> NoReturn:
        raise AbortWorkflowException

    def get_hosts(self) -> dict[str, int]:
        if is_running_on_slurm():
            return get_slurm_nodedict()
        else:
            return {'localhost': os.cpu_count() or 1}

    def get_hostname(self) -> str:
        if is_running_on_slurm():
            return get_slurm_nodename()
        else:
            return 'localhost'

    def get_available_cores(self, allocation: int) -> int:
        return allocation

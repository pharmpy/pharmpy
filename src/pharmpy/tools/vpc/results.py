from dataclasses import dataclass
from typing import Any, Optional

from pharmpy.modeling import plot_vpc
from pharmpy.workflows.results import Results


@dataclass(frozen=True)
class VPCResults(Results):
    plot: Optional[Any] = None


def calculate_results(input_model, simulation_results, stratify):
    vpc_plot = plot_vpc(input_model, simulation_results, stratify_on=stratify)
    res = VPCResults(
        plot=vpc_plot,
    )

    return res

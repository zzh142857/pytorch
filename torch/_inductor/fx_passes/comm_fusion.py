import logging
import operator
from typing import Any, Callable, Optional, Union

import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph

from ..pattern_matcher import (
    CallFunctionVarArgs,
    get_arg_value,
    MULTIPLE,
    PatternMatcherPass,
    register_graph_pattern,
)

logger = logging.getLogger(__name__)


comm_fusion_patterns = PatternMatcherPass()


def comm_fusion_passes(graph: torch.fx.Graph) -> None:
    """Pattern matching passes for collective communication fucntion fusion."""
    if not torch.distributed.is_available:
        logger.info(
            "Warning: torch.distributed is not available, skip communication fusion"
        )
        return

    print_graph(graph, "Before communication fusion in post grad pass.")
    comm_fusion_patterns.apply(graph)
    print_graph(graph, "After communication fusion in post grad pass.")


@register_graph_pattern(
    CallFunctionVarArgs(torch.ops.c10d_functional.all_reduce.default, users=MULTIPLE),
    pass_dict=comm_fusion_patterns,
)
def allreduce_split_to_reducescatter(match, *args, **kwargs):
    """
    Replace the following pattern:
        AllReduce() - wait() - split - getitem

    with:
        ReduceScatter() - wait()

    """
    node = match.nodes[0]
    graph = match.graph
    if len(node.args) != 5:
        logger.warning("AllReduce node has %s args, expected 5", len(node.args))
        return

    # Match the pattern, for now the it only works with single user/child node.
    wait_node = CommFusionHelper.get_only_child(
        node, "call_function", torch.ops.c10d_functional.wait_tensor.default
    )
    if wait_node is None:
        return

    split_node = CommFusionHelper.get_only_child(
        wait_node, "call_function", torch.ops.aten.split.Tensor
    )
    if split_node is None:
        return

    getitem_node = CommFusionHelper.get_only_child(
        split_node, "call_function", operator.getitem
    )
    if getitem_node is None or len(getitem_node.users) != 1:
        return

    # Make sure the output is contiguous to ensure numerical correctness
    if not getitem_node.meta["val"].is_contiguous():
        return

    # Only support split on default dim=0, check split size
    split_dim = get_arg_value(split_node, 2, "dim") or 0
    if split_dim != 0:
        return

    split_size = split_node.args[1]
    all_reduce_shape = node.meta["val"].shape[split_dim]
    if split_size * len(node.args[3]) != all_reduce_shape:
        return

    # Make sure the getitem index is the same as the rank idx
    current_rank = torch.distributed.get_rank()
    rank_list = node.args[3]
    getitem_idx = getitem_node.args[1]
    rank_idx = -1
    for i, rank in enumerate(rank_list):
        if rank == current_rank:
            rank_idx = i
            break
    if rank_idx != getitem_idx:
        return

    # Fuse to ReduceScatter
    output_node = list(getitem_node.users)[0]
    with graph.inserting_before(output_node):
        reduce_scatter_node = graph.call_function(
            torch.ops.c10d_functional.reduce_scatter_tensor.default,
            args=(
                node.args[0],  # input
                node.args[1],  # reduce_op
                node.args[2],  # tag
                node.args[3],  # ranks
                node.args[4],  # group_size
            ),
        )
        new_wait_node = graph.call_function(
            torch.ops.c10d_functional.wait_tensor.default,
            args=(reduce_scatter_node,),
        )

    # Update metadata
    reduce_scatter_node.meta.update(getitem_node.meta)
    reduce_scatter_node.meta.update(
        {
            "original_aten": torch.ops.c10d_functional.reduce_scatter_tensor.default,
        }
    )
    new_wait_node.meta.update(getitem_node.meta)
    new_wait_node.meta.update(
        {
            "original_aten": torch.ops.c10d_functional.wait_tensor.default,
        }
    )

    getitem_node.replace_all_uses_with(new_wait_node)
    graph.eliminate_dead_code()
    counters["inductor"]["comm_fusion"] += 1


class CommFusionHelper:
    """This is a helper class for collective communication fusion."""

    @staticmethod
    def get_only_child(
        node: torch.fx.Node, op: str, target: Union[Callable[..., Any], str]
    ) -> Optional[torch.fx.Node]:
        """Check if the node has only one user and if the user is the target function.
        Return the user node if the condition is met, otherwise return None.
        """
        if len(node.users) != 1:
            return None

        user = list(node.users)[0]
        if user.op == "call_function" and user.target == target:
            return user
        else:
            return None

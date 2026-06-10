"""Runtime patch for fixed-membership torchrun rendezvous shrink detection.

Load this module before invoking ``torch.distributed.run``/``torchrun`` logic.
It applies the behavior from ``/root/rendezvous.diff`` without modifying the
installed torch package files.
"""

import time
from typing import Optional

from torch.distributed.elastic.agent.server import api as agent_api
from torch.distributed.elastic.rendezvous import dynamic_rendezvous


def _num_nodes_participating(self) -> int:
    """Return the number of nodes participating in the current rendezvous."""
    try:
        with self._heartbeat_lock:
            self._state_holder.sync()

            return len(self._state_holder.state.participants)

    except Exception as e:
        self._record(
            message=f"{type(e).__name__}: {str(e)}",
            node_state=dynamic_rendezvous.NodeState.FAILED,
        )
        raise


def _stop_workers_if_fixed_rdzv_membership_shrank(
    self, worker_group: agent_api.WorkerGroup
) -> Optional[agent_api.RunResult]:
    spec = worker_group.spec
    rdzv_handler = spec.rdzv_handler
    settings = getattr(rdzv_handler, "settings", None)
    num_nodes_participating = getattr(rdzv_handler, "num_nodes_participating", None)

    if (
        settings is None
        or not callable(num_nodes_participating)
        or settings.min_nodes != settings.max_nodes
        or spec.max_restarts != 0
    ):
        return None

    participants = num_nodes_participating()
    if participants >= settings.min_nodes:
        return None

    agent_api.logger.warning(
        "[%s] Detected rendezvous participants below fixed membership: "
        "participants=%s, min_nodes=%s; stopping worker group",
        spec.role,
        participants,
        settings.min_nodes,
    )

    self._stop_workers(worker_group)
    worker_group.state = agent_api.WorkerState.FAILED

    return agent_api.RunResult(state=agent_api.WorkerState.FAILED)


def _invoke_run_with_fixed_rdzv_shrink_check(
    self, role: str = agent_api.DEFAULT_ROLE
) -> agent_api.RunResult:
    spec = self._worker_group.spec
    role = spec.role

    agent_api.logger.info(
        "[%s] starting workers for entrypoint: %s", role, spec.get_entrypoint_name()
    )

    self._initialize_workers(self._worker_group)
    monitor_interval = spec.monitor_interval
    rdzv_handler = spec.rdzv_handler

    while True:
        assert self._worker_group.state != agent_api.WorkerState.INIT
        time.sleep(monitor_interval)
        run_result = self._monitor_workers(self._worker_group)
        state = run_result.state
        self._worker_group.state = state

        agent_api.put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
        agent_api.put_metric(f"workers.{role}.{state.name.lower()}", 1)

        if state == agent_api.WorkerState.SUCCEEDED:
            agent_api.logger.info(
                "[%s] worker group successfully finished."
                " Waiting %s seconds for other agents to finish.",
                role,
                self._exit_barrier_timeout,
            )
            self._exit_barrier()
            return run_result
        elif state in {agent_api.WorkerState.UNHEALTHY, agent_api.WorkerState.FAILED}:
            if self._remaining_restarts > 0:
                agent_api.logger.info(
                    "[%s] Worker group %s. "
                    "%s/%s attempts left;"
                    " will restart worker group",
                    role,
                    state.name,
                    self._remaining_restarts,
                    spec.max_restarts,
                )
                self._remaining_restarts -= 1
                self._restart_workers(self._worker_group)
            else:
                self._stop_workers(self._worker_group)
                self._worker_group.state = agent_api.WorkerState.FAILED
                return run_result
        elif state == agent_api.WorkerState.HEALTHY:
            print(f"================ HEALTHY LOOP !!!")
            run_result = self._stop_workers_if_fixed_rdzv_membership_shrank(
                self._worker_group
            )
            if run_result is not None:
                return run_result

            # Membership changes do not count as retries.
            num_nodes_waiting = rdzv_handler.num_nodes_waiting()
            group_rank = self._worker_group.group_rank
            if num_nodes_waiting > 0:
                agent_api.logger.info(
                    "[%s] Detected %s "
                    "new nodes from group_rank=%s; "
                    "will restart worker group",
                    role,
                    num_nodes_waiting,
                    group_rank,
                )
                self._restart_workers(self._worker_group)
        else:
            raise Exception(f"[{role}] Worker group in {state.name} state")


def apply_patch() -> None:
    dynamic_rendezvous.DynamicRendezvousHandler.num_nodes_participating = (
        _num_nodes_participating
    )
    agent_api.SimpleElasticAgent._stop_workers_if_fixed_rdzv_membership_shrank = (
        _stop_workers_if_fixed_rdzv_membership_shrank
    )
    agent_api.SimpleElasticAgent._invoke_run = _invoke_run_with_fixed_rdzv_shrink_check
    agent_api.SimpleElasticAgent._fixed_rdzv_shrink_patch_applied = True


apply_patch()


if __name__ == "__main__":
    from torch.distributed.run import main

    main()

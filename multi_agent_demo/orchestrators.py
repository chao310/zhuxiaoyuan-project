"""
四种协同组织模式。

每个类只负责"谁在什么时机说话"，不关心智能体内部怎么思考。
这样同一批智能体可以换不同编排器复用。
"""
from __future__ import annotations

import json

from framework import (Agent, header, is_verbose, log, parse_assignments,
                       parse_state)


# ---------------------------------------------------------------------------
# 1. 中心化编排
# ---------------------------------------------------------------------------
class CentralizedOrchestrator:
    """一个总控负责规划和汇总，专业智能体只在自己领域执行。"""

    label = "中心化编排：总控规划 → 专业智能体执行 → 总控汇总"

    def __init__(self, planner: Agent, workers: dict[str, Agent]):
        self.planner = planner
        self.workers = workers

    def run(self, task: str, seed: str = "", banner: bool = True) -> str:
        if banner:
            header(self.label)

        plan_text = self.planner.run(task)
        log(self.planner.name, plan_text)

        receipts: list[tuple[str, str]] = []
        for who, subtask in parse_assignments(plan_text):
            worker = self.workers.get(who)
            if worker is None:
                print(f"  [跳过] 计划里出现了未知角色：{who}")
                continue

            out = worker.run(subtask, context=self._join(seed, receipts))
            log(worker.name, out, indent=1)
            receipts.append((worker.name, out))

        summary_context = self._join("", receipts)
        summary = self.planner.run("汇总以上结果", context=summary_context)
        if is_verbose():
            log(f"{self.planner.name}（汇总）", summary)
        return summary

    @staticmethod
    def _join(seed: str, receipts: list[tuple[str, str]]) -> str:
        parts = [seed] if seed.strip() else []
        parts.extend(f"【{name}】\n{text}" for name, text in receipts)
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 2. 战略—战术分层
# ---------------------------------------------------------------------------
class HierarchicalOrchestrator:
    """战略层只决定"哪个小队做什么"，小队内部再自主编排。"""

    label = "战略—战术分层：战略官 → 小队（各自有队长和队员）"

    def __init__(self, strategist: Agent,
                 teams: dict[str, CentralizedOrchestrator]):
        self.strategist = strategist
        self.teams = teams

    def run(self, task: str) -> str:
        header(self.label)

        directives = self.strategist.run(task)
        log(self.strategist.name, directives)

        reports: list[tuple[str, str]] = []
        for team_name, objective in parse_assignments(directives):
            team = self.teams.get(team_name)
            if team is None:
                print(f"  [跳过] 战略层提到了未知小队：{team_name}")
                continue

            print(f"\n  ── 交给「{team_name}」，目标：{objective} ──")
            seed = "\n\n".join(f"【{n}】\n{t}" for n, t in reports)
            out = team.run(objective, seed=seed, banner=False)
            reports.append((team_name, out))

        summary = self.strategist.run(
            "汇总各小队战报",
            context="\n\n".join(f"【{n}】\n{t}" for n, t in reports))
        if is_verbose():
            log(f"{self.strategist.name}（汇总）", summary)
        return summary


# ---------------------------------------------------------------------------
# 3. 按阶段 / 任务域分解
# ---------------------------------------------------------------------------
class PhasePipeline:
    """不设总控，把流程切成阶段，阶段之间只通过一份结构化状态衔接。"""

    label = "阶段/任务域分解：监控 → 诊断 → 处置 → 汇报（共享状态）"

    def __init__(self, phases: list[Agent]):
        self.phases = phases

    def run(self, task: str) -> str:
        header(self.label)

        state: dict[str, str] = {"target": task}
        last = ""
        for index, phase in enumerate(self.phases, 1):
            context = json.dumps(state, ensure_ascii=False, indent=2)
            out = phase.run(task, context=context)
            log(f"{index}. {phase.name}", out)
            state.update(parse_state(out))
            last = out

        if is_verbose():
            print("\n最终共享状态：")
            for key, value in state.items():
                if key != "target":
                    print(f"  {key} = {value}")
        return state.get("report", last)


# ---------------------------------------------------------------------------
# 4. 黑板架构
# ---------------------------------------------------------------------------
class Blackboard:
    """共享工作区，只记录三类东西：事实、意图、线索。"""

    KINDS = ("fact", "intent", "hint")

    def __init__(self):
        self.entries: list[dict] = []

    def add(self, kind: str, key: str, value: str, author: str) -> None:
        self.entries.append(
            {"kind": kind, "key": key, "value": value, "author": author})

    def value_of(self, key: str) -> str | None:
        for entry in reversed(self.entries):
            if entry["key"] == key:
                return entry["value"]
        return None

    def render(self) -> str:
        if not self.entries:
            return "（黑板为空）"
        return "\n".join(
            f"- {e['kind'].upper():<6} [{e['key']}] {e['value']}  (by {e['author']})"
            for e in self.entries)


class BlackboardOrchestrator:
    """智能体之间不直接喊话，只读写黑板；调度器看黑板决定下一步。"""

    label = "黑板架构：共享工作区 + 调度器"

    def __init__(self, controller: Agent, agents: dict[str, Agent],
                 max_rounds: int = 10):
        self.controller = controller
        self.agents = agents
        self.max_rounds = max_rounds

    def run(self, task: str) -> str:
        header(self.label)

        board = Blackboard()
        board.add("hint", "task", task, "system")
        if is_verbose():
            log("黑板（初始）", board.render())

        for round_no in range(1, self.max_rounds + 1):
            snapshot = board.render()
            decision = self.controller.run(
                "根据黑板状态决定下一步", context=snapshot)
            who = decision.strip().splitlines()[0].strip(" `'\"。.")

            log(f"调度器（第 {round_no} 轮）", f"下一步 → {who}")
            if who.upper().startswith("DONE"):
                break

            agent = self.agents.get(who)
            if agent is None:
                print(f"  [结束] 调度器返回了无法识别的角色：{who}")
                break

            out = agent.run(task, context=snapshot)
            log(agent.name, out)
            self._write(board, out, agent.name)
            if is_verbose():
                log("黑板（更新）", board.render(), indent=1)

        return board.value_of("report") or board.render()

    @staticmethod
    def _write(board: Blackboard, text: str, author: str) -> None:
        """约定：智能体用 `FACT: key = value` 这样的行往黑板上写。"""
        for raw in text.splitlines():
            line = raw.strip()
            lowered = line.lower()
            for kind in Blackboard.KINDS:
                if lowered.startswith(kind + ":") or lowered.startswith(kind + "："):
                    body = line[len(kind) + 1:].strip()
                    key, sep, value = body.partition("=")
                    if sep:
                        board.add(kind, key.strip(), value.strip(), author)
                    else:
                        board.add(kind, body[:24], "", author)
                    break

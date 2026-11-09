"""
教学场景：机房告警处置。

这个场景专门用来回答"什么是多智能体"：
    多个角色（监控员、诊断员、处置员、汇报员）
    各自有局部目标、局部知识和工具
    通过共享黑板通信、由调度器协调
    共同完成同一个任务：处理 A-12 机柜温度告警。

所有工具均为模拟实现，只返回固定字符串，不会发起任何真实操作。
"""
from __future__ import annotations

import json

from framework import Agent, LLMClient, Tool, text_brain, tool_brain
from orchestrators import (BlackboardOrchestrator, CentralizedOrchestrator,
                           HierarchicalOrchestrator, PhasePipeline)


# ---------------------------------------------------------------------------
# 模拟工具：每个角色只能看到自己领域的信息
# ---------------------------------------------------------------------------
def sim_get_alarms(target: str) -> str:
    return ("A-12 机柜温度 45℃（阈值 40℃）；"
            "srv-web-01 风扇转速 1200 RPM（正常 6000）；"
            "PDU-3 负载 92%")


def sim_get_asset(target: str) -> str:
    return ("srv-web-01：机柜 A-12 / U20；上联 SW-2 的 Gi0/3；"
            "承载 order-service；备机 srv-web-02")


def sim_get_metrics(target: str) -> str:
    return ("srv-web-01：CPU 35%，内存 60%，进风温度 38℃，"
            "风扇转速 1200 RPM（正常 6000）")


def sim_migrate(target: str) -> str:
    return "已将 order-service 从 srv-web-01 迁移到 srv-web-02（模拟，未真实操作）"


def sim_ticket(target: str) -> str:
    return "已创建工单 INC-2026-1109-001（模拟）"


TOOL_ALARMS = Tool("get_alarms", "读取当前告警（模拟）", sim_get_alarms)
TOOL_ASSET = Tool("get_asset", "查询资产与依赖关系（模拟）", sim_get_asset)
TOOL_METRICS = Tool("get_metrics", "读取设备指标（模拟）", sim_get_metrics)
TOOL_MIGRATE = Tool("migrate_service", "迁移业务到备机（模拟）", sim_migrate)
TOOL_TICKET = Tool("open_ticket", "创建处置工单（模拟）", sim_ticket)


# ---------------------------------------------------------------------------
# 角色说明：接真实 LLM 时，这些文字就是 system prompt 的一部分
# ---------------------------------------------------------------------------
ROLE_PLANNER = ("你是运维总控，只做规划和汇总，不直接调用工具。"
                "规划时每行输出 `角色名: 子任务`，不要输出其他内容。")
ROLE_STRATEGIST = ("你是运维指挥官，只决定交给哪个小队、目标是什么，"
                   "不关心具体操作。每条指令一行：`小队名: 目标`。")
ROLE_MONITOR = ("你是监控员，负责发现告警。只汇报告警事实，不判断原因。")
ROLE_DIAGNOSER = ("你是诊断员，负责结合资产信息和设备指标定位原因。"
                  "只给出原因判断和依据，不执行处置。")
ROLE_FIXER = ("你是处置员，负责按诊断结论执行处置并创建工单。"
              "不重新诊断，只执行。")
ROLE_REPORTER = ("你是汇报员，负责把整个过程整理成给运维主管看的事件报告。")
ROLE_CONTROLLER = ("你是黑板调度器。看黑板内容，只输出下一个应当行动的"
                   "智能体名字，或输出 DONE。不要解释。")


# ---------------------------------------------------------------------------
# 脚本化大脑：不接 LLM 时使用，便于观察协作过程
# ---------------------------------------------------------------------------
def planner_brain(agent, task, context):
    if "汇总" in task:
        return "处置完成：\n" + context
    return "\n".join([
        "监控员: 读取 A-12 机柜相关告警",
        "诊断员: 结合资产和指标定位原因",
        "处置员: 执行处置并创建工单",
        "汇报员: 输出事件报告",
    ])


def monitor_brain(agent, task, context):
    return "FACT: alarms = " + agent.use("get_alarms", "A-12")


def diagnose_brain(agent, task, context):
    asset = agent.use("get_asset", "srv-web-01")
    metrics = agent.use("get_metrics", "srv-web-01")
    return ("资产：srv-web-01 位于 A-12 / U20，承载 order-service\n"
            f"指标：{metrics}\n"
            "推理：风扇转速远低于正常值，散热不足导致机柜温度升高\n"
            "FACT: cause = srv-web-01 风扇故障导致散热不足")


def fix_brain(agent, task, context):
    action = agent.use("migrate_service", "order-service")
    ticket = agent.use("open_ticket", "A-12")
    return f"FACT: action = {action}\nFACT: ticket = {ticket}"


def report_brain(agent, task, context):
    return ("FACT: report = 【事件报告】A-12 机柜温度告警："
            "根因是 srv-web-01 风扇故障；已迁移 order-service 到 "
            "srv-web-02 并创建工单 INC-2026-1109-001。")


# 分层模式用的小队队长
def detect_lead_brain(agent, task, context):
    if "汇总" in task:
        return "感知小队回报：\n" + context
    return "\n".join([
        "监控员: 读取 A-12 告警",
        "诊断员: 定位温度升高的原因",
    ])


def handle_lead_brain(agent, task, context):
    if "汇总" in task:
        return "处置小队回报：\n" + context
    return "\n".join([
        "处置员: 迁移业务并创建工单",
        "汇报员: 输出事件报告",
    ])


def strategist_brain(agent, task, context):
    if "汇总" in task or "战报" in task:
        return "运维行动总结：\n" + context
    return "\n".join([
        "感知小队: 发现告警并定位原因",
        "处置小队: 执行处置并汇报结果",
    ])


# 阶段模式：每个阶段只读写结构化状态
def phase_monitor(agent, task, context):
    return f"alarms={agent.use('get_alarms', 'A-12')}"


def phase_diagnose(agent, task, context):
    state = json.loads(context)
    asset = agent.use("get_asset", "srv-web-01")
    metrics = agent.use("get_metrics", "srv-web-01")
    return (f"asset={asset}\n"
            f"metrics={metrics}\n"
            "cause=风扇故障导致散热不足")


def phase_fix(agent, task, context):
    return (f"action={agent.use('migrate_service', 'order-service')}\n"
            f"ticket={agent.use('open_ticket', 'A-12')}")


def phase_report(agent, task, context):
    state = json.loads(context)
    return ("report=【事件报告】A-12 温度告警："
            f"根因 {state.get('cause')}；处置 {state.get('action')}")


# 黑板模式：调度器只看黑板上的 key，不关心谁说了什么
def controller_brain(agent, task, context):
    if "[alarms]" not in context:
        return "监控员"
    if "[cause]" not in context:
        return "诊断员"
    if "[action]" not in context:
        return "处置员"
    if "[report]" not in context:
        return "汇报员"
    return "DONE"


def bb_monitor(agent, task, context):
    alarms = agent.use("get_alarms", "A-12")
    return (f"FACT: alarms = {alarms}\n"
            "HINT: suspect = 机柜温度与风扇转速可能相关")


def bb_diagnose(agent, task, context):
    asset = agent.use("get_asset", "srv-web-01")
    metrics = agent.use("get_metrics", "srv-web-01")
    return (f"FACT: asset = {asset}\n"
            f"FACT: cause = srv-web-01 风扇故障导致散热不足\n"
            f"HINT: metrics = {metrics}")


def bb_fix(agent, task, context):
    action = agent.use("migrate_service", "order-service")
    ticket = agent.use("open_ticket", "A-12")
    return f"FACT: action = {action}\nFACT: ticket = {ticket}"


def bb_reporter(agent, task, context):
    return ("FACT: report = 【事件报告】A-12 机柜温度告警："
            "根因是 srv-web-01 风扇故障；已迁移 order-service 到 "
            "srv-web-02 并创建工单 INC-2026-1109-001。")


# ---------------------------------------------------------------------------
# 组装：同一批角色，四种协作方式
# ---------------------------------------------------------------------------
def _maker(llm: LLMClient | None):
    """返回建 agent 的工厂：有 LLM 用 LLM 大脑，否则用脚本大脑。"""
    def make(name, role, tools, scripted_brain):
        if llm is None:
            brain = scripted_brain
        elif tools:
            brain = tool_brain(llm)
        else:
            brain = text_brain(llm)
        return Agent(name, role, brain, tools)
    return make


def build_centralized(llm=None) -> CentralizedOrchestrator:
    make = _maker(llm)
    planner = make("总控", ROLE_PLANNER, [], planner_brain)
    workers = {
        "监控员": make("监控员", ROLE_MONITOR, [TOOL_ALARMS], monitor_brain),
        "诊断员": make("诊断员", ROLE_DIAGNOSER,
                    [TOOL_ASSET, TOOL_METRICS], diagnose_brain),
        "处置员": make("处置员", ROLE_FIXER,
                    [TOOL_MIGRATE, TOOL_TICKET], fix_brain),
        "汇报员": make("汇报员", ROLE_REPORTER, [], report_brain),
    }
    return CentralizedOrchestrator(planner, workers)


def build_hierarchical(llm=None) -> HierarchicalOrchestrator:
    make = _maker(llm)

    detect_team = CentralizedOrchestrator(
        make("感知队长", ROLE_PLANNER, [], detect_lead_brain),
        {
            "监控员": make("监控员", ROLE_MONITOR, [TOOL_ALARMS], monitor_brain),
            "诊断员": make("诊断员", ROLE_DIAGNOSER,
                        [TOOL_ASSET, TOOL_METRICS], diagnose_brain),
        },
    )
    handle_team = CentralizedOrchestrator(
        make("处置队长", ROLE_PLANNER, [], handle_lead_brain),
        {
            "处置员": make("处置员", ROLE_FIXER,
                        [TOOL_MIGRATE, TOOL_TICKET], fix_brain),
            "汇报员": make("汇报员", ROLE_REPORTER, [], report_brain),
        },
    )
    strategist = make("指挥官", ROLE_STRATEGIST, [], strategist_brain)
    return HierarchicalOrchestrator(
        strategist, {"感知小队": detect_team, "处置小队": handle_team})


def build_pipeline(llm=None) -> PhasePipeline:
    make = _maker(llm)
    return PhasePipeline([
        make("监控阶段", ROLE_MONITOR, [TOOL_ALARMS], phase_monitor),
        make("诊断阶段", ROLE_DIAGNOSER,
             [TOOL_ASSET, TOOL_METRICS], phase_diagnose),
        make("处置阶段", ROLE_FIXER, [TOOL_MIGRATE, TOOL_TICKET], phase_fix),
        make("汇报阶段", ROLE_REPORTER, [], phase_report),
    ])


def build_blackboard(llm=None) -> BlackboardOrchestrator:
    make = _maker(llm)
    controller = make("调度器", ROLE_CONTROLLER, [], controller_brain)
    agents = {
        "监控员": make("监控员", ROLE_MONITOR, [TOOL_ALARMS], bb_monitor),
        "诊断员": make("诊断员", ROLE_DIAGNOSER,
                    [TOOL_ASSET, TOOL_METRICS], bb_diagnose),
        "处置员": make("处置员", ROLE_FIXER,
                    [TOOL_MIGRATE, TOOL_TICKET], bb_fix),
        "汇报员": make("汇报员", ROLE_REPORTER, [], bb_reporter),
    }
    return BlackboardOrchestrator(controller, agents)

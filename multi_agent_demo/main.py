#!/usr/bin/env python3
"""
简易多智能体平台 —— 命令行入口。

默认只输出关键步骤（谁做了什么、最终结果）。
想看完整黑板变化和汇总过程，加 --verbose。

用法：
    python main.py                 # 黑板协作（推荐先看）
    python main.py centralized     # 中心化编排
    python main.py hierarchical    # 战略—战术分层
    python main.py pipeline        # 阶段/任务域分解
    python main.py all             # 依次跑四种，便于对比
    python main.py --verbose       # 输出完整过程
    python main.py --llm           # 换成真实 LLM（需 OPENAI_API_KEY）

所有工具均为模拟实现，不发起真实网络请求或运维操作。
"""
from __future__ import annotations

import argparse
import sys

from framework import LLMClient, set_verbose
from scenario_ops import (build_blackboard, build_centralized,
                          build_hierarchical, build_pipeline)

BUILDERS = {
    "blackboard": ("黑板协作（推荐先看）", build_blackboard),
    "centralized": ("中心化编排", build_centralized),
    "hierarchical": ("战略—战术分层", build_hierarchical),
    "pipeline": ("阶段/任务域分解", build_pipeline),
}

GUIDE = """\
【阅读提示】观察多智能体的四个要素：
  1. 角色分工：每个角色只调用自己的工具
  2. 共享状态：所有结论都写到同一块黑板
  3. 协调：调度器只看黑板，决定下一个该谁行动
  4. 共同目标：四个角色都在处理同一个机柜告警
"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="简易多智能体平台演示")
    parser.add_argument(
        "pattern", nargs="?", default="blackboard",
        choices=[*BUILDERS, "all"],
        help="要运行的协作模式，默认 blackboard")
    parser.add_argument(
        "--task", default="A-12 机柜温度过高，请定位原因并完成处置",
        help="交给多智能体系统的总任务")
    parser.add_argument(
        "--llm", action="store_true",
        help="使用真实 LLM（需要 OPENAI_API_KEY 环境变量）")
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="LLM 模型名")
    parser.add_argument(
        "--verbose", action="store_true",
        help="输出完整黑板变化与汇总过程")
    args = parser.parse_args()

    set_verbose(args.verbose)
    if args.verbose:
        print(GUIDE)

    print(f"任务：{args.task}")

    names = list(BUILDERS) if args.pattern == "all" else [args.pattern]
    llm = LLMClient(model=args.model) if args.llm else None

    for index, name in enumerate(names, 1):
        label, build = BUILDERS[name]
        if args.verbose:
            print(f"\n\n{'#' * 58}")
            print(f"# [{index}/{len(names)}] {label}")
            print(f"{'#' * 58}")
        orchestrator = build(llm)
        result = orchestrator.run(args.task)
        print("\n最终输出：")
        print(result)


if __name__ == "__main__":
    main()

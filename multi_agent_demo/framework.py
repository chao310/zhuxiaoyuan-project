"""
极简多智能体框架核心。

只有三个概念：
    Tool   工具 —— 一个有名字的函数
    Agent  智能体 —— 名字 + 角色 + 工具 + 大脑
    Brain  大脑 —— 一个函数 (agent, task, context) -> str

默认使用"脚本化大脑"，不需要任何 API key 就能运行。
想接真实 LLM 时，用 text_brain() / tool_brain() 生成大脑即可。
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    """一个有名字、可被智能体调用的函数。"""
    name: str
    description: str
    fn: Callable[[str], str]

    def __call__(self, arg: str) -> str:
        return self.fn(arg)


# ---------------------------------------------------------------------------
# 智能体
# ---------------------------------------------------------------------------
# 大脑的签名：拿到「自己、任务、上下文」，返回一段文本
Brain = Callable[["Agent", str, str], str]


class Agent:
    """一个只知道自己角色和工具的智能体。"""

    def __init__(self, name: str, role: str, brain: Brain,
                 tools: Iterable[Tool] = ()):
        self.name = name
        self.role = role
        self.brain = brain
        self.tools = {t.name: t for t in tools}
        self.history: list[str] = []
        self.calls = 0

    def tool_spec(self) -> str:
        if not self.tools:
            return "（无）"
        return "\n".join(f"- {t.name}: {t.description}"
                         for t in self.tools.values())

    def use(self, tool_name: str, arg: str) -> str:
        """调用自己的某个工具。没权限就报错——这是角色边界的一部分。"""
        if tool_name not in self.tools:
            raise KeyError(f"「{self.name}」没有工具「{tool_name}」")
        return self.tools[tool_name](arg)

    def run(self, task: str, context: str = "") -> str:
        self.calls += 1
        result = self.brain(self, task, context)
        self.history.append(result)
        return result

    def __repr__(self) -> str:
        return f"<Agent {self.name}>"


# ---------------------------------------------------------------------------
# 日志：用来观察协作过程
# ---------------------------------------------------------------------------
VERBOSE = False


def set_verbose(flag: bool) -> None:
    """默认只输出关键步骤；开启后输出完整黑板与汇总过程。"""
    global VERBOSE
    VERBOSE = flag


def is_verbose() -> bool:
    return VERBOSE


def log(who: str, text: str, indent: int = 0) -> None:
    pad = "  " * indent
    print(f"{pad}▶ {who}")
    for line in str(text).strip().splitlines():
        print(f"{pad}  {line}")


def header(title: str) -> None:
    if VERBOSE:
        print(f"\n{'=' * 58}\n{title}\n{'=' * 58}")
    else:
        print(f"\n── {title} ──")


# ---------------------------------------------------------------------------
# 解析小工具
# ---------------------------------------------------------------------------
def parse_assignments(text: str) -> list[tuple[str, str]]:
    """把 `角色名: 子任务` 形式的文本解析成 [(角色, 子任务)]。"""
    pairs: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*• ").replace("：", ":")
        if ":" not in line:
            continue
        left, _, right = line.partition(":")
        left, right = left.strip(), right.strip()
        if left and right:
            pairs.append((left, right))
    return pairs


def parse_state(text: str) -> dict[str, str]:
    """把 `key=value` 行解析成字典，用于阶段间的结构化状态传递。"""
    state: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key.replace("_", "").isalnum() and not key[0].isdigit():
            state[key] = value.strip()
    return state


# ---------------------------------------------------------------------------
# 可选：接真实 LLM（OpenAI 兼容接口，只用标准库）
# ---------------------------------------------------------------------------
class LLMClient:
    """最小 LLM 客户端，支持 OpenAI 兼容的 /chat/completions 接口。"""

    def __init__(self, model: str = "gpt-4o-mini",
                 api_key: str | None = None,
                 base_url: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        if not self.api_key:
            raise RuntimeError(
                "没有找到 OPENAI_API_KEY。请先设置环境变量，"
                "或去掉 --llm 参数使用内置的脚本化大脑。")

    def complete(self, system: str, user: str,
                 temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


REACT_FORMAT = """你可以调用以下工具：
{tools}

需要调用工具时，只输出一行：
TOOL: 工具名 | 参数
拿到结果后继续思考；任务完成时只输出一行：
FINAL: 你的最终回答
除此以外不要输出任何内容。"""


def text_brain(client: LLMClient) -> Brain:
    """不需要工具的智能体：一次问答。"""
    def brain(agent: Agent, task: str, context: str) -> str:
        system = f"你是多智能体系统中的「{agent.name}」。{agent.role}"
        user = f"任务：{task}\n共享上下文：\n{context or '（空）'}"
        return client.complete(system, user).strip()
    return brain


def tool_brain(client: LLMClient) -> Brain:
    """需要工具的智能体：一个极简 ReAct 循环。"""
    def brain(agent: Agent, task: str, context: str) -> str:
        system = (f"你是多智能体系统中的「{agent.name}」。{agent.role}\n"
                  + REACT_FORMAT.format(tools=agent.tool_spec()))
        observations: list[str] = []
        for _ in range(5):
            user = (f"总任务：{task}\n共享上下文：\n{context or '（空）'}\n"
                    f"已有观察：\n" + ("\n".join(observations) or "（空）"))
            reply = client.complete(system, user).strip()
            if reply.startswith("TOOL:"):
                body = reply[5:].strip()
                name, _, arg = body.partition("|")
                try:
                    obs = agent.use(name.strip(), arg.strip())
                except KeyError as exc:
                    obs = f"错误：{exc}"
                observations.append(
                    f"调用 {name.strip()}({arg.strip()}) -> {obs}")
                continue
            if reply.startswith("FINAL:"):
                return reply[6:].strip()
            return reply
        return "\n".join(observations)
    return brain

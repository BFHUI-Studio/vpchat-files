import asyncio
import json
import os
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from routes.api import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Core Engine] - %(message)s'
)
logger = logging.getLogger(__name__)


class PactConfig:
    """深度学习计划生成服务配置参数。"""
    API_URL = os.getenv("API_URL", "https://open.xiaojingai.com/v1/chat/completions")
    API_KEY = os.getenv("API_KEY", "sk-yrGqTCnZUWh7cJHwC6F5B4Eb324340F89b3598Aa89D57e89")
    MODEL_ID = os.getenv("MODEL_ID", "gemini-1.5-pro-latest")
    TIMEOUT = 36000.0
    # 请求重试次数
    MAX_RETRY = 2


class GenerateTaskRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=300)
    mode: str = Field("quick", pattern="^(quick|deep)$")


class AdjustTaskRequest(BaseModel):
    current_task: Dict[str, Any]
    insights: List[str]


class AskCoachRequest(BaseModel):
    question: str
    context: Dict[str, Any]


class GenerateTaskFromDocRequest(BaseModel):
    content: str = Field(..., min_length=100)  # 保证文本具备一定长度


class InitiateRemedialLoopRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=100)
    struggle_point: str = Field(..., min_length=5, max_length=200)


class ElaborateTaskRequest(BaseModel):
    topic: str
    task_description: str


class GenerateTagsRequest(BaseModel):
    plan: Dict[str, Any]


class ExpandTagRequest(BaseModel):
    tag: str
    context: Dict[str, Any] = {}


class TagRelationsRequest(BaseModel):
    tags: List[str]


class CoreEngine:
    """
    核心引擎，负责调用语言模型API，执行学习计划生成与调整任务。
    """

    def __init__(self):
        self.config = PactConfig()
        logger.info("核心引擎已启动，准备接受请求。")

    async def _call_model(self, prompt: str, temperature: float) -> str:
        """
        调用语言模型接口获取响应。
        :param prompt: 发送给模型的提示语
        :param temperature: 模型生成的随机程度控制参数
        :return: 模型回复文本
        """
        if not self.config.API_KEY or self.config.API_KEY == "YOUR_API_KEY":
            raise ConnectionError("未配置有效的API_KEY，无法请求模型服务。")

        payload = {
            "model": self.config.MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 3000,
        }

        for attempt in range(self.config.MAX_RETRY + 1):
            try:
                async with httpx.AsyncClient(timeout=self.config.TIMEOUT) as client:
                    response = await client.post(
                        self.config.API_URL,
                        headers={"Authorization": f"Bearer {self.config.API_KEY}"},
                        json=payload
                    )
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                    if not content:
                        raise ValueError("模型返回内容为空。")
                    logger.info(f"模型调用成功 (尝试 {attempt + 1})。")
                    return content.strip()
            except Exception as e:
                logger.warning(f"模型调用第 {attempt + 1} 次失败: {e}")
                if attempt >= self.config.MAX_RETRY:
                    logger.error("所有模型调用尝试均失败。")
                    raise ConnectionAbortedError("与模型服务的连接已中断。") from e
                await asyncio.sleep(1.5 ** attempt)
        raise ConnectionAbortedError("意外错误，流程未按预期结束。")

    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        """
        从模型返回的文本中提取JSON数据。
        """
        logger.info("尝试从模型文本中提取JSON结构。")
        try:
            # 1) 优先检测 ```json ... ``` 或 ``` ... ``` 代码块，提取块内全部内容尝试解析
            #    使用贪婪匹配以包含嵌套对象，避免 `.*?` 过早截断
            code_block_match = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", raw_text, re.IGNORECASE)
            if code_block_match:
                logger.info("检测到 Markdown 代码块格式 JSON。")
                candidate = code_block_match.group(1).strip()
                # 如果代码块本身未必以 { 开头，我们尝试截取首尾花括号之间内容
                brace_start = candidate.find('{')
                brace_end = candidate.rfind('}')
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    candidate = candidate[brace_start:brace_end + 1]
                return json.loads(candidate)

            # 2) 尝试直接在文本中定位首尾花括号整体解析
            json_start = raw_text.find('{')
            json_end = raw_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                logger.info("检测到内嵌 JSON 字符串。")
                json_str = raw_text[json_start:json_end + 1]
                return json.loads(json_str)

            cleaned = raw_text.strip("` \n")
            try:
                return json.loads(cleaned)
            except Exception:
                pass

            import ast
            try:
                parsed = ast.literal_eval(cleaned)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except Exception:
                pass

            try:
                import simplejson as sjson  # type: ignore
                return sjson.loads(cleaned)
            except Exception:
                pass

            raise ValueError("未能识别有效的JSON格式内容。")
        except Exception as e:
            logger.error(f"提取JSON失败: {e}")
            raise ValueError("无法解析模型返回内容为JSON。") from e

    async def generate_learning_plan(self, topic: str, mode: str) -> Dict[str, Any]:
        """
        根据学习主题生成学习计划。
        此方法采用分步尝试策略，以确保生成结果的质量和完整性。
        """
        logger.info(f"开始为主题'{topic}'生成学习计划，模式: {mode}。")

        logger.info("阶段一：标准格式JSON计划生成。")
        prompt_1 = (
            f'请生成一个关于"{topic}"的{ "详细" if mode == "deep" else "简明" }学习计划，'
            '必须仅返回完整的JSON对象，结构示例如下'
            '{"plan": {"title": "...", "tasks": [{"description": "...", "subtasks": ["...", "..."]}]}}'
        )
        try:
            response_1 = await self._call_model(prompt_1, temperature=0.3)
            plan_1 = self._extract_json(response_1)
            if plan_1.get("plan", {}).get("tasks", [{}])[0].get("subtasks"):
                logger.info("成功生成标准JSON格式学习计划。")
                return plan_1
        except Exception as e:
            logger.warning(f"阶段一失败，准备进行降级尝试: {e}")

        logger.info("阶段二：生成Python列表格式任务。")
        prompt_2 = (
            f'请提供一个关于"{topic}"的核心学习步骤列表，'
            '要求返回Python列表字符串格式，例如：["步骤1", "步骤2", "步骤3"]，无额外说明。'
        )
        try:
            response_2 = await self._call_model(prompt_2, temperature=0.5)
            subtasks = eval(response_2)  # 注意：eval仅用于信任环境下，务必注意安全风险
            if isinstance(subtasks, list) and subtasks:
                logger.info("Python列表格式任务生成成功，进行结构组装。")
                return {
                    "plan": {
                        "title": f"{topic}（结构重组）",
                        "tasks": [{
                            "description": "此计划为自动结构重组版本，请结合实际调整。",
                            "subtasks": subtasks
                        }]
                    }
                }
        except Exception as e:
            logger.warning(f"阶段二失败，准备最终保底方案: {e}")

        logger.info("阶段三：从自然语言文本中提取学习步骤。")
        try:
            base_prompt = f"详细描述学习'{topic}'所需的关键步骤。"
            raw_text = await self._call_model(base_prompt, temperature=0.7)

            extraction_prompt = (
                f'请从以下文本中提取关于学习"{topic}"的重要步骤，返回Python列表字符串格式，示例：["步骤1", "步骤2"]，不添加解释。\n'
                f"文本内容:\n---\n{raw_text}\n---"
            )
            response_3 = await self._call_model(extraction_prompt, temperature=0.2)
            subtasks_final = eval(response_3)
            if isinstance(subtasks_final, list) and subtasks_final:
                logger.info("成功从文本中提取出核心学习步骤。")
                return {
                    "plan": {
                        "title": f"{topic}（文本提取版）",
                        "tasks": [{
                            "description": "此计划基于文本内容自动整理生成，需结合实际参考。",
                            "subtasks": subtasks_final
                        }]
                    }
                }
        except Exception as e:
            logger.error(f"文本提取阶段失败: {e}")

        logger.error("所有生成阶段均未成功，启用默认基础计划。")
        return {
            "plan": {
                "title": f"{topic}（默认基础计划）",
                "tasks": [{
                    "description": "未能提取有效学习计划，提供基础学习框架。",
                    "subtasks": [
                        f"理解'{topic}'的基本定义。",
                        f"学习'{topic}'相关的主要概念。",
                        "开展相关主题的基础阅读与实践。"
                    ]
                }]
            }
        }

    async def generate_plan_from_document(self, content: str) -> Dict[str, Any]:
        """
        根据输入的文本文档自动提炼主题，并生成相应学习计划。
        """
        logger.info("开始处理文档内容以生成学习计划。")

        prompt_extract = (
            "请阅读以下文档内容，提炼出该文档的主要学习主题。"
            "请仅返回一个简洁准确的主题名称，不要包含任何额外解释。\n"
            "文档内容:\n---\n"
            f"{content[:4000]}"
            "\n---"
        )
        try:
            topic = await self._call_model(prompt_extract, temperature=0.1)
            logger.info(f"成功提炼主题: '{topic}'")

            plan = await self.generate_learning_plan(topic.strip(), mode='deep')

            if plan.get("plan", {}).get("tasks"):
                plan["plan"]["tasks"][0]["description"] = "本学习计划基于用户导入文档自动生成。"

            return plan
        except Exception as e:
            logger.error(f"基于文档生成计划失败: {e}", exc_info=True)
            raise ValueError(f"文档解析及学习计划生成失败，错误信息: {e}")

    async def refine_plan(self, current_task: Dict[str, Any], insights: List[str]) -> Dict[str, Any]:
        """
        根据用户反馈的学习体会，对当前计划进行调整和优化。
        尝试多次调用模型，确保生成结构合理的JSON计划。
        """
        logger.info("启动学习计划调整流程。")

        base_prompt = (
            "您是一位学习规划专家，基于用户当前的学习计划和提供的学习体会，"
            "请调整并优化学习计划。\n\n"
            f"当前计划(JSON格式):\n{json.dumps(current_task, indent=2)}\n\n"
            "用户体会:\n- " + "\n- ".join(insights) + "\n\n"
            "请仅返回调整后的学习计划JSON，不要添加任何解释或额外文本，"
            "格式需保持为{ \"plan\": { \"title\": str, \"tasks\": [{\"description\": str, \"subtasks\": [str]}] } }，"
            "确保每个任务至少包含一个子任务。"
        )

        for attempt, temp in enumerate([0.3, 0.2, 0.1], start=1):
            prompt = base_prompt + f"\n请返回调整后的学习计划。尝试次数: {attempt}。"
            try:
                logger.info(f"调整尝试 {attempt}，模型温度: {temp}")
                response = await self._call_model(prompt, temperature=temp)
                adjusted_plan = self._extract_json(response)
                if adjusted_plan.get("plan", {}).get("tasks", [{}])[0].get("subtasks"):
                    logger.info("成功生成调整后的学习计划。")
                    return adjusted_plan
                else:
                    raise ValueError("返回数据缺少必要字段。")
            except Exception as e:
                logger.warning(f"尝试 {attempt} 失败: {e}")

        logger.error("所有计划调整尝试失败，返回原计划。")
        return current_task

    async def answer_coach_question(self, question: str, context: Dict[str, Any]) -> str:
        """
        回答用户在学习过程中提出的问题，尽可能结合当前学习上下文。
        """
        logger.info(f"开始回答用户问题: {question}")

        prompt = (
            f"你是一名耐心专业的学习辅导员，用户正在学习主题：'{context.get('topic', '当前主题')}'，"
            f"现针对问题：'{question}'寻求帮助。\n"
            f"当前的学习内容包括子任务：{json.dumps(context.get('subtasks', []), ensure_ascii=False)}\n\n"
            "请简明扼要地回答该问题，若问题超出主题范围，请礼貌引导用户回到主题。"
        )

        try:
            answer = await self._call_model(prompt, temperature=0.7)
            logger.info("成功返回辅导回答。")
            return answer
        except Exception as e:
            logger.error(f"辅导回答失败: {e}")
            return "抱歉，当前无法回答您的问题，请稍后再试。"

    async def generate_remedial_plan(self, topic: str, struggle_point: str) -> Dict[str, Any]:
        """
        针对用户困难点，生成针对性强化练习计划。
        """
        logger.info(f"开始生成强化练习，主题: '{topic}'，困难点: '{struggle_point}'.")

        prompt = (
            f"你是一位严谨而高效的学习导师。"
            f"学生在学习'{topic}'时，针对'{struggle_point}'遇到了困难。"
            "请生成一个简短（1-2个任务）、聚焦、具体的强化练习计划，"
            "必须仅返回JSON格式的完整结构：\n"
            '{"plan": {"title": "强化练习: [具体困难点]", "tasks": [{"description": "...", "subtasks": ["...", "..."]}]}}\n'
            "请确保任务描述和子任务聚焦具体困难点。"
        )

        try:
            response = await self._call_model(prompt, temperature=0.1)
            remedial_plan = self._extract_json(response)
            if (
                remedial_plan.get("plan", {}).get("title") and
                remedial_plan.get("plan", {}).get("tasks") and
                remedial_plan["plan"]["tasks"][0].get("subtasks")
            ):
                logger.info("成功生成强化练习计划。")
                return remedial_plan
            else:
                raise ValueError("返回的强化练习计划结构不完整。")
        except Exception as e:
            logger.error(f"生成强化练习计划失败: {e}", exc_info=True)
            # 提供基础强化任务作为回退方案
            return {
                "plan": {
                    "title": f"强化练习: {struggle_point} (默认)",
                    "tasks": [{
                        "description": f"针对'{struggle_point}'，请复习基本定义和核心概念。",
                        "subtasks": [
                            f"重新学习'{struggle_point}'相关定义。",
                            f"查找至少两个'{struggle_point}'的示例。",
                            f"尝试用自己的话解释'{struggle_point}'。"
                        ]
                    }]
                }
            }

    async def elaborate_task(self, topic: str, task_description: str) -> str:
        """针对当前任务，提供更详细的说明或步骤。"""
        logger.info(f"深化任务中，主题: '{topic}'，任务: '{task_description}'.")
        prompt = (
            f"作为一名学习导师，请针对学习主题 '{topic}' 中的任务 '{task_description}' 提供更详细的解释、背景知识或具体的执行步骤。"
            "请直接返回清晰、可执行的说明文本，帮助学生更好地理解和完成这个任务。"
        )
        try:
            elaboration = await self._call_model(prompt, temperature=0.4)
            logger.info("成功生成任务深化内容。")
            return elaboration
        except Exception as e:
            logger.error(f"任务深化失败: {e}", exc_info=True)
            return "抱歉，无法提供更详细的说明，请检查任务内容或稍后再试。"

    async def generate_tags(self, plan: Dict[str, Any]) -> List[str]:
        """根据学习计划任务生成简洁标签列表。"""
        logger.info("开始根据学习计划生成标签云。")

        # 收集所有待学习的文本条目
        tasks_text: List[str] = []
        for task in plan.get("tasks", []):
            if isinstance(task, dict):
                if desc := task.get("description"):
                    tasks_text.append(str(desc))
                subtasks = task.get("subtasks", [])
                if isinstance(subtasks, list):
                    tasks_text.extend(map(str, subtasks))

        tasks_concat = " | ".join(tasks_text)[:3500]

        prompt = (
            "请从以下学习任务文本中提炼 5~15 个高度概括的学习标签，"  # 数量大致范围
            "标签需简短有力，每个标签不超过 4 个汉字或 3 个英文单词，使用驼峰或纯中文均可，" 
            "不要编号，不要解释，只返回 JSON 数组字符串，例如 [\"矩阵\", \"特征值\"].\n\n"
            f"学习任务文本:\n{tasks_concat}"
        )

        try:
            response = await self._call_model(prompt, temperature=0.2)
            tags = self._extract_json(response)
            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                logger.info("成功生成标签云。")
                return tags
            raise ValueError("模型返回的标签格式不正确。")
        except Exception as e:
            logger.error(f"生成标签云失败: {e}")
            # 回退：从任务文本中随机截取关键词
            fallback = list({txt.strip()[:10] for txt in tasks_text})[:10]
            return fallback

    async def expand_tag(self, tag: str, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """扩展关键词，返回相关关键词与搜索查询。"""
        logger.info(f"开始扩展关键词: {tag}")
        prompt = (
            f"针对学习关键词 '{tag}'，列出 5~10 个紧密相关的进阶关键词，"
            "以及 3~5 条可直接复制到搜索引擎的查询语句。"
            "请仅返回 JSON 格式，例如 {\"related\": [\"A\", \"B\"], \"search\": [\"query1\"]}."
        )
        try:
            response = await self._call_model(prompt, temperature=0.3)
            data = self._extract_json(response)
            if (
                isinstance(data, dict)
                and isinstance(data.get("related"), list)
                and isinstance(data.get("search"), list)
            ):
                return data  # type: ignore
            raise ValueError("返回数据结构不正确")
        except Exception as e:
            logger.warning(f"扩展关键词失败: {e}")
            # fallback simple
            return {
                "related": [tag],
                "search": [f"{tag} 学习"]
            }

    async def tag_relations(self, tags: List[str]) -> Dict[str, Any]:
        """生成简单标签关系图，使用语言模型给出关联度(0-1)，返回 edge list"""
        logger.info("开始生成标签关系图。")
        if len(tags) <= 1:
            return {"nodes": tags, "edges": []}

        prompt = (
            "以下是学习标签列表：" + ", ".join(tags) + "。\n"
            "请为我返回它们之间的关联度评分(0~1)，只返回 JSON 对象形如 {\"edges\": [{\"source\": \"A\", \"target\": \"B\", \"weight\": 0.6}, ...]}，"
            "最多返回 30 条高关联度边，weight > 0.3。不要解释。"
        )
        try:
            res = await self._call_model(prompt, temperature=0.2)
            data = self._extract_json(res)
            if isinstance(data, dict) and isinstance(data.get("edges"), list):
                return {"nodes": tags, "edges": data["edges"]}
        except Exception as e:
            logger.warning(f"AI 生成关系失败: {e}")

        # fallback: 全连接 weight 0.5 的简单图
        edges = []
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                edges.append({"source": tags[i], "target": tags[j], "weight": 0.5})
        return {"nodes": tags, "edges": edges[:30]}

    # ---------------- 笔记整理 ----------------

    async def organize_notes(self, notes: List[str]) -> Dict[str, Any]:
        """根据考古学家提示整理笔记，返回结构化报告。"""
        logger.info("开始整理笔记，共 %d 条。", len(notes))
        joined_notes = "\n\n".join(f"- {n}" for n in notes[:100])  # 限制长度，防止 prompt 过长

        prompt = (
            "你是一位'灵魂的考古学家'，请按照以下 methodology 对笔记进行考古式分析，并以 JSON 格式输出：\n"
            "methodology: strata_identification -> connection_mapping -> unearthing_the_core_question.\n"
            "output_format: {title: str, sections: {\"Key Artifacts Discovered\": [str], \"Structural Analysis\": str, \"The Ghost in the Rubble\": str}}\n"
            "笔记内容如下（原样提供）:\n---\n" + joined_notes + "\n---"
        )
        try:
            res = await self._call_model(prompt, temperature=0.3)
            report = self._extract_json(res)
            if isinstance(report, dict):
                return report  # type: ignore
            raise ValueError("模型返回格式不正确")
        except Exception as e:
            logger.error(f"笔记整理失败: {e}")
            # fallback simple structure
            return {
                "title": "Dig Site Report",
                "sections": {
                    "Key Artifacts Discovered": notes[:5],
                    "Structural Analysis": "自动整理失败，返回原始笔记列表。",
                    "The Ghost in the Rubble": "N/A"
                }
            }


app = FastAPI(
    title="智能学习计划生成引擎",
    description="后端服务，负责生成与调整学习计划，辅助用户实现高效学习。",
    version="1.0.0"
)

# 配置CORS，允许所有来源访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = CoreEngine()

app.include_router(api_router, prefix="/api")


@app.post("/api/generate-task")
async def generate_task_endpoint(request: GenerateTaskRequest):
    """基于主题和模式生成学习计划。"""
    logger.info(f"接收到学习计划生成请求，主题: {request.topic}，模式: {request.mode}")
    try:
        plan = await engine.generate_learning_plan(request.topic, request.mode)
        return JSONResponse(content={"success": True, "plan": plan.get("plan")})
    except Exception as e:
        logger.error(f"学习计划生成失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="生成学习计划失败。")


@app.api_route("/api/generate-task-stream", methods=["GET", "POST"])
async def generate_task_stream_endpoint(request: Request):
    """以流式方式生成学习计划，支持前端渐进接收。"""
    if request.method == "GET":
        async def event_generator_get():
            yield f"data: {json.dumps({'role':'assistant','content': '请使用POST方法请求此接口。'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_generator_get(), media_type="text/event-stream")

    try:
        body = await request.json()
        req_data = GenerateTaskRequest(**body)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的请求体。")

    logger.info(f"流式生成请求，主题: {req_data.topic}，模式: {req_data.mode}")

    async def event_generator_post():
        try:
            plan = await engine.generate_learning_plan(req_data.topic, req_data.mode)
            plan_json = json.dumps(plan.get("plan", {}), ensure_ascii=False, indent=2)
            for ch in plan_json:
                yield f"data: {json.dumps({'role':'assistant','content': ch})}\n\n"
                await asyncio.sleep(0.01)
        except Exception as e:
            error_msg = f"生成失败: {e}"
            for ch in error_msg:
                yield f"data: {json.dumps({'role':'assistant','content': ch})}\n\n"
                await asyncio.sleep(0.01)
        finally:
            yield "data: [DONE]\n\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_generator_post(), media_type="text/event-stream", headers=headers)


@app.post("/api/generate-task-from-document")
async def generate_task_from_document_endpoint(request: GenerateTaskFromDocRequest):
    """根据上传的文档文本生成学习计划。"""
    logger.info("接收到文档内容，请求生成学习计划。")
    try:
        plan = await engine.generate_plan_from_document(request.content)
        return JSONResponse(content={"success": True, "plan": plan.get("plan")})
    except Exception as e:
        logger.error(f"文档生成计划失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="无法基于文档生成学习计划。")


@app.post("/api/adjust-task")
async def adjust_task_endpoint(request: AdjustTaskRequest):
    """根据用户反馈调整学习计划。"""
    try:
        adjusted_plan = await engine.refine_plan(request.current_task, request.insights)
        # 与 /api/generate-task 保持一致，只返回内部 "plan" 对象
        inner_plan: Dict[str, Any] | None = None
        if isinstance(adjusted_plan, dict):
            inner_plan = adjusted_plan.get("plan", adjusted_plan)  # 若已是内部结构则直接返回
        return {"success": True, "plan": inner_plan}
    except Exception as e:
        logger.error(f"调整学习计划失败: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": str(e)},
        )


@app.post("/api/ask-coach")
async def ask_coach_endpoint(request: AskCoachRequest):
    """回答用户关于学习的具体问题。"""
    try:
        answer = await engine.answer_coach_question(request.question, request.context)
        return {"success": True, "answer": answer}
    except Exception as e:
        logger.error(f"辅导回答失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="辅导服务暂时不可用，请稍后重试。")


@app.post("/api/initiate-remedial-loop")
async def initiate_remedial_loop_endpoint(request: InitiateRemedialLoopRequest):
    """生成针对用户学习薄弱点的强化练习计划。"""
    logger.info(f"接收到强化练习请求，主题: {request.topic}, 困难点: {request.struggle_point}")
    try:
        remedial_plan = await engine.generate_remedial_plan(request.topic, request.struggle_point)
        return JSONResponse(content={"success": True, "plan": remedial_plan.get("plan")})
    except Exception as e:
        logger.error(f"强化练习生成失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="生成强化练习计划失败。")


@app.post("/api/elaborate-task")
async def elaborate_task_endpoint(request: ElaborateTaskRequest):
    """根据用户请求，深化当前学习任务。"""
    logger.info(f"接收到任务深化请求，任务: {request.task_description}")
    try:
        elaboration = await engine.elaborate_task(request.topic, request.task_description)
        return {"success": True, "elaboration": elaboration}
    except Exception as e:
        logger.error(f"任务深化接口失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="任务深化失败。")


@app.get("/")
async def root():
    return {"message": "智能学习引擎运行中，等待请求。"}


# ---------------- 用户反馈 ----------------

class FeedbackRequest(BaseModel):
    id: str
    positive: bool
    comment: str = ''


@app.post("/api/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    logger.info(f"收到用户反馈: 内容ID {req.id}，反馈: {'正面' if req.positive else '负面'}，评论: '{req.comment}'.")
    # 在真实应用中，这里会写入数据库或分析系统
    # 例如: database.save_feedback(feedback_id=req.id, is_positive=req.positive, comment=req.comment)
    print(f"Feedback Received - ID: {req.id}, Positive: {req.positive}, Comment: {req.comment}")
    return {"success": True, "message": "感谢您的反馈！"}


@app.post("/api/generate-tags")
async def generate_tags_endpoint(request: GenerateTagsRequest):
    """根据学习计划生成标签云。"""
    try:
        tags = await engine.generate_tags(request.plan)
        return {"success": True, "tags": tags}
    except Exception as e:
        logger.error(f"标签云生成失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "detail": str(e)})


@app.post("/api/expand-tag")
async def expand_tag_endpoint(req: ExpandTagRequest):
    """扩展关键词，返回相关关键词与搜索语句"""
    try:
        result = await engine.expand_tag(req.tag, req.context)
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"扩展关键词失败: {e}")
        return JSONResponse(status_code=500, content={"success": False, "detail": str(e)})


@app.post("/api/tag-relations")
async def tag_relations_endpoint(req: TagRelationsRequest):
    try:
        graph = await engine.tag_relations(req.tags)
        return {"success": True, **graph}
    except Exception as e:
        logger.error(f"标签关系生成失败: {e}")
        return JSONResponse(status_code=500, content={"success": False, "detail": str(e)})


# ---------------- 历史记录存储 ----------------

class HistoryStore:
    """简单的学习记录存储，持久化到本地 JSON 文件。线程安全需求较低，适用于演示。"""

    _FILE_PATH = os.path.join(os.path.dirname(__file__), "history_data.json")

    def __init__(self) -> None:
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not os.path.exists(self._FILE_PATH):
            with open(self._FILE_PATH, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _read_all(self) -> List[Dict[str, Any]]:
        with open(self._FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_all(self, data: List[Dict[str, Any]]) -> None:
        with open(self._FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, topic: str, plan: Dict[str, Any]) -> str:
        """保存一条学习记录并返回其ID。"""
        record_id = str(uuid4())
        record = {
            "id": record_id,
            "topic": topic,
            "plan": plan,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        data = self._read_all()
        data.append(record)
        self._write_all(data)
        logger.info(f"已保存学习记录 {record_id} - {topic}")
        return record_id

    def list(self, keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有记录，按时间倒序，可通过关键词过滤。"""
        data = self._read_all()
        if keyword:
            data = [r for r in data if keyword.lower() in r.get("topic", "").lower()]
        # 最近的排前面
        return sorted(data, key=lambda r: r.get("created_at", ""), reverse=True)

    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """按ID获取单条记录。"""
        for r in self._read_all():
            if r.get("id") == record_id:
                return r
        return None


history_store = HistoryStore()

# ---------------- 历史记录 API ----------------


class SaveHistoryRequest(BaseModel):
    topic: str
    plan: Dict[str, Any]


@app.post("/api/history")
async def save_history_endpoint(req: SaveHistoryRequest):
    """保存学习记录。"""
    try:
        record_id = history_store.add(req.topic, req.plan)
        return {"success": True, "id": record_id}
    except Exception as e:
        logger.error(f"保存历史记录失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="无法保存学习记录。")


@app.get("/api/history")
async def list_history_endpoint(keyword: Optional[str] = None):
    """按关键词列出历史记录。"""
    try:
        records = history_store.list(keyword)
        # 仅返回概要信息
        summaries = [{"id": r["id"], "topic": r["topic"], "created_at": r["created_at"]} for r in records]
        return {"success": True, "history": summaries}
    except Exception as e:
        logger.error(f"获取历史记录失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="无法获取历史记录。")


@app.get("/api/history/{record_id}")
async def get_history_detail_endpoint(record_id: str):
    """获取单条历史记录详情。"""
    rec = history_store.get(record_id)
    if rec:
        return {"success": True, "record": rec}
    raise HTTPException(status_code=404, detail="记录不存在。")


# ---------------- 资源链接存储 ----------------

class ResourceStore:
    """用户自定义资源链接收藏，本地 JSON 文件持久化。"""

    _FILE_PATH = os.path.join(os.path.dirname(__file__), "resources_data.json")

    def __init__(self) -> None:
        if not os.path.exists(self._FILE_PATH):
            with open(self._FILE_PATH, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _read_all(self) -> List[Dict[str, Any]]:
        with open(self._FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_all(self, data: List[Dict[str, Any]]) -> None:
        with open(self._FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, title: str, url: str) -> str:
        rid = str(uuid4())
        entry = {
            "id": rid,
            "title": title,
            "url": url,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        data = self._read_all()
        data.append(entry)
        self._write_all(data)
        return rid

    def list(self) -> List[Dict[str, Any]]:
        return sorted(self._read_all(), key=lambda r: r.get("created_at", ""), reverse=True)

    def delete(self, rid: str) -> bool:
        data = self._read_all()
        new_data = [d for d in data if d.get("id") != rid]
        if len(new_data) == len(data):
            return False
        self._write_all(new_data)
        return True


resource_store = ResourceStore()


class AddResourceRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    url: str = Field(..., min_length=5, max_length=500)


@app.post("/api/resources")
async def add_resource_endpoint(req: AddResourceRequest):
    try:
        rid = resource_store.add(req.title, req.url)
        return {"success": True, "id": rid}
    except Exception as e:
        logger.error(f"添加资源失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="无法添加资源。")


@app.get("/api/resources")
async def list_resources_endpoint():
    try:
        resources = resource_store.list()
        return {"success": True, "resources": resources}
    except Exception as e:
        logger.error(f"获取资源失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="无法获取资源列表。")


@app.delete("/api/resources/{rid}")
async def delete_resource_endpoint(rid: str):
    if resource_store.delete(rid):
        return {"success": True}
    raise HTTPException(status_code=404, detail="资源不存在。")


# ---------------- 笔记整理 API ----------------


class OrganizeNotesRequest(BaseModel):
    notes: List[str] = Field(..., min_items=1)


@app.post("/api/organize-notes")
async def organize_notes_endpoint(req: OrganizeNotesRequest):
    try:
        report = await engine.organize_notes(req.notes)
        return {"success": True, "report": report}
    except Exception as e:
        logger.error(f"整理笔记失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="整理笔记失败。")


# 简易导出：返回所有笔记（客户端传）与报告 Markdown
@app.post("/api/notes/export")
async def export_notes_endpoint(req: OrganizeNotesRequest):
    report = await engine.organize_notes(req.notes)
    import io
    md_lines = [f"# {report.get('title', '笔记整理报告')}"]
    sections = report.get("sections", {})
    for k, v in sections.items():
        md_lines.append(f"\n## {k}\n")
        if isinstance(v, list):
            md_lines.extend([f"- {item}" for item in v])
        else:
            md_lines.append(str(v))
    md_content = "\n".join(md_lines)
    file_like = io.BytesIO(md_content.encode("utf-8"))
    headers = {"Content-Disposition": "attachment; filename=notes_report.md"}
    return StreamingResponse(file_like, media_type="text/markdown", headers=headers)


if __name__ == "__main__":
    logger.info("=============================================")
    logger.info("  智能学习计划生成引擎已启动")
    logger.info("  状态: 等待请求")
    logger.info("=============================================")
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)

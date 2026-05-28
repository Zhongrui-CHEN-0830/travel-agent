import json
import os
import re
import logging
import traceback
from typing import Dict, List, Optional, Any
from langchain.tools import tool
from coze_coding_dev_sdk import LLMClient, SearchClient, DocumentGenerationClient
from coze_coding_dev_sdk.s3 import S3SyncStorage
from coze_coding_utils.runtime_ctx.context import new_context
from coze_coding_utils.log.write_log import request_context
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    repair_json = None

logger = logging.getLogger(__name__)

FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


def _get_ctx():
    """获取当前请求上下文，用于链路追踪"""
    ctx = request_context.get()
    if ctx is None:
        ctx = new_context(method="travel_tool")
    return ctx


def _get_text_content(content) -> str:
    """安全提取LLM返回的文本内容"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        if content and isinstance(content[0], str):
            return " ".join(content)
        else:
            return " ".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
    return str(content)


def _extract_json_from_text(text: str) -> str:
    """从文本中提取JSON字符串"""
    # 尝试匹配代码块
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    # 尝试匹配普通代码块
    json_match = re.search(r"```\s*(\{.*?)\s*```", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    # 尝试匹配最外层的大括号
    json_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    return text.strip()


def _fix_common_json_errors(text: str) -> str:
    """修复LLM生成的常见JSON格式错误"""
    # 移除BOM和不可见字符
    text = text.strip().lstrip("\ufeff")
    # 移除尾部逗号: ,] -> ] 和 ,} -> }
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    # 修复单引号为双引号（但保留字符串内的单引号）
    # 简单的做法：先不处理，如果解析失败再尝试
    # 修复缺失的引号键名（简单场景）
    text = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', text)
    return text


def _safe_json_loads(text: str) -> dict:
    """安全解析JSON，尝试修复常见错误，支持多级降级修复"""
    # 级别1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 级别2: 修复常见格式错误
    fixed = _fix_common_json_errors(text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 级别3: 尝试找到最后一个完整的JSON对象（处理尾部多余文本）
    brace_count = 0
    last_valid_end = 0
    for i, ch in enumerate(text):
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                last_valid_end = i + 1
    if last_valid_end > 0:
        truncated = text[:last_valid_end]
        fixed = _fix_common_json_errors(truncated)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # 级别4: 使用 json_repair 库进行深度修复
    if HAS_JSON_REPAIR and repair_json is not None:
        try:
            repaired = repair_json(text)
            if isinstance(repaired, dict):
                return repaired
            if isinstance(repaired, str):
                return json.loads(repaired)
        except Exception:
            pass

    # 级别5: 再次尝试用 json_repair 修复截断后的文本
    if HAS_JSON_REPAIR and repair_json is not None and last_valid_end > 0:
        try:
            repaired = repair_json(text[:last_valid_end])
            if isinstance(repaired, dict):
                return repaired
            if isinstance(repaired, str):
                return json.loads(repaired)
        except Exception:
            pass

    raise json.JSONDecodeError("无法解析JSON，所有修复策略均失败", text, 0)


def _load_font(size: int):
    """加载中文字体，优先使用文泉驿字体"""
    from PIL import ImageFont

    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _calculate_budget_core(itinerary: dict) -> dict:
    """预算计算核心逻辑（普通函数，供多个工具复用）"""
    travelers = itinerary.get("travelers", 1)
    if not isinstance(travelers, int) or travelers < 1:
        travelers = 1

    budget_detail = {
        "门票费用": 0,
        "餐饮费用": 0,
        "住宿费用": 0,
        "交通费用": 0,
        "其他费用": 0,
    }

    days = itinerary.get("days", itinerary.get("itinerary", []))
    for day in days:
        # 兼容多种景点字段名
        spots = day.get("spots", day.get("attractions", day.get("places", [])))
        for spot in spots:
            price = spot.get("price", spot.get("ticket_price", spot.get("ticket", 0))) or 0
            budget_detail["门票费用"] += price * travelers

        # 兼容多种餐饮字段名
        meals = day.get("meals", day.get("food", day.get("dining", [])))
        for meal in meals:
            if isinstance(meal, dict):
                price = meal.get("price", meal.get("total_price", 0)) or 0
                budget_detail["餐饮费用"] += price

        # 兼容多种住宿字段名
        hotel = day.get("hotel", day.get("accommodation", day.get("lodging", {})))
        if hotel and isinstance(hotel, dict):
            budget_detail["住宿费用"] += hotel.get("price", hotel.get("cost", 0)) or 0

        # 兼容多种交通字段名
        transport = day.get("transport", day.get("transportation", {}))
        if transport and isinstance(transport, dict):
            budget_detail["交通费用"] += transport.get("cost", transport.get("price", 0)) or 0

    total = sum(budget_detail.values())
    per_person = total / travelers if travelers > 0 else total

    return {
        "目的地": itinerary.get("destination", ""),
        "出行人数": travelers,
        "预算明细": budget_detail,
        "总费用": total,
        "人均费用": round(per_person, 2),
        "currency": "CNY",
    }


def _upload_image(file_path: str, file_name: str) -> str:
    """上传图片到对象存储并返回签名URL"""
    try:
        storage = S3SyncStorage(
            endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
            access_key="",
            secret_key="",
            bucket_name=os.getenv("COZE_BUCKET_NAME"),
            region="cn-beijing",
        )
        with open(file_path, "rb") as f:
            image_bytes = f.read()
        key = storage.upload_file(
            file_content=image_bytes,
            file_name=file_name,
            content_type="image/png",
        )
        url = storage.generate_presigned_url(key=key, expire_time=86400)
        return url
    except Exception as e:
        logger.error(f"Upload image failed: {e}")
        return ""


def _build_itinerary_markdown(itinerary: dict, include_budget: bool = True) -> str:
    """将行程JSON转换为Markdown格式（普通函数，供导出工具复用）"""
    md = f"""# {itinerary.get('destination', '')} 旅行行程

## 行程概览

- **目的地**：{itinerary.get('destination', '')}
- **出行日期**：{itinerary.get('start_date', '')} 至 {itinerary.get('end_date', '')}
- **出行人数**：{itinerary.get('travelers', 1)}人
- **旅行偏好**：{itinerary.get('preferences', '无')}

## 每日行程
"""
    days = itinerary.get("days", itinerary.get("itinerary", []))
    for day in days:
        md += f"\n### 第{day.get('day', '')}天 ({day.get('date', '')})\n\n"

        spots = day.get("spots", day.get("attractions", day.get("places", [])))
        if spots:
            md += "**景点安排：**\n\n"
            for spot in spots:
                md += f"- **{spot.get('name', '')}** ({spot.get('duration', '')}) - {spot.get('description', '')}\n"
                md += f"  - 门票：{spot.get('price', spot.get('ticket_price', 0))}元 | 地址：{spot.get('address', '')}\n\n"

        meals = day.get("meals", day.get("food", day.get("dining", [])))
        if meals:
            md += "**餐饮推荐：**\n\n"
            for meal in meals:
                if isinstance(meal, dict):
                    md += f"- {meal.get('type', meal.get('meal', ''))}：{meal.get('recommendation', meal.get('name', ''))}（约{meal.get('price', 0)}元/人）\n"
                else:
                    md += f"- {meal}\n"
            md += "\n"

        hotel = day.get("hotel", day.get("accommodation", day.get("lodging")))
        if hotel:
            if isinstance(hotel, dict):
                md += f"**住宿**：{hotel.get('name', '')}（约{hotel.get('price', hotel.get('cost', 0))}元/晚）\n"
                md += f"- 地址：{hotel.get('address', '')}\n\n"
            else:
                md += f"**住宿**：{hotel}\n\n"

        transport = day.get("transport", day.get("transportation"))
        if transport:
            if isinstance(transport, dict):
                md += f"**交通**：{transport.get('description', '')}（约{transport.get('cost', transport.get('price', 0))}元）\n\n"
            else:
                md += f"**交通**：{transport}\n\n"

    if itinerary.get("tips"):
        md += "## 温馨提示\n\n"
        for tip in itinerary.get("tips", []):
            md += f"- {tip}\n"

    if include_budget:
        budget = _calculate_budget_core(itinerary)
        md += f"""\n## 预算参考

| 项目 | 费用（元） |
|------|-----------|
"""
        for item, cost in budget.get("预算明细", {}).items():
            md += f"| {item} | {cost} |\n"
        md += f"| **总计** | **{budget.get('总费用', 0)}** |\n"
        md += f"\n人均费用：约{budget.get('人均费用', 0)}元\n"

    return md


@tool
def search_attractions(
    destination: str, query_type: str = "attractions", count: int = 10
) -> str:
    """搜索目的地的景点、美食或酒店信息，获取实时旅游资讯。

    Args:
        destination: 目的地城市名称，如"北京"、"西安"
        query_type: 搜索类型，可选 attractions(景点)、food(美食)、hotel(酒店)
        count: 返回结果数量，默认10条
    """
    ctx = _get_ctx()
    try:
        client = SearchClient(ctx=ctx)
        query_map = {
            "attractions": f"{destination} 热门景点 旅游攻略 必去",
            "food": f"{destination} 特色美食 推荐餐厅 必吃",
            "hotel": f"{destination} 酒店推荐 住宿攻略 民宿",
        }
        query = query_map.get(query_type, f"{destination} 旅游")
        response = client.web_search(query=query, count=count, need_summary=True)

        results = []
        if response.web_items:
            for item in response.web_items:
                results.append(
                    {
                        "title": item.title,
                        "snippet": item.snippet,
                        "url": item.url,
                        "site_name": item.site_name,
                    }
                )

        summary = response.summary or ""
        return json.dumps(
            {"query": query, "summary": summary, "results": results},
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        logger.error(f"search_attractions error: {e}")
        return json.dumps(
            {"error": str(e), "query": f"{destination} {query_type}"},
            ensure_ascii=False,
        )


@tool
def plan_itinerary(
    destination: str,
    start_date: str,
    end_date: str,
    preferences: str = "",
    travelers: int = 1,
    budget: str = "",
) -> str:
    """使用AI智能规划旅行行程，生成包含景点、餐饮、酒店的完整行程计划。

    Args:
        destination: 目的地城市，如"北京"、"成都"
        start_date: 行程开始日期，格式YYYY-MM-DD
        end_date: 行程结束日期，格式YYYY-MM-DD
        preferences: 旅行偏好，如"历史文化、自然风光、美食体验、亲子游"
        travelers: 出行人数，默认1人
        budget: 预算范围，如"2000-5000元"或"经济型"
    """
    ctx = _get_ctx()
    try:
        # 先搜索目的地信息作为参考
        search_client = SearchClient(ctx=ctx)
        search_response = search_client.web_search(
            query=f"{destination} 热门景点 美食 酒店 旅游攻略",
            count=8,
            need_summary=True,
        )
        search_info = ""
        if search_response.summary:
            search_info = f"\n【网络搜索参考信息】\n{search_response.summary}\n"

        system_prompt = """你是一位资深的旅行规划师，擅长为用户制定详细实用的旅行行程。
请根据用户提供的信息，生成一份完整的行程计划JSON。

【严格要求 - 必须遵守】
1. 输出必须是合法、完整的JSON格式
2. 不要添加任何Markdown代码块标记（如 ```json）
3. 不要添加任何解释说明文字，只输出纯JSON
4. 所有字符串值必须使用双引号包裹
5. 对象和数组末尾不要有多余的逗号
6. 确保大括号和中括号完全匹配闭合

【字段命名强制规范 - 必须使用以下字段名，不可替换】
- 每天的景点列表字段名必须是 "spots"（不是attractions/places）
- 每天的餐饮列表字段名必须是 "meals"（不是food/dining）
- 每天的住宿字段名必须是 "hotel"
- 每天的交通字段名必须是 "transport"
- 每个景点的时长字段名必须是 "duration"（字符串，如"2小时"）
- 每个景点的价格字段名必须是 "price"（数字，单位元）
- 每个景点的纬度字段名必须是 "latitude"（数字）
- 每个景点的经度字段名必须是 "longitude"（数字）

【JSON结构示例】
{
  "destination": "城市名",
  "start_date": "2025-01-01",
  "end_date": "2025-01-03",
  "travelers": 2,
  "total_budget": 3000,
  "preferences": "偏好描述",
  "days": [
    {
      "day": 1,
      "date": "2025-01-01",
      "spots": [
        {"name": "景点名", "type": "attraction", "description": "...", "duration": "2小时", "price": 50, "address": "...", "latitude": 30.123456, "longitude": 120.123456}
      ],
      "meals": [
        {"type": "早餐", "recommendation": "推荐餐厅", "price": 30}
      ],
      "hotel": {"name": "酒店名", "price": 400, "address": "..."},
      "transport": {"description": "交通方式", "cost": 20}
    }
  ],
  "tips": ["提示1", "提示2"]
}"""

        user_prompt = f"""请为以下旅行需求制定详细行程：

目的地：{destination}
出行日期：{start_date} 至 {end_date}
出行人数：{travelers}人
旅行偏好：{preferences or '无特殊偏好'}
预算范围：{budget or '中等预算'}
{search_info}

【重要】请直接输出纯JSON，不要添加任何其他文字或格式标记。确保JSON语法100%正确。"""

        llm_client = LLMClient(ctx=ctx)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # 第一次尝试
        response = llm_client.invoke(
            messages=messages,
            model="doubao-seed-2-0-lite-260215",
            temperature=0.3,
            max_completion_tokens=8192,
        )

        raw_text = _get_text_content(response.content)
        content = _extract_json_from_text(raw_text)

        # 尝试解析
        try:
            itinerary = _safe_json_loads(content)
        except json.JSONDecodeError as e1:
            logger.warning(f"plan_itinerary first parse failed: {e1}, retrying...")
            # 第二次尝试：用更低温度重试，强调JSON格式
            messages.append(
                HumanMessage(
                    content="上一次的输出存在JSON格式错误，请重新输出一份语法完全正确的纯JSON行程计划，不要添加任何解释文字。"
                )
            )
            response2 = llm_client.invoke(
                messages=messages,
                model="doubao-seed-2-0-lite-260215",
                temperature=0.1,
                max_completion_tokens=8192,
            )
            raw_text2 = _get_text_content(response2.content)
            content = _extract_json_from_text(raw_text2)
            try:
                itinerary = _safe_json_loads(content)
            except json.JSONDecodeError as e2:
                logger.error(
                    f"plan_itinerary retry parse failed: {e2}, content: {content[:800]}"
                )
                return json.dumps(
                    {
                        "error": f"行程JSON解析失败，请重新描述需求后重试",
                        "detail": str(e2),
                    },
                    ensure_ascii=False,
                )

        # 确保基本字段存在
        itinerary.setdefault("destination", destination)
        itinerary.setdefault("start_date", start_date)
        itinerary.setdefault("end_date", end_date)
        itinerary.setdefault("travelers", travelers)
        itinerary.setdefault("preferences", preferences)
        itinerary.setdefault("days", [])
        itinerary.setdefault("tips", [])

        return json.dumps(itinerary, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"plan_itinerary error: {e}")
        return json.dumps(
            {"error": str(e), "message": "行程规划失败，请稍后重试"},
            ensure_ascii=False,
        )


@tool
def calculate_budget(itinerary_json: str) -> str:
    """计算并统计行程中的门票、酒店、餐饮、交通等各项费用，生成详细预算明细。

    Args:
        itinerary_json: 行程计划的JSON字符串。你应该从之前 plan_itinerary 的返回结果或对话上下文中直接获取，绝不要求用户手动提供。
    """
    try:
        itinerary = json.loads(itinerary_json)
        result = _calculate_budget_core(itinerary)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as e:
        return json.dumps(
            {"error": f"无效的行程JSON: {str(e)}"}, ensure_ascii=False
        )
    except Exception as e:
        logger.error(f"calculate_budget error: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def edit_itinerary(
    itinerary_json: str,
    operation: str,
    day_index: int = 0,
    spot_info: str = "",
) -> str:
    """编辑行程，支持添加、删除或调整景点顺序。

    Args:
        itinerary_json: 当前行程的JSON字符串。你应该从之前 plan_itinerary 的返回结果或对话上下文中直接获取，绝不要求用户手动提供。
        operation: 操作类型：add(添加景点)、remove(删除景点)、reorder(调整顺序)、update(修改景点)
        day_index: 目标天数索引（从1开始）
        spot_info: 操作相关的景点信息JSON字符串。add时为新景点对象；remove时为{"name":"景点名"}；reorder时为{"order":["景点A","景点B"]}；update时为完整景点对象
    """
    try:
        itinerary = json.loads(itinerary_json)
        # 兼容 days 和 itinerary 两种字段名
        days = itinerary.get("days", itinerary.get("itinerary", []))
        itinerary["days"] = days  # 统一为 days

        if day_index < 1 or day_index > len(days):
            return json.dumps(
                {
                    "error": f"无效的天数索引 {day_index}，行程共 {len(days)} 天"
                },
                ensure_ascii=False,
            )

        target_day = days[day_index - 1]
        spots = target_day.get("spots", [])

        if operation == "add":
            if not spot_info:
                return json.dumps(
                    {"error": "添加操作需要提供spot_info"}, ensure_ascii=False
                )
            new_spot = json.loads(spot_info)
            spots.append(new_spot)
            target_day["spots"] = spots

        elif operation == "remove":
            if not spot_info:
                return json.dumps(
                    {"error": "删除操作需要提供spot_info"}, ensure_ascii=False
                )
            spot_name = json.loads(spot_info).get("name", "")
            original_count = len(spots)
            spots = [s for s in spots if s.get("name") != spot_name]
            if len(spots) == original_count:
                return json.dumps(
                    {"error": f"未找到景点 '{spot_name}'"}, ensure_ascii=False
                )
            target_day["spots"] = spots

        elif operation == "reorder":
            if not spot_info:
                return json.dumps(
                    {"error": "调整顺序操作需要提供spot_info"}, ensure_ascii=False
                )
            order = json.loads(spot_info).get("order", [])
            name_to_spot = {s.get("name"): s for s in spots}
            new_spots = []
            for name in order:
                if name in name_to_spot:
                    new_spots.append(name_to_spot[name])
            if not new_spots:
                return json.dumps(
                    {"error": "调整后的顺序无效"}, ensure_ascii=False
                )
            target_day["spots"] = new_spots

        elif operation == "update":
            if not spot_info:
                return json.dumps(
                    {"error": "修改操作需要提供spot_info"}, ensure_ascii=False
                )
            update_info = json.loads(spot_info)
            spot_name = update_info.get("name", "")
            found = False
            for i, spot in enumerate(spots):
                if spot.get("name") == spot_name:
                    spots[i].update(update_info)
                    found = True
                    break
            if not found:
                return json.dumps(
                    {"error": f"未找到景点 '{spot_name}'"}, ensure_ascii=False
                )

        else:
            return json.dumps(
                {"error": f"不支持的操作类型: {operation}，支持 add/remove/reorder/update"},
                ensure_ascii=False,
            )

        return json.dumps(itinerary, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as e:
        return json.dumps(
            {"error": f"JSON解析失败: {str(e)}"}, ensure_ascii=False
        )
    except Exception as e:
        logger.error(f"edit_itinerary error: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _get_coords(spot: dict):
    """兼容 latitude/longitude 和 coordinates 两种坐标格式"""
    lat = spot.get("latitude")
    lon = spot.get("longitude")
    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon)
        except (ValueError, TypeError):
            pass
    coords = spot.get("coordinates")
    if coords and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        try:
            c0, c1 = float(coords[0]), float(coords[1])
            if 70 <= abs(c0) <= 140 and 15 <= abs(c1) <= 55:
                return c1, c0
            else:
                return c0, c1
        except (ValueError, TypeError):
            pass
    return None, None


def _generate_pillow_map(all_spots, days_to_show, destination, day_index):
    """使用 Pillow 纯绘制地图，不依赖外部网络瓦片。"""
    from PIL import Image, ImageDraw, ImageFont

    WIDTH, HEIGHT = 1200, 900
    PAD = 80

    img = Image.new("RGB", (WIDTH, HEIGHT), "#E8EEF5")
    draw = ImageDraw.Draw(img)

    title_font = _load_font(22)
    legend_font = _load_font(14)
    spot_font = _load_font(12)
    label_font = _load_font(11)

    # 标题栏
    draw.rectangle([0, 0, WIDTH, 50], fill="#2C5F8A")
    title = f"{destination} 旅行地图"
    if day_index > 0:
        title += f" - 第{day_index}天"
    draw.text((20, 12), title, fill="white", font=title_font)

    # 计算经纬度范围
    lats = [s[1] for s in all_spots]
    lons = [s[0] for s in all_spots]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # 增加边距
    lat_margin = max((max_lat - min_lat) * 0.15, 0.005)
    lon_margin = max((max_lon - min_lon) * 0.15, 0.005)
    min_lat -= lat_margin
    max_lat += lat_margin
    min_lon -= lon_margin
    max_lon += lon_margin

    map_top = 60
    map_bottom = HEIGHT - 120
    map_left = PAD
    map_right = WIDTH - PAD

    def _to_px(lon, lat):
        x = map_left + (lon - min_lon) / (max_lon - min_lon) * (map_right - map_left)
        y = map_bottom - (lat - min_lat) / (max_lat - min_lat) * (map_bottom - map_top)
        return int(x), int(y)

    # 绘制地图背景网格（模拟道路/街区）
    draw.rectangle([map_left, map_top, map_right, map_bottom], fill="#F5F8FA", outline="#CCD6E0", width=2)

    # 画网格线
    grid_count = 6
    for i in range(1, grid_count):
        gx = map_left + i * (map_right - map_left) // grid_count
        draw.line([(gx, map_top), (gx, map_bottom)], fill="#E0E6ED", width=1)
    for i in range(1, grid_count):
        gy = map_top + i * (map_bottom - map_top) // grid_count
        draw.line([(map_left, gy), (map_right, gy)], fill="#E0E6ED", width=1)

    # 绘制区域色块（模拟不同区域）
    draw.rectangle([map_left + 50, map_top + 40, map_left + 200, map_top + 180], fill="#D4E6F1", outline="#A9CCE3")
    draw.rectangle([map_right - 250, map_top + 60, map_right - 80, map_top + 200], fill="#D5F5E3", outline="#A9DFBF")
    draw.rectangle([map_left + 300, map_bottom - 200, map_left + 500, map_bottom - 50], fill="#FCF3CF", outline="#F9E79F")
    draw.rectangle([map_right - 400, map_bottom - 180, map_right - 150, map_bottom - 40], fill="#FADBD8", outline="#F5B7B1")

    # 绘制"主要道路"
    road_color = "#FDFEFE"
    road_border = "#D5D8DC"
    # 横向主干道
    draw.rectangle([map_left, map_top + 120, map_right, map_top + 145], fill=road_color, outline=road_border)
    draw.rectangle([map_left, map_top + 320, map_right, map_top + 345], fill=road_color, outline=road_border)
    draw.rectangle([map_left, map_bottom - 200, map_right, map_bottom - 175], fill=road_color, outline=road_border)
    # 纵向主干道
    draw.rectangle([map_left + 180, map_top, map_left + 205, map_bottom], fill=road_color, outline=road_border)
    draw.rectangle([map_left + 500, map_top, map_left + 525, map_bottom], fill=road_color, outline=road_border)
    draw.rectangle([map_right - 220, map_top, map_right - 195, map_bottom], fill=road_color, outline=road_border)

    colors = [
        "#E74C3C",
        "#27AE60",
        "#2980B9",
        "#F39C12",
        "#8E44AD",
        "#16A085",
    ]

    # 按天分组建线和标记
    day_groups = {}
    for lon, lat, name, color in all_spots:
        for d_idx, day in enumerate(days_to_show):
            for spot in day.get("spots", day.get("attractions", day.get("places", []))):
                slat, slon = _get_coords(spot)
                if slat is not None and slon is not None:
                    if abs(slat - lat) < 0.0001 and abs(slon - lon) < 0.0001:
                        if d_idx not in day_groups:
                            day_groups[d_idx] = []
                        day_groups[d_idx].append((lon, lat, name))
                        break

    # 绘制路线（先画线，后画点，避免线覆盖点）
    for d_idx, spots in day_groups.items():
        color = colors[d_idx % len(colors)]
        if len(spots) >= 2:
            pts = [_to_px(s[0], s[1]) for s in spots]
            # 画虚线路径
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
                # 箭头
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                draw.ellipse([mx - 4, my - 4, mx + 4, my + 4], fill=color)

    # 绘制景点标记
    for d_idx, spots in day_groups.items():
        color = colors[d_idx % len(colors)]
        for i, (lon, lat, name) in enumerate(spots):
            x, y = _to_px(lon, lat)
            # 外圈
            draw.ellipse([x - 14, y - 14, x + 14, y + 14], fill="white", outline=color, width=3)
            # 内圈
            draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=color)
            # 编号
            num_text = str(i + 1)
            tw, th = draw.textbbox((0, 0), num_text, font=label_font)[2:4]
            draw.text((x - tw // 2, y - th // 2), num_text, fill="white", font=label_font)
            # 名称标签（带背景）
            label = name[:10]
            tw, th = draw.textbbox((0, 0), label, font=spot_font)[2:4]
            lx, ly = x + 18, y - 8
            draw.rectangle([lx - 2, ly - 2, lx + tw + 4, ly + th + 2], fill="rgba(255,255,255,220)")
            draw.text((lx, ly), label, fill="#2C3E50", font=spot_font)

    # 绘制经纬度刻度
    for i in range(5):
        lon_val = min_lon + i * (max_lon - min_lon) / 4
        x = map_left + i * (map_right - map_left) // 4
        draw.text((x, map_bottom + 5), f"{lon_val:.3f}°", fill="#7F8C8D", font=label_font)
    for i in range(5):
        lat_val = min_lat + i * (max_lat - min_lat) / 4
        y = map_bottom - i * (map_bottom - map_top) // 4
        draw.text((5, y - 5), f"{lat_val:.3f}°", fill="#7F8C8D", font=label_font)

    # 图例
    legend_x = WIDTH - 200
    legend_y = HEIGHT - 110
    draw.rectangle([legend_x - 10, legend_y - 10, legend_x + 180, legend_y + 10 + len(day_groups) * 22], fill="white", outline="#BDC3C7")
    for d_idx in sorted(day_groups.keys()):
        color = colors[d_idx % len(colors)]
        ly = legend_y + d_idx * 22
        draw.ellipse([legend_x, ly, legend_x + 12, ly + 12], fill=color)
        day_label = f"第{d_idx + 1}天"
        draw.text((legend_x + 18, ly - 1), day_label, fill="#2C3E50", font=legend_font)

    # 底部说明
    draw.text((20, HEIGHT - 25), "说明：相同颜色的点和连线表示同一天的游览路线，数字表示游览顺序", fill="#7F8C8D", font=label_font)

    tmp_path = f"/tmp/travel_map_{os.urandom(4).hex()}.png"
    img.save(tmp_path)
    return tmp_path


@tool
def visualize_map(itinerary_json: str, day_index: int = 0) -> str:
    """在地图上标注行程的景点位置，并绘制游览路线，生成地图图片。

    Args:
        itinerary_json: 行程计划的JSON字符串。你应该从之前 plan_itinerary 的返回结果或对话上下文中直接获取，绝不要求用户手动提供。
        day_index: 指定天数（从1开始），0表示全部天数
    """
    try:
        itinerary = json.loads(itinerary_json)

        all_spots = []
        days_to_show = []

        # 兼容 days 和 itinerary 两种字段名
        days = itinerary.get("days", itinerary.get("itinerary", []))
        if day_index == 0:
            days_to_show = days
        elif 1 <= day_index <= len(days):
            days_to_show = [days[day_index - 1]]
        else:
            return json.dumps(
                {"error": f"无效的天数索引 {day_index}，行程共 {len(days)} 天"},
                ensure_ascii=False,
            )

        colors = [
            "#E74C3C",
            "#27AE60",
            "#2980B9",
            "#F39C12",
            "#8E44AD",
            "#16A085",
        ]

        for d_idx, day in enumerate(days_to_show):
            color = colors[d_idx % len(colors)]
            day_spots = day.get("spots", day.get("attractions", day.get("places", [])))
            for spot in day_spots:
                lat, lon = _get_coords(spot)
                if lat is not None and lon is not None:
                    all_spots.append((lon, lat, spot.get("name", ""), color))

        if not all_spots:
            return json.dumps(
                {"error": "行程中没有包含有效经纬度坐标的景点"},
                ensure_ascii=False,
            )

        # 使用 Pillow 纯绘制地图（无网络依赖，不会超时）
        tmp_path = _generate_pillow_map(all_spots, days_to_show, itinerary.get("destination", ""), day_index)

        # 上传到对象存储
        file_name = f"travel_maps/{itinerary.get('destination', 'trip')}_map.png"
        url = _upload_image(tmp_path, file_name)

        if url:
            return json.dumps(
                {
                    "map_image_url": url,
                    "spots_count": len(all_spots),
                    "message": f"地图已生成，包含 {len(all_spots)} 个景点标注和游览路线",
                },
                ensure_ascii=False,
            )
        else:
            return json.dumps(
                {
                    "map_image_path": tmp_path,
                    "spots_count": len(all_spots),
                    "message": f"地图已生成（本地），包含 {len(all_spots)} 个景点标注",
                },
                ensure_ascii=False,
            )

    except Exception as e:
        logger.error(f"visualize_map error: {e}\n{traceback.format_exc()}")
        return json.dumps(
            {"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False
        )


@tool
def export_itinerary(itinerary_json: str, export_format: str = "pdf") -> str:
    """将规划的行程导出为PDF或图片格式，便于保存和分享。

    Args:
        itinerary_json: 行程计划的JSON字符串。你应该从之前 plan_itinerary 的返回结果或对话上下文中直接获取，绝不要求用户手动提供。
        export_format: 导出格式，可选 pdf 或 image
    """
    ctx = _get_ctx()
    try:
        itinerary = json.loads(itinerary_json)

        if export_format.lower() == "pdf":
            md = _build_itinerary_markdown(itinerary, include_budget=True)

            client = DocumentGenerationClient()
            title = f"travel_plan_{itinerary.get('destination', 'trip')}"
            title = re.sub(r"[^a-zA-Z0-9_]", "_", title).lower()
            url = client.create_pdf_from_markdown(md, title)

            return json.dumps(
                {
                    "download_url": url,
                    "format": "pdf",
                    "message": "行程PDF已生成，点击链接下载",
                },
                ensure_ascii=False,
            )

        elif export_format.lower() == "image":
            from PIL import Image, ImageDraw, ImageFont

            width, height = 900, 1400
            img = Image.new("RGB", (width, height), color="#F8F9FA")
            draw = ImageDraw.Draw(img)

            try:
                title_font = ImageFont.truetype(FONT_PATH, 40)
                subtitle_font = ImageFont.truetype(FONT_PATH, 24)
                body_font = ImageFont.truetype(FONT_PATH, 20)
                small_font = ImageFont.truetype(FONT_PATH, 18)
            except Exception:
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
                body_font = ImageFont.load_default()
                small_font = ImageFont.load_default()

            # 顶部标题栏
            draw.rectangle([0, 0, width, 160], fill="#2C5F8A")
            draw.text(
                (40, 40),
                f"{itinerary.get('destination', '')} 旅行行程",
                fill="white",
                font=title_font,
            )
            draw.text(
                (40, 105),
                f"{itinerary.get('start_date', '')} ~ {itinerary.get('end_date', '')} | {itinerary.get('travelers', 1)}人",
                fill="#E0E0E0",
                font=subtitle_font,
            )

            y = 190
            max_y = height - 100

            days = itinerary.get("days", itinerary.get("itinerary", []))
            for day in days:
                if y > max_y:
                    break

                # 日期标题
                draw.rectangle([30, y, width - 30, y + 4], fill="#2C5F8A")
                y += 15
                draw.text(
                    (40, y),
                    f"Day {day.get('day', '')}  {day.get('date', '')}",
                    fill="#1A1A1A",
                    font=subtitle_font,
                )
                y += 45

                # 景点
                for spot in day.get("spots", [])[:4]:
                    if y > max_y:
                        break
                    draw.text(
                        (60, y),
                        f"{spot.get('name', '')}",
                        fill="#333333",
                        font=body_font,
                    )
                    y += 32
                    info = f"  {spot.get('duration', '')}  |  门票约{spot.get('price', 0)}元"
                    draw.text((80, y), info, fill="#666666", font=small_font)
                    y += 28

                # 餐饮
                if day.get("meals") and y < max_y:
                    meals_text = "  ".join(
                        [
                            f"{m.get('type')}: {m.get('recommendation', '')}"
                            for m in day.get("meals", [])
                        ]
                    )
                    draw.text(
                        (60, y),
                        f"餐饮: {meals_text[:55]}",
                        fill="#555555",
                        font=small_font,
                    )
                    y += 32

                # 住宿
                if day.get("hotel") and y < max_y:
                    hotel = day.get("hotel", {})
                    draw.text(
                        (60, y),
                        f"住宿: {hotel.get('name', '')} 约{hotel.get('price', 0)}元/晚",
                        fill="#555555",
                        font=small_font,
                    )
                    y += 32

                y += 20

            # 预算汇总
            budget = _calculate_budget_core(itinerary)
            if y < max_y:
                y += 10
                draw.rectangle([30, y, width - 30, y + 3], fill="#2C5F8A")
                y += 18
                draw.text(
                    (40, y),
                    f"预估总费用: {budget.get('总费用', 0)}元 (人均约{budget.get('人均费用', 0)}元)",
                    fill="#1A1A1A",
                    font=subtitle_font,
                )

            # 底部提示
            draw.text(
                (40, height - 50),
                "由智能旅行助手生成",
                fill="#999999",
                font=small_font,
            )

            tmp_path = f"/tmp/travel_poster_{os.urandom(4).hex()}.png"
            img.save(tmp_path)

            # 上传到对象存储
            file_name = f"travel_exports/{itinerary.get('destination', 'trip')}_poster.png"
            url = _upload_image(tmp_path, file_name)

            if url:
                return json.dumps(
                    {
                        "download_url": url,
                        "format": "image",
                        "message": "行程海报已生成，点击链接查看",
                    },
                    ensure_ascii=False,
                )
            else:
                return json.dumps(
                    {
                        "image_path": tmp_path,
                        "format": "image",
                        "message": "行程海报已生成（本地）",
                    },
                    ensure_ascii=False,
                )
        else:
            return json.dumps(
                {"error": f"不支持的导出格式: {export_format}，支持 pdf 或 image"},
                ensure_ascii=False,
            )

    except json.JSONDecodeError as e:
        return json.dumps(
            {"error": f"无效的行程JSON: {str(e)}"}, ensure_ascii=False
        )
    except Exception as e:
        logger.error(f"export_itinerary error: {e}\n{traceback.format_exc()}")
        return json.dumps(
            {"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False
        )

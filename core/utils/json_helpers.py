import json
from typing import Any, Dict, List, Optional, Union


def ensure_dict(
    value: Union[str, Dict[str, Any], None], default: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    确保一个值是字典类型。

    处理以下情况：
    - None -> 返回默认值或 {}
    - Dict -> 原样返回
    - JSON 字符串 -> 解析并返回字典
    - 其他类型 -> 返回默认值或 {}

    参数:
        value: 需要确保为字典的值
        default: 转换失败时的默认值

    返回:
        一个字典
    """
    if default is None:
        default = {}

    if value is None:
        return default

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            return default
        except (json.JSONDecodeError, TypeError):
            return default

    return default


def ensure_list(
    value: Union[str, List[Any], None], default: Optional[List[Any]] = None
) -> List[Any]:
    """
    确保一个值是列表类型。

    处理以下情况：
    - None -> 返回默认值或 []
    - List -> 原样返回
    - JSON 字符串 -> 解析并返回列表
    - 其他类型 -> 返回默认值或 []

    参数:
        value: 需要确保为列表的值
        default: 转换失败时的默认值

    返回:
        一个列表
    """
    if default is None:
        default = []

    if value is None:
        return default

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            return default
        except (json.JSONDecodeError, TypeError):
            return default

    return default


def safe_json_parse(
    value: Union[str, Dict, List, Any], default: Optional[Any] = None
) -> Any:
    """
    安全地解析可能是 JSON 字符串或已解析对象的值。

    这用于处理过渡期，其中一些数据可能以 JSON 字符串形式存储（旧格式），
    而另一些以正确的对象形式存储（新格式）。

    参数:
        value: 需要解析的值
        default: 解析失败时的默认值

    返回:
        解析后的值或默认值
    """
    if value is None:
        return default

    # 如果已经是字典或列表，直接返回
    if isinstance(value, (dict, list)):
        return value

    # 如果是字符串，尝试解析
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # 如果不是有效的 JSON，返回字符串本身
            return value

    # 对于其他类型，原样返回
    return value


def to_json_string(value: Any) -> str:
    """
    将值转换为 JSON 字符串（如果需要）。

    这用于向后兼容，当需要生成期望 JSON 字符串的数据时使用。

    参数:
        value: 需要转换的值

    返回:
        JSON 字符串表示
    """
    if isinstance(value, str):
        # 如果已经是字符串，检查是否是有效的 JSON
        try:
            json.loads(value)
            return value  # 已经是 JSON 字符串
        except (json.JSONDecodeError, TypeError):
            # 是普通字符串，将其编码为 JSON
            return json.dumps(value)

    # 对于所有其他类型，转换为 JSON
    return json.dumps(value)


def format_for_yield(message_object: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化消息对象以便生成，确保 content 和 metadata 是 JSON 字符串。

    这保持与期望 JSON 字符串的客户端的向后兼容性，
    而数据库现在存储正确的对象。

    参数:
        message_object: 来自数据库的消息对象

    返回:
        content 和 metadata 为 JSON 字符串的消息对象
    """
    if not message_object:
        return message_object

    # 创建副本以避免修改原始对象
    formatted = message_object.copy()

    # 确保 content 是 JSON 字符串
    if "content" in formatted and not isinstance(formatted["content"], str):
        formatted["content"] = json.dumps(formatted["content"])

    # 确保 metadata 是 JSON 字符串
    if "metadata" in formatted and not isinstance(formatted["metadata"], str):
        formatted["metadata"] = json.dumps(formatted["metadata"])

    return formatted

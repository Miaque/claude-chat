import inspect
import json
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union


class SchemaType(Enum):
    """工具定义支持的模式类型"""

    OPENAPI = "openapi"


@dataclass
class ToolSchema:
    """工具模式，包含类型信息

    属性:
        schema_type (SchemaType): 模式类型 (OpenAPI)
        schema (dict[str, Any]): 模式定义
    """

    schema_type: SchemaType
    schema: dict[str, Any]


@dataclass
class ToolResult:
    """工具执行结果

    属性:
        success (bool): 工具执行是否成功
        output (str): 输出消息或错误描述
    """

    success: bool
    output: str


@dataclass
class ToolMetadata:
    """工具级别元数据

    属性:
        display_name (str): 工具名称
        description (str): 工具描述
        icon (Optional[str]): 图标标识
        color (Optional[str]): 颜色
        is_core (bool): 是否为核心工具 (默认False)
        weight (int): 排序顺序 (默认100)
        visible (bool): 是否在前端UI中可见 (默认False)
    """

    display_name: str
    description: str
    icon: Optional[str] = None
    color: Optional[str] = None
    is_core: bool = False
    weight: int = 100
    visible: bool = False


@dataclass
class MethodMetadata:
    """方法级别元数据

    属性:
        display_name (str): 方法名称
        description (str): 方法描述
        is_core (bool): 是否为核心方法 (默认False)
        visible (bool): 是否在前端UI中可见 (默认True)
    """

    display_name: str
    description: str
    is_core: bool = False
    visible: bool = True


class Tool(ABC):
    """抽象基类，所有工具的基类

    提供了实现工具的基类，包括模式注册和结果处理能力。

    属性:
        _schemas (dict[str, list[ToolSchema]]): 注册的工具方法的模式
        _metadata (ToolMetadata | None): 工具级别元数据
        _method_metadata (dict[str, MethodMetadata]): 方法级别元数据

    方法:
        get_schemas: 获取所有注册的工具模式
        get_metadata: 获取工具元数据
        get_method_metadata: 获取所有方法的元数据
        success_response: 返回执行成功的结果
        fail_response: 返回执行失败的结果

    """

    def __init__(self):
        """Initialize tool with empty schema registry."""
        self._schemas: dict[str, list[ToolSchema]] = {}
        self._metadata: ToolMetadata | None = None
        self._method_metadata: dict[str, MethodMetadata] = {}
        # logger.debug(f"Initializing tool class: {self.__class__.__name__}")
        self._register_metadata()
        self._register_schemas()

    def _register_metadata(self):
        """注册类和方法的元数据"""
        # Register tool-level metadata
        if hasattr(self.__class__, "__tool_metadata__"):
            self._metadata = self.__class__.__tool_metadata__

        # Register method-level metadata
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "__method_metadata__"):
                self._method_metadata[name] = method.__method_metadata__

    def _register_schemas(self):
        """注册所有装饰器方法的模式"""
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "tool_schemas"):
                self._schemas[name] = method.tool_schemas
                # logger.debug(f"Registered schemas for method '{name}' in {self.__class__.__name__}")

    def get_schemas(self) -> dict[str, list[ToolSchema]]:
        """获取所有注册的工具模式。

        返回:
            dict[str, list[ToolSchema]]: 方法名称到模式定义的映射
        """
        return self._schemas

    def get_metadata(self) -> Optional[ToolMetadata]:
        """获取工具级别元数据。

        返回:
            ToolMetadata | None: 工具级别元数据或None
        """
        return self._metadata

    def get_method_metadata(self) -> dict[str, MethodMetadata]:
        """获取所有方法的元数据。

        返回:
            dict[str, MethodMetadata]: 方法名称到元数据的映射
        """
        return self._method_metadata

    def success_response(self, data: Union[dict[str, Any], str]) -> ToolResult:
        """返回一个成功的工具执行结果。

        参数:
            data: 结果数据 (字典或字符串)

        返回:
            ToolResult: 包含成功标志和格式化输出的结果
        """
        if isinstance(data, str):
            text = data
        else:
            text = json.dumps(data, indent=2)
        # logger.debug(f"Created success response for {self.__class__.__name__}")
        return ToolResult(success=True, output=text)

    def fail_response(self, msg: str) -> ToolResult:
        """返回一个失败的工具执行结果。

        参数:
            msg: 错误消息描述失败

        返回:
            ToolResult: 包含失败标志和错误消息的结果
        """
        # logger.debug(f"Tool {self.__class__.__name__} returned failed result: {msg}")
        return ToolResult(success=False, output=msg)


def _add_schema(func, schema: ToolSchema):
    """添加模式到函数。"""
    if not hasattr(func, "tool_schemas"):
        func.tool_schemas = []
    func.tool_schemas.append(schema)
    # logger.debug(f"Added {schema.schema_type.value} schema to function {func.__name__}")
    return func


def openapi_schema(schema: dict[str, Any]):
    """OpenAPI模式工具的装饰器。"""

    def decorator(func):
        # logger.debug(f"Applying OpenAPI schema to function {func.__name__}")
        return _add_schema(func, ToolSchema(schema_type=SchemaType.OPENAPI, schema=schema))

    return decorator


def tool_metadata(
    display_name: str,
    description: str,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    is_core: bool = False,
    weight: int = 100,
    visible: bool = False,
):
    """添加元数据到工具类的装饰器。

    参数:
        display_name: 可读的工具名称
        description: 工具描述
        icon: 图标 (可选)
        color: 颜色 (可选)
        is_core: 是否为核心工具 (默认False)
        weight: 排序顺序 (默认100)
        visible: 是否在前端UI中可见 (默认False)

    示例:
        @tool_metadata(
            display_name="文件操作",
            description="创建、读取、编辑和管理文件",
            icon="FolderOpen",
            color="bg-blue-100 dark:bg-blue-800/50",
            weight=20,
            visible=True
        )
        class SandboxFilesTool(Tool):
            ...

        # Example: Hidden from UI (internal tool)
        @tool_metadata(
            display_name="内部功能",
            description="内部功能不显示在前端UI中",
            visible=False
        )
        class InternalTool(Tool):
            ...
    """

    def decorator(cls):
        cls.__tool_metadata__ = ToolMetadata(
            display_name=display_name,
            description=description,
            icon=icon,
            color=color,
            is_core=is_core,
            weight=weight,
            visible=visible,
        )
        return cls

    return decorator


def method_metadata(display_name: str, description: str, is_core: bool = False, visible: bool = True):
    """添加元数据到工具方法的装饰器。

    参数:
        display_name: 方法名称
        description: 方法描述
        is_core: 是否为核心方法 (默认False)
        visible: 是否在前端UI中可见 (默认True)

    示例:
        @method_metadata(
            display_name="创建文件",
            description="创建新的文件",
            visible=True
        )
        @openapi_schema({...})
        def create_file(self, ...):
            ...

        # Example: Hidden from UI (internal method)
        @method_metadata(
            display_name="内部助手",
            description="内部功能不显示在前端UI中",
            visible=False
        )
        @openapi_schema({...})
        def internal_method(self, ...):
            ...
    """

    def decorator(func):
        func.__method_metadata__ = MethodMetadata(
            display_name=display_name,
            description=description,
            is_core=is_core,
            visible=visible,
        )
        return func

    return decorator

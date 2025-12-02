import json
import traceback

from loguru import logger

from core.services.db import get_db
from core.services.llm import make_llm_api_call
from core.utils.icon_generator import RELEVANT_ICONS
from models.project import Project


async def generate_and_update_project_name(project_id: str, prompt: str):
    """
    使用LLM生成项目名称和图标，并更新数据库。

    这通常在项目创建后作为后台任务运行。

    参数:
        project_id: 要更新的项目ID
        prompt: 用于生成名称/图标的初始用户提示
    """
    logger.debug(f"开始为项目生成名称和图标的后台任务: {project_id}")

    try:
        model_name = "glm-4.6"

        # 使用预加载的 Lucide React 图标
        relevant_icons = RELEVANT_ICONS
        system_prompt = f"""You are a helpful assistant that generates extremely concise titles (2-4 words maximum) and selects appropriate icons for chat threads based on the user's message. Always respond in Chinese.

        Available Lucide React icons to choose from:
        {", ".join(relevant_icons)}

        Respond with a JSON object containing:
        - "title": A concise 2-4 word title for the thread
        - "icon": The most appropriate icon name from the list above

        Example response:
        {{"title": "Code Review Help", "icon": "code"}}"""

        user_message = f'Generate an extremely brief title (2-4 words only) and select the most appropriate icon for a chat thread that starts with this message: "{prompt}"'
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        logger.debug(f"开始生成项目 {project_id} 的名称和图标。")
        response = await make_llm_api_call(
            messages=messages,
            model_name=model_name,
            stream=False,
            system_prompt=system_prompt,
            prompt=user_message,
            output_format={
                "type": "json_schema",
                "schema": {
                    "title": {"type": "string"},
                    "icon": {"type": "string", "enum": relevant_icons},
                },
            },
        )

        generated_name = None
        selected_icon = None

        if response and isinstance(response, dict) and response.get("result"):
            raw_content = response["result"].strip()
            try:
                parsed_response = json.loads(raw_content)

                if isinstance(parsed_response, dict):
                    # 提取标题
                    title = parsed_response.get("title", "").strip()
                    if title:
                        generated_name = title.strip("'\" \n\t")
                        logger.debug(f"已生成项目 {project_id} 的名称: '{generated_name}'")

                    # 提取图标
                    icon = parsed_response.get("icon", "").strip()
                    if icon and icon in relevant_icons:
                        selected_icon = icon
                        logger.debug(f"已选择项目 {project_id} 的图标: '{selected_icon}'")
                    else:
                        logger.warning(f"项目 {project_id} 选择了无效的图标 '{icon}'，使用默认的 'message-circle'")
                        selected_icon = "message-circle"
                else:
                    logger.warning(f"项目 {project_id} 返回了非字典类型的JSON: {parsed_response}")

            except json.JSONDecodeError as e:
                logger.warning(f"解析项目 {project_id} 的LLM JSON响应失败: {e}。原始内容: {raw_content}")
                # 从原始内容中提取标题作为回退方案
                cleaned_content = raw_content.strip("'\" \n\t{}")
                if cleaned_content:
                    generated_name = cleaned_content[:50]  # 限制回退标题长度
                selected_icon = "message-circle"  # 默认图标
        else:
            logger.warning(f"为项目 {project_id} 命名时未能从LLM获得有效响应。响应: {response}")

        if generated_name:
            # 将标题和图标存储到专用字段中
            with get_db() as db:
                project = db.query(Project).filter(Project.project_id == project_id).first()
                if project:
                    project.name = generated_name
                    if selected_icon:
                        project.icon_name = selected_icon
                        logger.debug(f"存储项目 {project_id}，标题为: '{generated_name}'，图标为: '{selected_icon}'")
                    else:
                        logger.debug(f"存储项目 {project_id}，标题为: '{generated_name}' (无图标)")
                    db.commit()

            if project:
                logger.debug(f"成功更新项目 {project_id}，包含清晰的标题和专用图标字段")
            else:
                logger.error(f"在数据库中更新项目 {project_id} 失败。更新结果: {project}")
        else:
            logger.warning(f"没有生成名称，跳过项目 {project_id} 的数据库更新。")

    except Exception as e:
        logger.error(f"项目 {project_id} 的后台命名任务出错: {str(e)}\n{traceback.format_exc()}")
    finally:
        logger.debug(f"完成项目的后台命名和图标选择任务: {project_id}")

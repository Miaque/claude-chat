import asyncio
import json
from email import message
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Type, Union, cast

from loguru import logger

from core.error_processor import ErrorProcessor
from core.response_processor import ProcessorConfig, ResponseProcessor
from core.services.llm import LLMError

ToolChoice = Literal["auto", "required", "none"]


class ThreadManager:
    """Manages conversation threads with LLM models and tool execution."""

    def __init__(
        self,
        agent_config: Optional[dict] = None,
    ):
        self.db = DBConnection()
        self.tool_registry = ToolRegistry()

        self.agent_config = agent_config
        self.response_processor = ResponseProcessor(
            tool_registry=self.tool_registry,
            add_message_callback=self.add_message,
            agent_config=self.agent_config,
        )

    def add_tool(
        self,
        tool_class: Type[Tool],
        function_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """Add a tool to the ThreadManager."""
        self.tool_registry.register_tool(tool_class, function_names, **kwargs)

    async def create_thread(
        self,
        account_id: Optional[str] = None,
        project_id: Optional[str] = None,
        is_public: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new thread in the database."""
        # logger.debug(f"Creating new thread (account_id: {account_id}, project_id: {project_id})")
        client = await self.db.client

        thread_data = {"is_public": is_public, "metadata": metadata or {}}
        if account_id:
            thread_data["account_id"] = account_id
        if project_id:
            thread_data["project_id"] = project_id

        try:
            result = await client.table("threads").insert(thread_data).execute()
            if result.data and len(result.data) > 0 and "thread_id" in result.data[0]:
                thread_id = result.data[0]["thread_id"]
                logger.info(f"Successfully created thread: {thread_id}")
                return thread_id
            else:
                raise Exception("Failed to create thread: no thread_id returned")
        except Exception as e:
            logger.error(f"Failed to create thread: {str(e)}", exc_info=True)
            raise Exception(f"Thread creation failed: {str(e)}")

    async def add_message(
        self,
        thread_id: str,
        type: str,
        content: Union[Dict[str, Any], List[Any], str],
        is_llm_message: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        agent_version_id: Optional[str] = None,
    ):
        """Add a message to the thread in the database."""
        # logger.debug(f"Adding message of type '{type}' to thread {thread_id}")
        client = await self.db.client

        data_to_insert = {
            "thread_id": thread_id,
            "type": type,
            "content": content,
            "is_llm_message": is_llm_message,
            "metadata": metadata or {},
        }

        if agent_id:
            data_to_insert["agent_id"] = agent_id
        if agent_version_id:
            data_to_insert["agent_version_id"] = agent_version_id

        try:
            result = await client.table("messages").insert(data_to_insert).execute()

            if result.data and len(result.data) > 0 and "message_id" in result.data[0]:
                saved_message = result.data[0]

                return saved_message
            else:
                logger.error(f"Insert operation failed for thread {thread_id}")
                return None
        except Exception as e:
            logger.error(
                f"Failed to add message to thread {thread_id}: {str(e)}", exc_info=True
            )
            raise

    async def get_llm_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a thread."""
        logger.debug(f"Getting messages for thread {thread_id}")
        client = await self.db.client

        try:
            all_messages = []
            batch_size = 1000
            offset = 0

            while True:
                result = (
                    await client.table("messages")
                    .select("message_id, type, content, metadata")
                    .eq("thread_id", thread_id)
                    .eq("is_llm_message", True)
                    .order("created_at")
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )

                if not result.data:
                    break

                all_messages.extend(result.data)
                if len(result.data) < batch_size:
                    break
                offset += batch_size

            if not all_messages:
                return []

            messages = []
            for item in all_messages:
                # Check if this message has a compressed version in metadata
                content = item["content"]
                metadata = item.get("metadata", {})
                is_compressed = False

                # If compressed, use compressed_content for LLM instead of full content
                if isinstance(metadata, dict) and metadata.get("compressed"):
                    compressed_content = metadata.get("compressed_content")
                    if compressed_content:
                        content = compressed_content
                        is_compressed = True
                        # logger.debug(f"Using compressed content for message {item['message_id']}")

                # Parse content and add message_id
                if isinstance(content, str):
                    try:
                        parsed_item = json.loads(content)
                        parsed_item["message_id"] = item["message_id"]
                        messages.append(parsed_item)
                    except json.JSONDecodeError:
                        # If compressed, content is a plain string (not JSON) - this is expected
                        if is_compressed:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": content,
                                    "message_id": item["message_id"],
                                }
                            )
                        else:
                            logger.error(f"Failed to parse message: {content[:100]}")
                else:
                    content["message_id"] = item["message_id"]
                    messages.append(content)

            return messages

        except Exception as e:
            logger.error(
                f"Failed to get messages for thread {thread_id}: {str(e)}",
                exc_info=True,
            )
            return []

    async def run_thread(
        self,
        thread_id: str,
        system_prompt: Dict[str, Any],
        stream: bool = True,
        temporary_message: Optional[Dict[str, Any]] = None,
        llm_model: str = "gpt-5",
        llm_temperature: float = 0,
        llm_max_tokens: Optional[int] = None,
        processor_config: Optional[ProcessorConfig] = None,
        tool_choice: ToolChoice = "auto",
        native_max_auto_continues: int = 25,
        max_xml_tool_calls: int = 0,
        latest_user_message_content: Optional[str] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator]:
        """Run a conversation thread with LLM integration and tool execution."""
        logger.debug(
            f"🚀 Starting thread execution for {thread_id} with model {llm_model}"
        )

        # Ensure we have a valid ProcessorConfig object
        if processor_config is None:
            config = ProcessorConfig()
        elif isinstance(processor_config, ProcessorConfig):
            config = processor_config
        else:
            logger.error(
                f"Invalid processor_config type: {type(processor_config)}, creating default"
            )
            config = ProcessorConfig()

        if max_xml_tool_calls > 0 and not config.max_xml_tool_calls:
            config.max_xml_tool_calls = max_xml_tool_calls

        auto_continue_state = {
            "count": 0,
            "active": True,
            "continuous_state": {"accumulated_content": "", "thread_run_id": None},
        }

        # Single execution if auto-continue is disabled
        if native_max_auto_continues == 0:
            result = await self._execute_run(
                thread_id,
                system_prompt,
                llm_model,
                llm_temperature,
                llm_max_tokens,
                tool_choice,
                config,
                stream,
                auto_continue_state,
                temporary_message,
                latest_user_message_content,
                cancellation_event,
            )

            # If result is an error dict, convert it to a generator that yields the error
            if isinstance(result, dict) and result.get("status") == "error":
                return self._create_single_error_generator(result)

            return result

        # Auto-continue execution
        return self._auto_continue_generator(
            thread_id,
            system_prompt,
            llm_model,
            llm_temperature,
            llm_max_tokens,
            tool_choice,
            config,
            stream,
            auto_continue_state,
            temporary_message,
            native_max_auto_continues,
            latest_user_message_content,
            cancellation_event,
        )

    async def _execute_run(
        self,
        thread_id: str,
        system_prompt: Dict[str, Any],
        llm_model: str,
        llm_temperature: float,
        llm_max_tokens: Optional[int],
        tool_choice: ToolChoice,
        config: ProcessorConfig,
        stream: bool,
        auto_continue_state: Dict[str, Any],
        temporary_message: Optional[Dict[str, Any]] = None,
        latest_user_message_content: Optional[str] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator]:
        """Execute a single LLM run."""

        # CRITICAL: Ensure config is always a ProcessorConfig object
        if not isinstance(config, ProcessorConfig):
            logger.error(
                f"ERROR: config is {type(config)}, expected ProcessorConfig. Value: {config}"
            )
            config = ProcessorConfig()  # Create new instance as fallback

        try:
            estimated_total_tokens = (
                None  # Will be passed to response processor to avoid recalculation
            )

            # CRITICAL: Check if this is an auto-continue iteration FIRST (before any token counting)
            is_auto_continue = auto_continue_state.get("count", 0) > 0

            # Always fetch messages (needed for LLM call)
            # Fast path just skips compression, not fetching!
            messages = await self.get_llm_messages(thread_id)

            # Handle auto-continue context
            if auto_continue_state["count"] > 0 and auto_continue_state[
                "continuous_state"
            ].get("accumulated_content"):
                partial_content = auto_continue_state["continuous_state"][
                    "accumulated_content"
                ]
                messages.append({"role": "assistant", "content": partial_content})

            # Get tool schemas for LLM API call (after compression)
            openapi_tool_schemas = (
                self.tool_registry.get_openapi_schemas()
                if config.native_tool_calling
                else None
            )

            prepared_messages = messages

            # Note: We don't log token count here because cached blocks give inaccurate counts
            # The LLM's usage.prompt_tokens (reported after the call) is the accurate source of truth
            logger.info(f"📤 Sending {len(prepared_messages)} prepared messages to LLM")

            # Make LLM call
            try:
                llm_response = await make_llm_api_call(
                    prepared_messages,
                    llm_model,
                    temperature=llm_temperature,
                    max_tokens=llm_max_tokens,
                    tools=openapi_tool_schemas,
                    tool_choice=tool_choice if config.native_tool_calling else "none",
                    stream=stream,
                )
            except LLMError as e:
                return {"type": "status", "status": "error", "message": str(e)}

            # Check for error response
            if isinstance(llm_response, dict) and llm_response.get("status") == "error":
                return llm_response

            if stream and hasattr(llm_response, "__aiter__"):
                return self.response_processor.process_streaming_response(
                    cast(AsyncGenerator, llm_response),
                    thread_id,
                    prepared_messages,
                    llm_model,
                    config,
                    True,
                    auto_continue_state["count"],
                    auto_continue_state["continuous_state"],
                    estimated_total_tokens,
                    cancellation_event,
                )
            else:
                return self.response_processor.process_non_streaming_response(
                    llm_response,
                    thread_id,
                    prepared_messages,
                    llm_model,
                    config,
                    estimated_total_tokens,
                )

        except Exception as e:
            processed_error = ErrorProcessor.process_system_error(
                e, context={"thread_id": thread_id}
            )
            ErrorProcessor.log_error(processed_error)
            return processed_error.to_stream_dict()

    async def _auto_continue_generator(
        self,
        thread_id: str,
        system_prompt: Dict[str, Any],
        llm_model: str,
        llm_temperature: float,
        llm_max_tokens: Optional[int],
        tool_choice: ToolChoice,
        config: ProcessorConfig,
        stream: bool,
        auto_continue_state: Dict[str, Any],
        temporary_message: Optional[Dict[str, Any]],
        native_max_auto_continues: int,
        latest_user_message_content: Optional[str] = None,
        cancellation_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator:
        """Generator that handles auto-continue logic."""
        logger.debug(
            f"Starting auto-continue generator, max: {native_max_auto_continues}"
        )
        # logger.debug(f"Config type in auto-continue generator: {type(config)}")

        # Ensure config is valid ProcessorConfig
        if not isinstance(config, ProcessorConfig):
            logger.error(
                f"Invalid config type in auto-continue: {type(config)}, creating new one"
            )
            config = ProcessorConfig()

        while (
            auto_continue_state["active"]
            and auto_continue_state["count"] < native_max_auto_continues
        ):
            auto_continue_state["active"] = False  # Reset for this iteration

            try:
                # Check for cancellation before continuing
                if cancellation_event and cancellation_event.is_set():
                    logger.info(
                        f"Cancellation signal received in auto-continue generator for thread {thread_id}"
                    )
                    break

                response_gen = await self._execute_run(
                    thread_id,
                    system_prompt,
                    llm_model,
                    llm_temperature,
                    llm_max_tokens,
                    tool_choice,
                    config,
                    stream,
                    auto_continue_state,
                    temporary_message if auto_continue_state["count"] == 0 else None,
                    latest_user_message_content
                    if auto_continue_state["count"] == 0
                    else None,
                    cancellation_event,
                )

                # Handle error responses
                if (
                    isinstance(response_gen, dict)
                    and response_gen.get("status") == "error"
                ):
                    yield response_gen
                    break

                # Process streaming response
                if hasattr(response_gen, "__aiter__"):
                    async for chunk in cast(AsyncGenerator, response_gen):
                        # Check for cancellation
                        if cancellation_event and cancellation_event.is_set():
                            logger.info(
                                f"Cancellation signal received while processing stream in auto-continue for thread {thread_id}"
                            )
                            break

                        # Check for auto-continue triggers
                        should_continue = self._check_auto_continue_trigger(
                            chunk, auto_continue_state, native_max_auto_continues
                        )

                        # Skip finish chunks that trigger auto-continue (but NOT tool execution, FE needs those)
                        if should_continue:
                            if chunk.get("type") == "status":
                                try:
                                    content = json.loads(chunk.get("content", "{}"))
                                    # Only skip length limit finish statuses (frontend needs tool execution finish)
                                    if content.get("finish_reason") == "length":
                                        continue
                                except (json.JSONDecodeError, TypeError):
                                    pass

                        yield chunk
                else:
                    yield response_gen

                if not auto_continue_state["active"]:
                    break

            except Exception as e:
                processed_error = ErrorProcessor.process_system_error(
                    e, context={"thread_id": thread_id}
                )
                ErrorProcessor.log_error(processed_error)
                yield processed_error.to_stream_dict()
                return

        # Handle max iterations reached
        if (
            auto_continue_state["active"]
            and auto_continue_state["count"] >= native_max_auto_continues
        ):
            logger.warning(
                f"Reached maximum auto-continue limit ({native_max_auto_continues})"
            )
            yield {
                "type": "content",
                "content": f"\n[Agent reached maximum auto-continue limit of {native_max_auto_continues}]",
            }

    def _check_auto_continue_trigger(
        self,
        chunk: Dict[str, Any],
        auto_continue_state: Dict[str, Any],
        native_max_auto_continues: int,
    ) -> bool:
        """Check if a response chunk should trigger auto-continue."""
        if chunk.get("type") == "status":
            try:
                content = (
                    json.loads(chunk.get("content", "{}"))
                    if isinstance(chunk.get("content"), str)
                    else chunk.get("content", {})
                )
                finish_reason = content.get("finish_reason")
                tools_executed = content.get("tools_executed", False)

                # Trigger auto-continue for: native tool calls, length limit, or XML tools executed
                if finish_reason == "tool_calls" or tools_executed:
                    if native_max_auto_continues > 0:
                        logger.debug(
                            f"Auto-continuing for tool execution ({auto_continue_state['count'] + 1}/{native_max_auto_continues})"
                        )
                        auto_continue_state["active"] = True
                        auto_continue_state["count"] += 1
                        return True
                elif finish_reason == "length":
                    logger.debug(
                        f"Auto-continuing for length limit ({auto_continue_state['count'] + 1}/{native_max_auto_continues})"
                    )
                    auto_continue_state["active"] = True
                    auto_continue_state["count"] += 1
                    return True
                elif finish_reason == "xml_tool_limit_reached":
                    logger.debug("Stopping auto-continue due to XML tool limit")
                    auto_continue_state["active"] = False
            except (json.JSONDecodeError, TypeError):
                pass

        return False

    async def _create_single_error_generator(self, error_dict: Dict[str, Any]):
        """Create an async generator that yields a single error message."""
        yield error_dict

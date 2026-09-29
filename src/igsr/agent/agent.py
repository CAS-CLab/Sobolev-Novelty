from abc import ABC, abstractmethod
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Callable, Dict, List, Optional, Tuple

import litellm
import structlog

from igsr.agent.completion_cache import (
    CompletionCache,
    CompletionCacheCorruptionError,
    deterministic_request_seed,
)
from igsr.agent.retry import RetryConfig, RetryHandler
from igsr.const import NAME
from igsr.logging_setup import get_logger


@dataclass
class PostprocessorOutput:
    success: bool
    inform_agent: bool
    feedback: Optional[str] = None
    parsed_output: Optional[Any] = None


@dataclass
class AgentResponse:
    """Response from an agent's execution.

    Args:
        success (bool): Whether the overall task was completed successfully.
        message (str): Final message or explanation from the agent.
        error (Optional[str]): Error message if any.
    """

    success: bool
    message: str
    postprocessor_output: Optional[PostprocessorOutput] = None
    error: Optional[str] = None


class Agent(ABC):
    """Abstract base class for AI agents."""

    def __init__(
        self,
        task_description: str,
        logger: Optional[structlog.BoundLogger] = None,
    ):
        """Initialize an agent.

        Args:
            task_description (str): Description of the task to perform.
            logger (Optional[structlog.BoundLogger]): Logger instance. If not provided, a default logger will be created.
        """
        if logger is None:
            logger = get_logger(f"{NAME}.agent.base")
        self.logger: structlog.stdlib.BoundLogger = logger  # type: ignore
        self.task_description = task_description


CENSOR_KEYS_FOR_LOGGING = ["api_key", "api_secret", "api_token", "password", "secret", "token", "key"]


def censor_sensitive_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Censor sensitive keys in a dictionary before logging.

    Args:
        data (Dict[str, Any]): Data to log.

    Returns:
        Dict[str, Any]: Data with sensitive keys censored.
    """
    censored_data = data.copy()
    for key in CENSOR_KEYS_FOR_LOGGING:
        if key in censored_data:
            censored_data[key] = "*****"
    return censored_data


TASK_COMPLETE_SENTINEL: str = "TASK COMPLETE"
TASK_FAILED_SENTINEL: str = "TASK FAILED"


class BaseOpenAIAgent(Agent):
    """Base class for OpenAI-format agents with common functionality."""

    def __init__(
        self,
        task_description: str,
        system_prompt: Optional[str] = None,
        logger: Optional[structlog.BoundLogger] = None,
        max_steps: Optional[int] = None,
        retry_config: Optional[RetryConfig] = None,
        completion_kwargs: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        task_complete_sentinels: Optional[List[str]] = (TASK_COMPLETE_SENTINEL,),
        task_failed_sentinels: Optional[List[str]] = (TASK_FAILED_SENTINEL,),
        postprocessor: Optional[Callable[..., PostprocessorOutput]] = None,
        postprocessor_args: Optional[Tuple[Any, ...]] = None,
        postprocessor_kwargs: Optional[Dict[str, Any]] = None,
        reminder: Optional[str] = None,
    ):
        """Initialize base OpenAI agent.

        Note:
            Optional `postprocessor` callable is to postprocess the agent's output. Namely, it should do two things:
                - Validate the agent's output.
                - (Optionally) Return a parsed output from the agent's output.
            Positional and keyword arguments that will be passed to the `postprocessor` every time it is invoked.
            The callable must have the signature:
                ``(agent: BaseOpenAIAgent, *args, **kwargs) -> PostprocessorOutput``
            Returning:
                - ``success`` (bool): True if the task was actually completed successfully
                - ``inform_agent`` (bool): Whether the postprocessor's feedback should be sent back to the LLM as the
                    next user message so it can self-correct and continue.
                - ``feedback`` (str): Human-readable feedback from the postprocessor.
                - ``parsed_output`` (Any): Parsed output from the postprocessor.
            If `postprocessor` is not supplied, the agent's own determination of task success is used.

        Args:
            task_description (str): Task description.
            system_prompt (Optional[str], optional): System prompt that will be the first part of the first
                system message. If not provided, a default system prompt will be used.
            logger (Optional[structlog.BoundLogger]): Logger instance. Defaults to None. If not provided,
                a default logger will be created.
            max_steps (Optional[int], optional): Maximum steps before stopping. Defaults to None (no limit).
            retry_config (Optional[RetryConfig], optional): Configuration for retry behavior. Defaults to None.
                If not provided, a default configuration will be used.
            completion_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass to the
                completion API. Defaults to None.
            conversation_history (Optional[List[Dict[str, Any]]], optional): List of conversation history messages to
                continue from. Defaults to None.
            task_complete_sentinels (Optional[List[str]], optional): List of task complete sentinels.
                Defaults to ("TASK COMPLETE",).
            task_failed_sentinels (Optional[List[str]], optional): List of task failed sentinels.
                Defaults to ("TASK FAILED",).
            postprocessor (Optional[Callable[..., PostprocessorOutput]], optional): Optional callable to postprocess
                the agent's output.
            postprocessor_args (Optional[Tuple[Any, ...]], optional): Positional arguments to pass to the postprocessor.
            postprocessor_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments to pass to the postprocessor.
        """
        if logger is None:
            logger = get_logger(f"{NAME}.agent.openai_base")
        super().__init__(task_description, logger)

        self.max_steps = max_steps
        self.completion_kwargs = completion_kwargs or dict()

        self.system_prompt = system_prompt

        self.task_complete_sentinels = task_complete_sentinels
        self.task_failed_sentinels = task_failed_sentinels

        self.reminder = reminder

        self.conversation_history = []
        self.usage_history = []
        if conversation_history is not None:
            for message in conversation_history:
                self._add_to_history(message["role"], message["content"])
        self.num_llm_calls = 0

        self.n_steps_taken = 0
        self._system_prompt_logged = None

        if retry_config is not None:
            self.retry_handler = RetryHandler(retry_config)
        else:
            self.retry_handler = RetryHandler(RetryConfig())

        # Store validator related configuration before any potential use.
        self.postprocessor = postprocessor
        self._postprocessor_args: Tuple[Any, ...] = postprocessor_args or tuple()
        self._postprocessor_kwargs: Dict[str, Any] = postprocessor_kwargs or dict()

    def _create_system_prompt(self) -> str:
        """Create the system prompt."""
        SYSTEM_PROMPT_PLACEHOLDER = "%%SYSTEM_PROMPT%%"
        TASK_DESCRIPTION_PLACEHOLDER = "%%TASK_DESCRIPTION%%"

        task_complete_sentinels_str_repl = "%TASK_COMPLETE_SENTINELS%"
        task_complete_sentinels_str = "\n".join([f"{s}" for s in self.task_complete_sentinels])
        task_failed_sentinels_str_repl = "%TASK_FAILED_SENTINELS%"
        task_failed_sentinels_str = "\n".join([f"{s}" for s in self.task_failed_sentinels])
        default_system_prompt = dedent(f"""
            # Your role
            You are a helpful AI agent.

            # Reasoning
            For each step:
            1. Think about what to do next and output your message.
            2. Reflect on the result and proceed to the next step.

            # Task completion
            To finish the task, you MUST respond with the following format:
            
            * If the task is complete, include any of the following sentinel(s) in your response:
            {task_complete_sentinels_str_repl}
            
            * If the task cannot be completed, include any of the following sentinel(s) in your response:
            {task_failed_sentinels_str_repl}
            
            # Your task
            ==============================
            """)
        default_system_prompt = default_system_prompt.replace(
            task_complete_sentinels_str_repl, task_complete_sentinels_str
        )
        default_system_prompt = default_system_prompt.replace(task_failed_sentinels_str_repl, task_failed_sentinels_str)
        system_prompt = dedent(
            f"""
            {SYSTEM_PROMPT_PLACEHOLDER}
            {TASK_DESCRIPTION_PLACEHOLDER}
            """
        )
        system_prompt = system_prompt.replace(
            SYSTEM_PROMPT_PLACEHOLDER,
            dedent(default_system_prompt if (self.system_prompt is None) else self.system_prompt),
        )
        system_prompt = system_prompt.replace(TASK_DESCRIPTION_PLACEHOLDER, dedent(self.task_description))

        # Log the system prompt if it has changed.
        if self._system_prompt_logged != system_prompt:
            if self._system_prompt_logged is not None:
                self.logger.debug("system_prompt_changed", system_prompt=system_prompt)
            else:
                self.logger.debug("system_prompt_created", system_prompt=system_prompt)
            self._system_prompt_logged = system_prompt
        return system_prompt

    def _add_to_history(self, role: str, content: str, usage: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation history."""
        self.logger.debug("adding_to_history", role=role, content_length=len(content), content=content)
        self.conversation_history.append({"role": role, "content": content})
        self.usage_history.append(usage or dict())

    @abstractmethod
    def _create_chat_completion(self, messages: List[Dict[str, str]], completion_kwargs: Dict[str, Any]) -> Any:
        """Create a chat completion using the appropriate client.

        This method should be implemented by specific OpenAI agent classes
        to handle their unique API requirements.

        Args:
            messages (List[Dict[str, str]]): List of messages in the conversation.
            completion_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the completion API.
        """
        ...

    def _task_complete(self, message: str) -> List[str]:
        """Check if the task is complete."""
        return [sentinel for sentinel in self.task_complete_sentinels if sentinel in message]

    def _task_failed(self, message: str) -> List[str]:
        """Check if the task is failed."""
        return [sentinel for sentinel in self.task_failed_sentinels if sentinel in message]

    def _execute_step(self) -> Optional[AgentResponse]:
        """Execute a single step."""
        self.logger.info("executing_step", step_count=self.n_steps_taken)

        try:
            # Create chat completion using the appropriate client
            self.logger.debug(
                "making_api_call",
                history_length=len(self.conversation_history),
                completion_kwargs=censor_sensitive_keys(self.completion_kwargs),
            )

            messages = [{"role": "system", "content": self._create_system_prompt()}] + self.conversation_history
            response = self._create_chat_completion(messages, self.completion_kwargs)
            assistant_message = response.choices[0].message.content
            if not assistant_message:
                raise ValueError("Empty message content received from API")
            self._add_to_history("assistant", assistant_message, usage=response.usage.to_dict())
            self.num_llm_calls += 1
            self.n_steps_taken += 1

            # Check task status
            if self._task_complete(assistant_message):
                for sentinel in self.task_complete_sentinels:
                    assistant_message = assistant_message.replace(sentinel, "").strip()
                completed_message = assistant_message
                self.logger.info("task_completed", message=completed_message)

                # Optionally postprocess the task if a postprocessor has been provided.
                if self.postprocessor is not None:
                    try:
                        postprocessor_output = self.postprocessor(
                            self, *self._postprocessor_args, **self._postprocessor_kwargs
                        )
                    except Exception as e:  # pylint: disable=W0718
                        # If the postprocessor itself raises an exception we raise it.
                        error_msg = f"Postprocessor execution error: {str(e)}"
                        self.logger.error(
                            "task_postprocessing_error", error_type=type(e).__name__, error=error_msg, exc_info=True
                        )
                        raise

                    # If the postprocessor confirms the task was completed successfully:
                    if postprocessor_output.success:
                        # We still may want to attach feedback for the consumer even if successful.
                        self.logger.info("task_success_postprocessing", feedback=postprocessor_output.feedback)
                        return AgentResponse(
                            success=True, message=completed_message, postprocessor_output=postprocessor_output
                        )

                    # If the postprocessor determines the task failed:
                    if postprocessor_output.inform_agent:
                        # Feed the feedback back into the conversation so the LLM can self-correct.
                        self.logger.info("postprocessor_feedback_to_agent", feedback=postprocessor_output.feedback)
                        self._add_to_history("user", postprocessor_output.feedback)
                        return None

                    # Do not inform the agent - fail immediately.
                    self.logger.info("task_failed_postprocessing", feedback=postprocessor_output.feedback)
                    return AgentResponse(
                        success=False, message=completed_message, postprocessor_output=postprocessor_output
                    )

                # No postprocessor - assume the task was completed successfully.
                return AgentResponse(success=True, message=completed_message)

            if self._task_failed(assistant_message):
                for sentinel in self.task_failed_sentinels:
                    assistant_message = assistant_message.replace(sentinel, "").strip()
                failed_message = assistant_message
                self.logger.info("task_failed", message=failed_message)
                return AgentResponse(success=False, message=failed_message)

            if self.reminder:
                self._add_to_history("user", self.reminder)

            return None  # Continue the run loop.

        except Exception as e:
            self.logger.error("step_execution_error", error_type=type(e).__name__, error=str(e), exc_info=True)
            raise

    def run(self) -> AgentResponse:
        """Run the agent on its assigned task."""
        self.logger.info("starting_agent_run", task_description=self.task_description)

        self.n_steps_taken = 0
        while (self.n_steps_taken < self.max_steps) if (self.max_steps is not None) else True:
            result = self._execute_step()
            if result is not None:
                return result

        self.logger.warn("max_steps_exceeded", steps_taken=self.n_steps_taken)
        return AgentResponse(
            success=False,
            message="Maximum steps exceeded",
            error="Agent did not complete task within maximum allowed steps",
        )


class LiteLLMAgent(BaseOpenAIAgent):
    """Agent implementation using LiteLLM's API."""

    def __init__(
        self,
        task_description: str,
        system_prompt: Optional[str] = None,
        logger: Optional[structlog.BoundLogger] = None,
        model: str = "openai/gpt-4",
        max_steps: Optional[int] = None,
        retry_config: Optional[RetryConfig] = None,
        completion_kwargs: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        task_complete_sentinels: Optional[List[str]] = (TASK_COMPLETE_SENTINEL,),
        task_failed_sentinels: Optional[List[str]] = (TASK_FAILED_SENTINEL,),
        postprocessor: Optional[Callable[..., PostprocessorOutput]] = None,
        postprocessor_args: Optional[Tuple[Any, ...]] = None,
        postprocessor_kwargs: Optional[Dict[str, Any]] = None,
        reminder: Optional[str] = None,
        cache_context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize an LiteLLM agent.

        Same parameters as BaseOpenAIAgent.
        """
        if logger is None:
            logger = get_logger(f"{NAME}.agent.litellm")
            logger.info(
                "initializing_litellm_agent", model=model, max_steps=max_steps, completion_kwargs=completion_kwargs
            )
        super().__init__(
            task_description,
            system_prompt,
            logger,
            max_steps,
            retry_config,
            completion_kwargs,
            conversation_history,
            task_complete_sentinels,
            task_failed_sentinels,
            postprocessor,
            postprocessor_args,
            postprocessor_kwargs,
            reminder,
        )

        self.model = model
        self.completion_cache = CompletionCache.from_environment()
        self.cache_context = dict(cache_context or {})

    def _create_chat_completion(self, messages: List[Dict[str, str]], completion_kwargs: Dict[str, Any]) -> Any:
        """Create a chat completion using LiteLLM's API.

        Args:
            messages (List[Dict[str, str]]): List of messages in the conversation.
            completion_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the completion API.
        """

        request_kwargs = dict(completion_kwargs)
        request_cache_context = dict(self.cache_context)
        if request_cache_context.get("protocol") == "igsr-paired-completion-v2":
            request_cache_context["agent_step"] = self.n_steps_taken
            if "seed" in request_kwargs:
                request_kwargs["seed"] = deterministic_request_seed(request_cache_context)
        self.logger.debug(
            "resolved_completion_request",
            agent_step=request_cache_context.get("agent_step"),
            request_seed=request_kwargs.get("seed"),
        )

        @self.retry_handler.with_retries()
        def _make_request() -> litellm.ModelResponse:  # pyright: ignore
            metadata = dict()
            return litellm.completion(  # pyright: ignore
                model=self.model,
                messages=messages,
                metadata=metadata,
                **request_kwargs,
            )

        def _restore_response(response_data: Dict[str, Any]) -> litellm.ModelResponse:  # pyright: ignore
            try:
                return litellm.ModelResponse(**response_data)  # pyright: ignore
            except Exception as exc:  # pylint: disable=W0718
                raise CompletionCacheCorruptionError(
                    "Cached LiteLLM response cannot be reconstructed as a ModelResponse"
                ) from exc

        return self.completion_cache.complete(
            model=self.model,
            messages=messages,
            completion_kwargs=request_kwargs,
            make_request=_make_request,
            restore_response=_restore_response,
            cache_context=request_cache_context,
        )

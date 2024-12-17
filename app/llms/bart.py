"""
This module implements a BART language model using HuggingFace's transformers library.
It integrates with the LlamaIndex framework for retrieval-augmented generation (RAG) tasks.

Classes:
    BartHuggingFaceLLM: A custom language model class using HuggingFace's BART model for text generation.

Dependencies:
    logging: Used for logging information during model operations.
    threading: Used for handling streaming generation in a separate thread.
    typing: Used for type annotations.
    torch: PyTorch library, used for tensor operations and model handling.
    transformers: HuggingFace transformers library, used for model and tokenizer operations.
    llama_index.core: Contains core functionality for document management and indexing.
    llama_index.core.base.embeddings.base: Provides the base class for embedding models.
    llama_index.core.base.llms.generic_utils: Utility functions for LLMs.
    llama_index.core.base.llms.types: Defines types for chat messages, completion responses, and metadata.
    llama_index.core.bridge.pydantic: Provides fields for pydantic models.
    llama_index.core.callbacks: Manages callback functions for LLMs.
    llama_index.core.constants: Defines constants used across the LlamaIndex framework.
    llama_index.core.llms.callbacks: Callback decorators for LLM completions.
    llama_index.core.llms.custom: Custom LLM base class.
    llama_index.core.prompts.base: Base class for prompt templates.
    llama_index.core.types: Defines types for output parsers and program modes.

Usage:
    This module is designed to be used as part of a larger system that requires text generation capabilities.
    It can be initialized with specific parameters, and used to generate text based on input prompts.

Example:
    Creating an instance of BartHuggingFaceLLM and generating text:

        from transformers import BartForConditionalGeneration, BartTokenizer

        bart_llm = BartHuggingFaceLLM()
        prompt = "Once upon a time"
        response = bart_llm.complete(prompt)
        print(response.text)

Environment Variables:
    None
"""

import logging
from threading import Thread
from typing import Any, Callable, List, Optional, Sequence, Union

import torch
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.core.llms.callbacks import (
    llm_completion_callback,
)
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList, BartForConditionalGeneration, BartTokenizer,
)

logger = logging.getLogger(__name__)


class BartHuggingFaceLLM(CustomLLM):
    """
    A custom language model class using HuggingFace's BART model for text generation.

    This class integrates with the LlamaIndex framework and provides methods for generating text
    completions and streaming completions using the BART model.

    Attributes:
        model_name (str): The model name to use from HuggingFace.
        context_window (int): The maximum number of tokens available for input.
        max_new_tokens (int): The maximum number of tokens to generate.
        system_prompt (str): The system prompt, containing any extra instructions or context.
        query_wrapper_prompt (PromptTemplate): The query wrapper prompt, containing the query placeholder.
        device_map (str): The device map to use for model deployment.
        stopping_ids (List[int]): The stopping ids to use for generation.
        tokenizer_outputs_to_remove (List[str]): The outputs to remove from the tokenizer.
        tokenizer_kwargs (dict): The kwargs to pass to the tokenizer.
        model_kwargs (dict): The kwargs to pass to the model during initialization.
        generate_kwargs (dict): The kwargs to pass to the model during generation.
        is_chat_model (bool): Indicates if the model is a chat model.
        _model (Any): The internal model instance.
        _tokenizer (Any): The internal tokenizer instance.
        _stopping_criteria (Any): The stopping criteria for generation.

    Methods:
        class_name() -> str:
            Returns the class name.
        metadata() -> LLMMetadata:
            Returns the model metadata.
        _tokenizer_messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
            Converts messages to a prompt using the tokenizer.
        complete(prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
            Generates a text completion for the given prompt.
        stream_complete(prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
            Generates a streaming text completion for the given prompt.
    """

    model_name: str = Field(
        default='facebook/bart-large-cnn',
        description=(
            "The model name to use from HuggingFace. "
            "Unused if `model` is passed in directly."
        )
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of tokens available for input.",
        gt=0,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    system_prompt: str = Field(
        default="",
        description=(
            "The system prompt, containing any extra instructions or context. "
            "The model card on HuggingFace should specify if this is needed."
        ),
    )
    query_wrapper_prompt: PromptTemplate = Field(
        default=PromptTemplate("{query_str}"),
        description=(
            "The query wrapper prompt, containing the query placeholder. "
            "The model card on HuggingFace should specify if this is needed. "
            "Should contain a `{query_str}` placeholder."
        ),
    )
    device_map: str = Field(
        default="auto", description="The device_map to use. Defaults to 'auto'."
    )
    stopping_ids: List[int] = Field(
        default_factory=list,
        description=(
            "The stopping ids to use. "
            "Generation stops when these token IDs are predicted."
        ),
    )
    tokenizer_outputs_to_remove: list = Field(
        default_factory=list,
        description=(
            "The outputs to remove from the tokenizer. "
            "Sometimes huggingface tokenizers return extra inputs that cause errors."
        ),
    )
    tokenizer_kwargs: dict = Field(
        default_factory=dict, description="The kwargs to pass to the tokenizer."
    )
    model_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during initialization.",
    )
    generate_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during generation.",
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
                LLMMetadata.__fields__["is_chat_model"].field_info.description
                + " Be sure to verify that you either pass an appropriate tokenizer "
                  "that can convert prompts to properly formatted chat messages or a "
                  "`messages_to_prompt` that does so."
        ), )

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _stopping_criteria: Any = PrivateAttr()

    def __init__(
            self,
            context_window: int = DEFAULT_CONTEXT_WINDOW,
            max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
            query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}",
            device_map: Optional[str] = "auto",
            stopping_ids: Optional[List[int]] = None,
            tokenizer_kwargs: Optional[dict] = None,
            tokenizer_outputs_to_remove: Optional[list] = None,
            model_kwargs: Optional[dict] = None,
            generate_kwargs: Optional[dict] = None,
            is_chat_model: Optional[bool] = False,
            callback_manager: Optional[CallbackManager] = None,
            system_prompt: str = "",
            messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
            completion_to_prompt: Optional[Callable[[str], str]] = None,
            pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
            output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        """
        Initialize the BART language model with the specified parameters.

        Parameters:
            context_window (int): The maximum number of tokens available for input.
            max_new_tokens (int): The maximum number of tokens to generate.
            query_wrapper_prompt (Union[str, PromptTemplate]): The query wrapper prompt.
            device_map (Optional[str]): The device map to use for model deployment.
            stopping_ids (Optional[List[int]]): The stopping ids to use for generation.
            tokenizer_kwargs (Optional[dict]): The kwargs to pass to the tokenizer.
            tokenizer_outputs_to_remove (Optional[list]): The outputs to remove from the tokenizer.
            model_kwargs (Optional[dict]): The kwargs to pass to the model during initialization.
            generate_kwargs (Optional[dict]): The kwargs to pass to the model during generation.
            is_chat_model (Optional[bool]): Indicates if the model is a chat model.
            callback_manager (Optional[CallbackManager]): The callback manager for handling callbacks.
            system_prompt (str): The system prompt, containing any extra instructions or context.
            messages_to_prompt (Optional[Callable[[Sequence[ChatMessage]], str]]): Function to convert messages to a
            prompt.
            completion_to_prompt (Optional[Callable[[str], str]]): Function to convert completions to a prompt.
            pydantic_program_mode (PydanticProgramMode): The Pydantic program mode.
            output_parser (Optional[BaseOutputParser]): The output parser for handling model outputs.
        """
        model_kwargs = model_kwargs or {}
        model_name = "facebook/bart-large-cnn"
        self._model = BartForConditionalGeneration.from_pretrained(model_name, use_safetensors=True, **model_kwargs)

        # check context_window
        config_dict = self._model.config.to_dict()
        model_context_window = int(
            config_dict.get("max_position_embeddings", context_window)
        )
        if model_context_window and model_context_window < context_window:
            logger.warning(
                f"Supplied context_window {context_window} is greater "
                f"than the model's max input size {model_context_window}. "
                "Disable this warning by setting a lower context_window."
            )
            context_window = model_context_window

        # load tokenizer and use safetensors
        tokenizer_kwargs = tokenizer_kwargs or {}
        if "max_length" not in tokenizer_kwargs:
            tokenizer_kwargs["max_length"] = context_window
        self._tokenizer = BartTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        tokenizer_outputs_to_remove = tokenizer_outputs_to_remove.append(
            "past_key_values") if tokenizer_outputs_to_remove else ["past_key_values"]
        generate_kwargs = generate_kwargs.update(
            {"decoder_start_token_id": self._tokenizer.pad_token_id}) if generate_kwargs else {
            'decoder_start_token_id': self._tokenizer.pad_token_id}

        if self._tokenizer.name_or_path != model_name:
            logger.warning(
                f"The model `{model_name}` and tokenizer `{self._tokenizer.name_or_path}` "
                f"are different, please ensure that they are compatible."
            )

        # setup stopping criteria
        stopping_ids_list = stopping_ids or []

        class StopOnTokens(StoppingCriteria):
            def __call__(
                    self,
                    input_ids: torch.LongTensor,
                    scores: torch.FloatTensor,
                    **kwargs: Any,
            ) -> bool:
                for stop_id in stopping_ids_list:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        self._stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        if isinstance(query_wrapper_prompt, str):
            query_wrapper_prompt = PromptTemplate(query_wrapper_prompt)

        messages_to_prompt = messages_to_prompt or self._tokenizer_messages_to_prompt

        super().__init__(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=model_name,
            model_name=model_name,
            device_map=device_map,
            stopping_ids=stopping_ids or [],
            tokenizer_kwargs=tokenizer_kwargs or {},
            tokenizer_outputs_to_remove=tokenizer_outputs_to_remove or [],
            model_kwargs=model_kwargs or {},
            generate_kwargs=generate_kwargs or {},
            is_chat_model=is_chat_model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        """
        Returns the class name.

        Returns:
            str: The class name.
        """
        return "BART_HuggingFace_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        """
        Returns the model metadata.

        Returns:
            LLMMetadata: The model metadata.
        """
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name='facebook/bart-large-cnn',
            is_chat_model=False,
        )

    def _tokenizer_messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """
        Converts messages to a prompt using the tokenizer.

        Parameters:
            messages (Sequence[ChatMessage]): The messages to convert.

        Returns:
            str: The converted prompt.
        """
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages_dict = [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ]
            tokens = self._tokenizer.apply_chat_template(messages_dict)
            return self._tokenizer.decode(tokens)

        return generic_messages_to_prompt(messages)

    @llm_completion_callback()
    def complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """
        Generates a text completion for the given prompt.

        Parameters:
            prompt (str): The input prompt.
            formatted (bool): Indicates if the prompt is already formatted.
            kwargs (Any): Additional keyword arguments.

        Returns:
            CompletionResponse: The generated completion response.
        """
        full_prompt = prompt
        if not formatted:
            if self.query_wrapper_prompt:
                full_prompt = self.query_wrapper_prompt.format(query_str=prompt)
            if self.system_prompt:
                full_prompt = f"{self.system_prompt} {full_prompt}"

        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        inputs = inputs.to(self._model.device)

        # remove keys from the tokenizer if needed, to avoid HF errors
        for key in self.tokenizer_outputs_to_remove:
            if key in inputs:
                inputs.pop(key, None)

        tokens = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self._stopping_criteria,
            **self.generate_kwargs,
        )
        completion = self._tokenizer.decode(tokens[0], skip_special_tokens=True)

        return CompletionResponse(text=completion, raw={"model_output": tokens})

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """
        Generates a streaming text completion for the given prompt.

        Parameters:
            prompt (str): The input prompt.
            formatted (bool): Indicates if the prompt is already formatted.
            kwargs (Any): Additional keyword arguments.

        Returns:
            CompletionResponseGen: The streaming completion response generator.
        """
        from transformers import TextIteratorStreamer

        full_prompt = prompt
        if not formatted:
            if self.query_wrapper_prompt:
                full_prompt = self.query_wrapper_prompt.format(query_str=prompt)
            if self.system_prompt:
                full_prompt = f"{self.system_prompt} {full_prompt}"

        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        inputs = inputs.to(self._model.device)

        # remove keys from the tokenizer if needed, to avoid HF errors
        for key in self.tokenizer_outputs_to_remove:
            if key in inputs:
                inputs.pop(key, None)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self._stopping_criteria,
            **self.generate_kwargs,
        )

        # generate in background thread
        # NOTE/TODO: token counting doesn't work with streaming
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        # create generator based off of streamer
        def gen() -> CompletionResponseGen:
            text = ""
            for x in streamer:
                text += x
                yield CompletionResponse(text=text, delta=x)

        return gen()

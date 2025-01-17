# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.models.llama.chatbot import LLaMA3Chatbot as LLaMA3Chatbot
from fairseq2.models.llama.chatbot import LLaMAChatbot as LLaMAChatbot
from fairseq2.models.llama.chatbot import create_llama_chatbot as create_llama_chatbot
from fairseq2.models.llama.factory import LLAMA_FAMILY as LLAMA_FAMILY
from fairseq2.models.llama.factory import LLaMABuilder as LLaMABuilder
from fairseq2.models.llama.factory import LLaMAConfig as LLaMAConfig
from fairseq2.models.llama.factory import create_llama_model as create_llama_model
from fairseq2.models.llama.factory import get_llama_lora_config as get_llama_lora_config
from fairseq2.models.llama.factory import llama_arch as llama_arch
from fairseq2.models.llama.factory import llama_archs as llama_archs
from fairseq2.models.llama.setup import load_llama_config as load_llama_config
from fairseq2.models.llama.setup import load_llama_model as load_llama_model
from fairseq2.models.llama.setup import load_llama_tokenizer as load_llama_tokenizer
from fairseq2.models.llama.tokenizer import LLaMA3Tokenizer as LLaMA3Tokenizer

# isort: split

from fairseq2.data.text import load_text_tokenizer
from fairseq2.models.chatbot import create_chatbot
from fairseq2.models.loader import load_model


def _register_llama() -> None:
    load_model.register(LLAMA_FAMILY, load_llama_model)

    load_text_tokenizer.register(LLAMA_FAMILY, load_llama_tokenizer)

    create_chatbot.register(LLAMA_FAMILY, create_llama_chatbot)

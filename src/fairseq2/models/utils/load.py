# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, NoReturn, Optional

import torch
from torch.serialization import MAP_LOCATION
from typing_extensions import TypeAlias

from fairseq2.data.typing import PathLike

CheckpointUpgrader: TypeAlias = Callable[[Dict[str, Any]], Dict[str, Any]]


def load_checkpoint(
    pathname: PathLike,
    model_name: str,
    checkpoint_name: Optional[str] = None,
    map_location: MAP_LOCATION = None,
    restrict: bool = False,
    upgrader: Optional[CheckpointUpgrader] = None,
) -> Dict[str, Any]:
    """Load the checkpoint stored in ``pathname``.

    :param pathname:
        The pathname of the checkpoint.
    :param model_name:
        The name of the associated model.
    :param checkpoint_name:
        The name of the checkpoint.
    :param map_location:
        Same as the ``map_location`` parameter of :meth:`torch.load`.
    :param restrict:
        If ``True``, the Python unpickler will be restricted to loading only
        tensors, primitive types, and dictionaries.
    :param upgrader:
        The callable to which the loaded checkpoint will be passed for further
        processing. Typically used to upgrade legacy checkpoints.
    """

    def raise_error(cause: Exception) -> NoReturn:
        if not checkpoint_name:
            display_name = f"checkpoint of the model '{model_name}'"
        else:
            display_name = f"'{checkpoint_name}' checkpoint of the model '{model_name}'"

        raise RuntimeError(
            f"The load of the {display_name} has failed. Please file a bug report."
        ) from cause

    try:
        checkpoint: Dict[str, Any] = torch.load(
            str(pathname), map_location, weights_only=restrict
        )
    except IOError as ex:
        raise_error(ex)

    if upgrader is not None:
        try:
            checkpoint = upgrader(checkpoint)
        except (KeyError, ValueError) as ex:
            raise_error(ex)

    return checkpoint


def upgrade_fairseq_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade a checkpoint generated by the original fairseq.

    .. note::
        This function does not intent to handle all checkpoints generated by
        fairseq. It specificially targets models such as NLLB that have been
        ported to fairseq2.
    """
    old_state_dict = checkpoint["model"]

    new_state_dict = {}

    key_map = _get_fairseq_param_key_map()

    # Convert module keys from fairseq to fairseq2.
    for key in old_state_dict.keys():
        modified_key = key

        for old, new in key_map.items():
            modified_key = modified_key.replace(old, new)

        new_state_dict[modified_key] = old_state_dict[key]

    # Use the built-in version attribute of Module.
    try:
        del new_state_dict["encoder.version"]
    except KeyError:
        pass
    try:
        del new_state_dict["decoder.version"]
    except KeyError:
        pass

    # Positional embeddings don't have to be stored in the checkpoint since
    # we can generate them on-the-fly.
    del new_state_dict["encoder.embed_positions._float_tensor"]
    del new_state_dict["decoder.embed_positions._float_tensor"]

    return {"model": new_state_dict}


def _get_fairseq_param_key_map() -> Dict[str, str]:
    return {
        ".encoder_attn": ".enc_dec_attn",
        ".fc1.": ".ffn.inner_proj.",
        ".fc2.": ".ffn.out_proj.",
        ".final_layer_norm.": ".ffn_layer_norm.",
        "decoder.embed_tokens.weight": "decoder_frontend.embed.weight",
        "decoder.output_projection.weight": "score_proj.weight",
        "encoder.embed_tokens.weight": "encoder_frontend.embed.weight",
        "encoder.subsample.conv_layers": "encoder_frontend.subsampler.convs",
        "encoder.transformer_layers": "encoder.layers",
    }
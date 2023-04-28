# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from fairseq2.nn.utils.mask import compute_mask, to_padding_mask
from tests.common import assert_equal, device


def test_compute_mask_returns_same_number_of_masked_elements_in_each_row() -> None:
    shape = (32, 512)

    mask = compute_mask(shape, span_len=10, max_mask_prob=0.65, device=device)

    assert mask is not None

    num_masked = torch.count_nonzero(mask, dim=-1)

    assert num_masked[0] > 0
    assert num_masked[0] < 512

    assert (num_masked == num_masked[0]).all() == True


def test_compute_mask_returns_mask_with_correct_shape_device_dtype() -> None:
    shape = (4, 16)

    mask = compute_mask(shape, span_len=4, max_mask_prob=1.0, device=device)

    assert mask is not None

    assert mask.shape == shape
    assert mask.device == device
    assert mask.dtype == torch.bool

    assert mask.any()


def test_compute_mask_returns_mask_with_correct_shape_device_dtype_if_row_lens_is_specified() -> None:
    shape = (4, 16)

    row_lens = torch.tensor([16, 14, 15, 16], device="cpu")

    mask = compute_mask(
        shape, span_len=4, max_mask_prob=1.0, device=device, row_lens=row_lens
    )

    assert mask is not None

    assert mask.shape == shape
    assert mask.device == device
    assert mask.dtype == torch.bool

    assert mask.any()


def test_compute_mask_raises_error_if_row_length_is_smaller_than_span_len() -> None:
    shape = (4, 16)

    row_lens = torch.tensor([16, 8, 5, 3], device=device)

    with pytest.raises(
        ValueError,
        match=r"^All lengths in `row_lens` must be greater than 4, but at least one length is smaller\. row_lens: tensor",
    ):
        compute_mask(
            shape, span_len=4, max_mask_prob=1.0, row_lens=row_lens, device=device
        )


def test_to_padding_mask_with_dim1() -> None:
    seq_lens = torch.tensor([4, 2, 0, 5], device=device, dtype=torch.int32)

    mask = to_padding_mask(seq_lens, mask_seq_len=6)

    # fmt: off
    expected_mask = torch.tensor(
        [[False, False, False, False, True,  True],
         [False, False, True,  True,  True,  True],
         [True,  True,  True,  True,  True,  True],
         [False, False, False, False, False, True]], device=device)
    # fmt: on

    assert_equal(mask, expected_mask)


def test_to_padding_mask_with_dim2() -> None:
    seq_lens = torch.tensor([4, 2, 0, 5], device=device, dtype=torch.int32)

    seq_lens = seq_lens.unsqueeze(-1)

    mask = to_padding_mask(seq_lens, mask_seq_len=6)

    # fmt: off
    expected_mask = torch.tensor(
        [[False, False, False, False, True,  True],
         [False, False, True,  True,  True,  True],
         [True,  True,  True,  True,  True,  True],
         [False, False, False, False, False, True]], device=device)
    # fmt: on

    assert_equal(mask, expected_mask)


def test_to_padding_mask_with_dim0() -> None:
    seq_lens = torch.tensor(2, device=device, dtype=torch.int32)

    mask = to_padding_mask(seq_lens, mask_seq_len=4)

    expected_mask = torch.tensor([False, False, True, True], device=device)

    assert_equal(mask, expected_mask)


def test_to_padding_mask_with_single_seq_len() -> None:
    seq_lens = torch.tensor([4], device=device, dtype=torch.int32)

    mask = to_padding_mask(seq_lens, mask_seq_len=6)

    expected_mask = torch.tensor(
        [[False, False, False, False, True, True]], device=device
    )

    assert_equal(mask, expected_mask)
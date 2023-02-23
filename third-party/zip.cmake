# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

macro(fairseq2_add_zip)
    if(NOT TARGET zip::zip)
        set(tmp ${BUILD_SHARED_LIBS})

        # Force the library to be static.
        set(BUILD_SHARED_LIBS FALSE)

        add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/zip EXCLUDE_FROM_ALL)

        # Revert.
        set(BUILD_SHARED_LIBS ${tmp})

        unset(tmp)
    endif()
endmacro()
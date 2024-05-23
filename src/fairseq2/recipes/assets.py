# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import List, Optional, Tuple, final

from fairseq2.assets import AssetStore, default_asset_store
from fairseq2.recipes.cli import Cli, CliCommandHandler
from fairseq2.typing import override


def _setup_asset_cli(cli: Cli) -> None:
    group = cli.add_group(
        "assets", help="list and show assets (e.g. models, tokenizers, datasets)"
    )

    handler = AssetListCommandHandler()

    group.add_command("list", handler, help="list assets")


@final
class AssetListCommandHandler(CliCommandHandler):
    _asset_store: AssetStore

    def __init__(self, asset_store: Optional[AssetStore] = None) -> None:
        self._asset_store = asset_store or default_asset_store

    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        pass

    @override
    def __call__(self, args: Namespace) -> None:
        print("user:")

        self._dump_assets(user=True)

        print()

        print("global:")

        self._dump_assets(user=False)

    def _dump_assets(self, user: bool) -> None:
        assets = self._retrieve_assets(user=user)

        if assets:
            assets.sort(key=lambda e: e[0])

            for source, names in assets:
                names.sort(key=lambda e: e[0])

                print(f"  {source}")

                for name in names:
                    print(f"   - {name}")

                print()
        else:
            print("  n/a")

    def _retrieve_assets(self, user: bool) -> List[Tuple[str, List[str]]]:
        assets = defaultdict(list)

        asset_names = self._asset_store.retrieve_names(user=user)

        for name in asset_names:
            if (env_at := name.find("@")) == -1:
                base_name = name
            else:
                base_name = name[:env_at]

            card = self._asset_store.retrieve_card(
                base_name, ignore_envs=base_name == name
            )

            try:
                source = card.metadata["__source__"]
            except KeyError:
                source = "unknown source"

            assets[source].append(name)

        return [(source, names) for source, names in assets.items()]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, final

from fairseq2.assets.card import AssetCard, AssetCardError
from fairseq2.assets.metadata_provider import (
    AssetMetadataProvider,
    AssetNotFoundError,
    FileAssetMetadataProvider,
    PackageAssetMetadataProvider,
)
from fairseq2.assets.utils import _get_path_from_env
from fairseq2.logging import get_log_writer
from fairseq2.typing import override

log = get_log_writer(__name__)


class AssetStore(ABC):
    """Represents a store of assets."""

    @abstractmethod
    def retrieve_card(self, name: str) -> AssetCard:
        """Retrieve the card of the specified asset.

        :param name:
            The name of the asset.
        """


@final
class StandardAssetStore(AssetStore):
    """Represents a store of assets."""

    env_resolvers: List[EnvironmentResolver]
    metadata_providers: List[AssetMetadataProvider]
    user_metadata_providers: List[AssetMetadataProvider]

    def __init__(self, metadata_provider: AssetMetadataProvider) -> None:
        """
        :param storage:
            The default asset metadata provider.
        """
        self.env_resolvers = []
        self.metadata_providers = [metadata_provider]
        self.user_metadata_providers = []

    @override
    def retrieve_card(self, name: str) -> AssetCard:
        if "@" in name:
            raise ValueError("`name` must not contain the reserved '@' character.")

        envs = self._resolve_envs()

        return self._do_retrieve_card(name, envs)

    def _resolve_envs(self) -> List[str]:
        envs = []

        for resolver in self.env_resolvers:
            if env := resolver():
                envs.append(env)

        # This is a special, always available environment for users to override
        # asset metadata. For instance, a user can set the checkpoint path of a
        # gated model locally by having a same named asset with @user suffix.
        envs.append("user")

        return envs

    def _do_retrieve_card(self, name: str, envs: List[str]) -> AssetCard:
        metadata = self._get_metadata(name)

        # If we have environment-specific metadata, merge it with `metadata`.
        for env in envs:
            try:
                env_metadata = self._get_metadata(f"{name}@{env}")

                # Do not allow overriding 'name'.
                del env_metadata["name"]

                metadata.update(env_metadata)
            except AssetNotFoundError:
                pass

        try:
            base_name = metadata["base"]
        except KeyError:
            base_name = None

        base_card: Optional[AssetCard] = None

        # If the metadata has a base specified, we have to recursively load the
        # entire chain up to the root.
        if base_name:
            if not isinstance(base_name, str):
                raise AssetCardError(
                    f"The value of the field 'base' of the asset card '{name}' must be of type `{str}`, but is of type `{type(base_name)}` instead."
                )

            base_card = self._do_retrieve_card(base_name, envs)

        return AssetCard(metadata, base_card)

    def _get_metadata(self, name: str) -> Dict[str, Any]:
        for provider in reversed(self.user_metadata_providers):
            try:
                return provider.get_metadata(name)
            except AssetNotFoundError:
                continue

        for provider in reversed(self.metadata_providers):
            try:
                return provider.get_metadata(name)
            except AssetNotFoundError:
                continue

        raise AssetNotFoundError(f"An asset with the name '{name}' cannot be found.")

    def clear_cache(self) -> None:
        """Clear the cache of the underlying metadata providers."""
        for provider in self.metadata_providers:
            provider.clear_cache()

        for provider in self.user_metadata_providers:
            provider.clear_cache()

    def add_file_metadata_provider(self, path: Path, user: bool = False) -> None:
        """Add a new :class:`FileAssetMetadataProvider` pointing to ``path``.

        :param path:
            The directory under which asset metadata is stored.
        :param user:
            If ``True``, adds the metadata provider to the user scope.
        """
        providers = self.user_metadata_providers if user else self.metadata_providers

        providers.append(FileAssetMetadataProvider(path))

    def add_package_metadata_provider(self, package_name: str) -> None:
        """Add a new :class:`PackageAssetMetadataProvider` for ``package_name``.

        :param package_name:
            The name of the package in which asset metadata is stored.
        """
        self.metadata_providers.append(PackageAssetMetadataProvider(package_name))


class EnvironmentResolver(Protocol):
    """Resolves the environment within which assets should be loaded.

    Assets can have varying metadata depending on the environment that they are
    loaded in due to regulatory or technical requirements.
    """

    def __call__(self) -> Optional[str]:
        ...


def _create_default_asset_store() -> StandardAssetStore:
    metadata_provider = PackageAssetMetadataProvider("fairseq2.assets.cards")

    return StandardAssetStore(metadata_provider)


default_asset_store = _create_default_asset_store()


def _load_asset_directory() -> None:
    asset_dir = _get_path_from_env("FAIRSEQ2_ASSET_DIR", log)
    if asset_dir is None:
        asset_dir = Path("/etc/fairseq2/assets").resolve()
        if not asset_dir.exists():
            return

    default_asset_store.add_file_metadata_provider(asset_dir)


_load_asset_directory()


def _load_user_asset_directory() -> None:
    asset_dir = _get_path_from_env("FAIRSEQ2_USER_ASSET_DIR", log)
    if asset_dir is None:
        asset_dir = _get_path_from_env("XDG_CONFIG_HOME", log)
        if asset_dir is None:
            asset_dir = Path("~/.config").expanduser()

        asset_dir = asset_dir.joinpath("fairseq2/assets").resolve()
        if not asset_dir.exists():
            return

    default_asset_store.add_file_metadata_provider(asset_dir, user=True)


_load_user_asset_directory()

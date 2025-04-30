"""Used to make pytest functions available globally."""

#  Copyright (c) 2024 zfit
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def setup_teardown():
    import zfit2

    # reset the backend to default
    zfit2.backend.set_backend()

    import gc

    gc.collect()


def pytest_addoption(parser):
    pass


def pytest_configure():
    pass
    # try:
    #     import matplotlib.pyplot as plt
    #     import platform
    #     from pathlib import Path
    #
    #     here = Path(__file__).parent
    #     images_dir = Path(here).joinpath("..", "docs", "images", "_generated_by_tests")
    #     images_dir.mkdir(exist_ok=True)
    #
    #     def savefig(figure=None, folder=None):
    #         """Save the current figure to the images directory.
    #
    #         Skip saving on windows as it fails.
    #         """
    #         # do not save on windows
    #         if platform.system() == "Windows":
    #             return
    #         if figure is None:
    #             figure = plt.gcf()
    #         title_sanitized = (
    #             figure.axes[0].get_title().replace(" ", "_").replace("$", "_").replace("\\", "_").replace("__", "_")
    #         )
    #         title_sanitized = title_sanitized.replace("/", "_").replace(".", "_").replace(":", "_").replace(",", "")
    #         if not title_sanitized:
    #             msg = "Title has to be set for plot that should be saved."
    #             raise RuntimeError(msg)
    #         foldersave = images_dir
    #         if folder is not None:
    #             foldersave = foldersave.joinpath(folder)
    #         foldersave.mkdir(exist_ok=True, parents=True)
    #         savepath = foldersave.joinpath(title_sanitized)
    #         plt.savefig(str(savepath))
    #         plt.close()
    #
    #     pytest.zfit_savefig = savefig
    # except ImportError:
    #     # If matplotlib is not installed, provide a dummy savefig function
    #     def dummy_savefig(*args, **kwargs):
    #         pass
    #
    #     pytest.zfit_savefig = dummy_savefig

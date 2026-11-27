#  @author Giovanni Petrucciani
#  @date 2024-12

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations

import time

# Global default controlling whether a progress bar is shown for distributed
# RDataFrame computations. It applies both to graphs triggered with
# ROOT.RDF.RunGraphs and to graphs triggered implicitly (e.g. by accessing the
# value of an action). It can be changed with `SetProgressBar`.
_PROGRESS_BAR_ENABLED = True


def SetProgressBar(enabled: bool) -> None:
    """
    Globally enable or disable the progress bar visualization of distributed
    RDataFrame computations. This applies both to graphs triggered with
    `RunGraphs` and to graphs triggered implicitly, e.g. by accessing the value
    of an action. The `progressBar` argument of `RunGraphs` takes precedence
    over this setting for the graphs it triggers.
    """
    global _PROGRESS_BAR_ENABLED
    _PROGRESS_BAR_ENABLED = bool(enabled)


def GetProgressBar() -> bool:
    """
    Return whether the progress bar visualization of distributed RDataFrame
    computations is enabled (see `SetProgressBar`).
    """
    return _PROGRESS_BAR_ENABLED


class TaskProgressBar:
    def __init__(self, tasks=(), pollInterval=0.5):
        self._tasks = list(tasks)
        self._pollInterval = pollInterval

    def addFutures(self, tasks):
        self._tasks += tasks

    def run(self) -> None:
        if not self._tasks:
            return
        try:
            import tqdm

            self._run_tqdm(tqdm.tqdm)
        except ImportError:
            self._run_basic()
        self._tasks.clear()

    def _run_tqdm(self, tqdm) -> None:
        last_total = len(self._tasks)
        last_ndone = 0
        with tqdm(total=last_total) as pbar:
            waited = 0
            while True:
                states = [f.done() for f in self._tasks]
                total = len(states)
                ndone = sum(states)
                if ndone == total:
                    break
                waited += 1
                if last_total != total:
                    pbar.total = total
                    pbar.n = ndone
                    pbar.refresh()
                    last_total = total
                    last_ndone = ndone
                    waited = 0
                elif (ndone != last_ndone) or (waited > 0):
                    pbar.update(ndone - last_ndone)
                    last_ndone = ndone
                    waited = 0
                time.sleep(self._pollInterval)

    def _run_basic(self) -> None:
        last_total = len(self._tasks)
        last_ndone = 0
        started = time.monotonic()
        waited = 0
        interval = int(max(1, 2 / self._pollInterval))  # one dot every 2 s
        while True:
            states = [f.done() for f in self._tasks]
            total = len(states)
            ndone = sum(states)
            waited += 1
            now = time.monotonic()
            if ndone != last_ndone or total != last_total:
                print(
                    f"after {now - started:.0f}s, {ndone}/{total} done ({ndone / total * 100:3.0f}%)",
                    end=("\n" if ndone == total else ""),
                    flush=True,
                )
                last_ndone = ndone
                last_total = total
                waited = 0
                if ndone == total:
                    break
            elif waited == interval:
                print(".", end="", flush=True)
                waited = 0
                if now - started > 10 * 60:
                    interval = int(max(1, 2 * 60 / self._pollInterval))  # one dot every 2 mins
                elif now - started > 60:
                    interval = int(max(1, 10 / self._pollInterval))  # one dot every 10 s
            time.sleep(self._pollInterval)

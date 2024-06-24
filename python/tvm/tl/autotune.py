# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Autotuner for TL programs."""

from typing import List
import tempfile
import os

from tvm.runtime import load_module
from tvm.contrib.popen_pool import PopenPoolExecutor

from .engine import lower
from .utils import Profiler


def compile_and_save(func):
    mod, params = lower(func)
    tmp = tempfile.mktemp(suffix=".so")
    mod.export_library(tmp)
    return tmp, params


class AutoTuner(object):
    def __init__(self, max_workers=None) -> None:
        self.pool = PopenPoolExecutor(max_workers=max_workers, timeout=30)

    def run(self, func_template: callable, configs: List, return_all: bool = True):
        assert len(configs) > 0
        futures = []
        for config in configs:
            if isinstance(config, dict):
                func = func_template(**config)
            else:
                func = func_template(*config)
            futures.append(self.pool.submit(compile_and_save, func))

        results = []
        for future in futures:
            try:
                tmp, params = future.result()
            except ChildProcessError:
                continue
            mod = load_module(tmp)
            os.remove(tmp)
            profiler = Profiler(mod, params, [])
            latency = profiler.do_bench(profiler.func)
            results.append([config, mod, latency])

        if len(results) == 0:
            return None
        results = sorted(results, key=lambda x: x[2])
        if return_all:
            return results
        else:
            return results[0]

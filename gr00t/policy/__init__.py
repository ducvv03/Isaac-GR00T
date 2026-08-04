# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .policy import BasePolicy, PolicyWrapper


__all__ = [
    "BasePolicy",
    "Gr00tPolicy",
    "PolicyWrapper",
]


def __getattr__(name):
    # Gr00tPolicy pulls in torch; deferred so lightweight clients (see
    # getting_started/policy.md's server-client section) can import
    # gr00t.policy.server_client without a torch install.
    if name == "Gr00tPolicy":
        from .gr00t_policy import Gr00tPolicy

        return Gr00tPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

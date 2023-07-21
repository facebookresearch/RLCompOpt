
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import distutils.util

import setuptools

setuptools.setup(
    name="rlcompopt",
    version="0.0.1",
    description="Compiler pass ordering with machine learning",
    author="Facebook AI Research",
    url="https://github.com/facebookresearch/RLCompOpt",
    license="MIT",
    packages=[
        "rlcompopt",
        "rlcompopt.env_wrapper",
        "rlcompopt.cl",
        "rlcompopt.cl.models",
        "rlcompopt.pipeline",
        "rlcompopt.pipeline.lib",
    ],
    package_data={
        "rlcompopt": [
            "env_wrapper/database_schema4.sql",
            "env_wrapper/database_schema2.sql",
            "env_wrapper/database_schema.sql",
            "env_wrapper/database_model.sql",
            "cl/conf/*",
            "cl/conf/model/*",
            "cl/database_socket.sql",
        ]
    },
    python_requires=">=3.8",
    platforms=[distutils.util.get_platform()],
    zip_safe=False,
)

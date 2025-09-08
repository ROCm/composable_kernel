# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# generate kernel instances to speed up compilation

import os.path as path


def update_file(file_path, content):
    """Update the file at file_path with the given content if it differs from the existing content.

    It avoids unnecessary touching of the file which triggers rebuilds
    """

    existing_content = ""
    if path.exists(file_path):
        with open(file_path, "r") as file:
            existing_content = file.read()
    if existing_content == content:
        return
    with open(file_path, "w") as file:
        file.write(content)

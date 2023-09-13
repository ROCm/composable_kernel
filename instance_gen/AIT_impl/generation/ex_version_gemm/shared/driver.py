import enum
import ck_types
from copy import deepcopy
from dataclasses import dataclass
from enum import auto
from typing import List
import os.path
import shutil
import functools
import operator
import collections
import subprocess
import re
import gemm_op
from gemm_op import *
import user
from ck_types import *
from gemm_ex import *
from make_template import *

# holds multiple gemm instances
op_collection = user.CreateGemmOperator()
instances = []
for op in op_collection:
    instances.append((str(op.tile_desc.block_size) + "_" + str(op.tile_desc.m_per_block) + "_" + str(op.tile_desc.n_per_block) + "_" + str(op.tile_desc.k_per_block) + "_" + str(op.tile_desc.k1) + ".o "))
    x = EmitGemmInstance()
    x.emit(op)
m = EmitMake()
m.emit(instances)
#print(str(instances))

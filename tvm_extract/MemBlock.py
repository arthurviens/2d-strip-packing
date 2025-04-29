import hashlib
import tvm
import numpy as np
from typing import Tuple, Optional, Sequence, List

VERBOSE = True

class MemBlock:
    """
    Abstraction representing a memory-allocated tensor (intermediate, input, or output).
    Holds metadata like shape, dtype, memory size, origin of allocation, and dependency info.
    """
    def __init__(
            self, 
            name: str, 
            shape: Sequence[int], 
            dtype: str, 
            origin: str=None,
            depends_on: List['MemBlock']=None, 
            links_to: List['MemBlock']=None, 
            first_used: Optional[int]=None, 
            last_used: Optional[int]=None, 
            verbose:bool =False
        ):
        self.name = name  # human-readable tensor name (e.g., lv, lv3, pad_temp, etc.)
        self.shape = shape  # tuple of dimensions (e.g., (1, 64, 128, 128))
        self.dtype = dtype  # data type string (e.g., "float32")
        self.origin = origin  # where the tensor came from (Relax func, TIR PrimFunc, etc.)
        self.depends_on = depends_on or []  # tensors this MemBlock depends on
        self.links_to = links_to or []  # for composed MemBlocks (e.g., function output containing intermediates)
        self.first_used = first_used  # index/time of first usage (optional for lifetime modeling)
        self.last_used = last_used  # index/time of last usage (optional)
        self._id = MemBlock.compute_id(shape, dtype, origin)

    @staticmethod
    def compute_id(shape, dtype, origin: str=None):
        if origin and origin.startswith("relax"):
            origin = "relax"
        return hashlib.sha1(
            (str(shape) + str(dtype) + str(origin)).encode()
        ).hexdigest()[:8]

    @staticmethod
    def compute_id_from_buffer(buffer, origin=None):
        shape = tuple(int(dim) for dim in buffer.shape)
        dtype = buffer.dtype
        return MemBlock.compute_id(shape, dtype, origin=origin)

    @staticmethod
    def compute_id_from_relax_varbinding(vb: tvm.relax.VarBinding, origin=None):
        shape = tuple(int(dim) for dim in vb.value.sinfo_args[0].shape)
        dtype = vb.value.sinfo_args[0].dtype
        return MemBlock.compute_id(shape, dtype, origin=origin)

    @staticmethod
    def compute_id_from_relax_sinfo(sinfo: tvm.relax.TensorStructInfo, origin=None):
        shape = tuple(int(dim) for dim in sinfo.shape)
        dtype = sinfo.dtype
        return MemBlock.compute_id(shape, dtype, origin=origin)

    @property
    def size_bytes(self):
        # Computes total memory usage in bytes
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize

    def __repr__(self):
        global VERBOSE
        if not VERBOSE:
            return f"MemBlock({self.name}, shape={self.shape}, dtype={self.dtype}, size={self.size_bytes})"
        else:
            return f"MemBlock({self.name}:{self._id}, shape={self.shape}, dtype={self.dtype}, size={self.size_bytes}, origin={self.origin})"

    @staticmethod
    def from_tir_buffer(name: str, buffer, origin=None):
        """
        Builds a MemBlock from a TIR Buffer object, extracting shape and dtype.
        """
        shape = tuple(int(dim) for dim in buffer.shape)
        dtype = buffer.dtype
        return MemBlock(name, shape, dtype, origin=origin)

    @staticmethod
    def from_struct_info(name: str, sinfo: tvm.relax.struct_info.TensorStructInfo, origin: str=None):
        """
        Builds a MemBlock from a Relax function output var and its StructInfo.
        """
        shape = tuple(int(s) for s in sinfo.shape)
        dtype = sinfo.dtype
        return MemBlock(name, shape, dtype, origin=origin)

import hashlib
import tvm
import numpy as np

VERBOSE = True

class MemBlock:
    """
    Abstraction representing a memory-allocated tensor (intermediate, input, or output).
    Holds metadata like shape, dtype, memory size, origin of allocation, and dependency info.
    """
    def __init__(self, name, shape, dtype, origin=None, depends_on=None, links_to=None, first_used=None, last_used=None, use_origin_for_id=False, verbose=False):
        self.name = name  # human-readable tensor name (e.g., lv, lv3, pad_temp, etc.)
        self.shape = shape  # tuple of dimensions (e.g., (1, 64, 128, 128))
        self.dtype = dtype  # data type string (e.g., "float32")
        self.origin = origin  # where the tensor came from (Relax func, TIR PrimFunc, etc.)
        self.use_origin_for_id = use_origin_for_id
        self.depends_on = depends_on or []  # tensors this MemBlock depends on
        self.links_to = links_to or []  # for composed MemBlocks (e.g., function output containing intermediates)
        self.first_used = first_used  # index/time of first usage (optional for lifetime modeling)
        self.last_used = last_used  # index/time of last usage (optional)
        self._id = MemBlock.compute_id(name, shape, dtype, origin)

    @staticmethod
    def compute_id(name, shape, dtype, origin: str=None):
        # return hashlib.sha1(
        #     (name+str(shape)+str(dtype)+str(origin)).encode()
        # ).hexdigest()[:8]
        return hashlib.sha1(
            (name + str(shape) + str(dtype)).encode()
        ).hexdigest()[:8]

    @staticmethod
    def compute_id_from_buffer(name, buffer, origin=None):
        shape = tuple(int(dim) for dim in buffer.shape)
        dtype = buffer.dtype
        return MemBlock.compute_id(name, shape, dtype, origin=origin)

    @staticmethod
    def compute_id_from_relax_varbinding(vb: tvm.relax.VarBinding, origin='relax.output'):
        shape = tuple(int(dim) for dim in vb.value.sinfo_args[0].shape)
        dtype = vb.value.sinfo_args[0].dtype
        return MemBlock.compute_id(str(vb.var), shape, dtype, origin=origin)

    @staticmethod
    def compute_id_from_relax_var_sinfo(var: tvm.relax.Var, sinfo: tvm.relax.TensorStructInfo, origin=None):
        shape = tuple(int(dim) for dim in sinfo.shape)
        dtype = sinfo.dtype
        return MemBlock.compute_id(str(var), shape, dtype, origin=origin)

    @property
    def size_bytes(self):
        # Computes total memory usage in bytes
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize

    def walk_dependencies(self):
        # Depth-first traversal of all transitive dependencies
        for dep in self.depends_on:
            dep.walk_dependencies()

    def __repr__(self):
        global VERBOSE
        if not VERBOSE:
            return f"MemBlock({self.name}, shape={self.shape}, dtype={self.dtype}, size={self.size_bytes})"
        else:
            return f"MemBlock({self.name}:{self._id}, shape={self.shape}, dtype={self.dtype}, size={self.size_bytes}, origin={self.origin})"

    @staticmethod
    def from_tir_buffer(name, buffer, origin=None):
        """
        Builds a MemBlock from a TIR Buffer object, extracting shape and dtype.
        """
        # print(f"Shape when it works {buffer.shape}")
        shape = tuple(int(dim) for dim in buffer.shape)
        dtype = buffer.dtype
        return MemBlock(name, shape, dtype, origin=origin)

    @staticmethod
    def from_relax_output(var: tvm.relax.Var, sinfo: tvm.relax.struct_info.TensorStructInfo):
        """
        Builds a MemBlock from a Relax function output var and its StructInfo.
        """
        shape = tuple(int(s) for s in sinfo.shape)
        dtype = sinfo.dtype
        return MemBlock(str(var), shape, dtype, origin="relax.output")
    
    @staticmethod
    def from_relax_input(var: tvm.relax.Var, sinfo: tvm.relax.struct_info.TensorStructInfo):
        """
        Builds a MemBlock from a Relax function output var and its StructInfo.
        """
        shape = tuple(int(s) for s in sinfo.shape)
        dtype = sinfo.dtype
        return MemBlock(str(var), shape, dtype, origin="relax.input")

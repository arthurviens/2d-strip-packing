import hashlib
import tvm
import numpy as np
from typing import Tuple, Optional, Sequence, List
import uuid

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
            origin: str = None,
            depends_on: List['MemBlock'] = None, 
            links_to: List['MemBlock'] = None, 
            first_used: Optional[int] = None, 
            last_used: Optional[int] = None, 
            verbose: bool = False,
            unique_id: str = None
        ):
        self.name = name  # human-readable tensor name (e.g., lv, lv3, pad_temp, etc.)
        self.shape = tuple(shape)  # tuple of dimensions (e.g., (1, 64, 128, 128))
        self.dtype = dtype  # data type string (e.g., "float32")
        self.origin = origin  # where the tensor came from (Relax func, TIR PrimFunc, etc.)
        self.depends_on = depends_on or []  # tensors this MemBlock depends on
        self.links_to = links_to or []  # tensors that depend on this one
        self.first_used = first_used  # index/time of first usage (optional for lifetime modeling)
        self.last_used = last_used  # index/time of last usage (optional)
        
        # Identity management: unique per instance, not content-based
        self._id = unique_id or str(uuid.uuid4())[:8]
        
        # Keep a content signature for debugging/logging purposes only
        self._content_signature = self._compute_content_signature()

    def _compute_content_signature(self):
        """Compute a content-based signature for debugging purposes only."""
        return hashlib.sha1(
            (str(self.shape) + str(self.dtype) + str(self.origin)).encode()
        ).hexdigest()[:8]

    @property
    def size_bytes(self):
        """Computes total memory usage in bytes."""
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize

    def __repr__(self):
        global VERBOSE
        if not VERBOSE:
            return f"MemBlock({self.name}, shape={self.shape}, dtype={self.dtype}, size={self.size_bytes})"
        else:
            return f"MemBlock({self.name}:{self._id}, shape={self.shape}, dtype={self.dtype}, size={self.size_bytes}, origin={self.origin})"

    def __eq__(self, other):
        """Equality based on unique ID, not content."""
        if not isinstance(other, MemBlock):
            return False
        return self._id == other._id

    def __hash__(self):
        """Hash based on unique ID."""
        return hash(self._id)

    @staticmethod
    def from_tir_buffer(name: str, buffer, origin=None, unique_id=None):
        """
        Builds a MemBlock from a TIR Buffer object, extracting shape and dtype.
        """
        shape = tuple(int(dim) for dim in buffer.shape)
        dtype = buffer.dtype
        return MemBlock(name, shape, dtype, origin=origin, unique_id=unique_id)

    @staticmethod
    def from_struct_info(name: str, sinfo: tvm.relax.struct_info.TensorStructInfo, origin: str = None, unique_id=None):
        """
        Builds a MemBlock from a Relax function output var and its StructInfo.
        """
        shape = tuple(int(s) for s in sinfo.shape)
        dtype = sinfo.dtype
        return MemBlock(name, shape, dtype, origin=origin, unique_id=unique_id)

    @staticmethod
    def compute_id(shape, dtype, origin: str = None):
        """
        DEPRECATED: Content-based ID computation. 
        Use structural identity tracking instead.
        """
        import warnings
        warnings.warn("compute_id is deprecated. Use structural identity tracking.", DeprecationWarning)
        return hashlib.sha1(
            (str(shape) + str(dtype) + str(origin)).encode()
        ).hexdigest()[:8]

    @staticmethod
    def compute_id_from_buffer(buffer, origin=None):
        """DEPRECATED: Use structural identity tracking instead."""
        import warnings
        warnings.warn("compute_id_from_buffer is deprecated.", DeprecationWarning)
        shape = tuple(int(dim) for dim in buffer.shape)
        dtype = buffer.dtype
        return MemBlock.compute_id(shape, dtype, origin=origin)

    @staticmethod
    def compute_id_from_relax_varbinding(vb: tvm.relax.VarBinding, origin=None):
        """DEPRECATED: Use structural identity tracking instead."""
        import warnings
        warnings.warn("compute_id_from_relax_varbinding is deprecated.", DeprecationWarning)
        shape = tuple(int(dim) for dim in vb.value.sinfo_args[0].shape)
        dtype = vb.value.sinfo_args[0].dtype
        return MemBlock.compute_id(shape, dtype, origin=origin)

    @staticmethod
    def compute_id_from_relax_sinfo(sinfo: tvm.relax.TensorStructInfo, origin=None):
        """DEPRECATED: Use structural identity tracking instead."""
        import warnings
        warnings.warn("compute_id_from_relax_sinfo is deprecated.", DeprecationWarning)
        shape = tuple(int(dim) for dim in sinfo.shape)
        dtype = sinfo.dtype
        return MemBlock.compute_id(shape, dtype, origin=origin)

    # Utility methods for dependency management
    def add_dependency(self, other: 'MemBlock'):
        """Add a dependency: this MemBlock depends on 'other'."""
        if other not in self.depends_on:
            self.depends_on.append(other)
        if self not in other.links_to:
            other.links_to.append(self)

    def remove_dependency(self, other: 'MemBlock'):
        """Remove a dependency."""
        if other in self.depends_on:
            self.depends_on.remove(other)
        if self in other.links_to:
            other.links_to.remove(self)

    def get_all_dependencies(self, visited=None):
        """Get all transitive dependencies."""
        if visited is None:
            visited = set()
        
        if self in visited:
            return set()
        
        visited.add(self)
        deps = set(self.depends_on)
        
        for dep in self.depends_on:
            deps.update(dep.get_all_dependencies(visited))
        
        return deps

    def is_leaf(self):
        """Returns True if this MemBlock has no dependencies."""
        return len(self.depends_on) == 0

    def is_root(self):
        """Returns True if no other MemBlocks depend on this one."""
        return len(self.links_to) == 0
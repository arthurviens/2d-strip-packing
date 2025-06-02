# System imports
from typing import Optional, List, Dict, Set, Tuple
import warnings

# 3rd party Library imports
import tvm
from tvm import relax, tir

# Local imports
from MemBlock import MemBlock

def _buffer_matches_output_shape(buffer: tir.Buffer, output_mb: MemBlock) -> bool:
    """Check if a TIR buffer matches the expected output MemBlock shape."""
    try:
        buffer_shape = tuple(int(dim) for dim in buffer.shape)
        return buffer_shape == output_mb.shape and str(buffer.dtype) == str(output_mb.dtype)
    except:
        return False

class AllocationFinder(relax.PyExprVisitor):
    """
    Improved version that tracks tensor identity structurally rather than through content hashing.
    Fixed to handle multiple calls to the same TIR function with different parameters.
    """
    def __init__(self, mod: tvm.IRModule, verbose=False) -> None:
        super().__init__()
        self.mod_ = mod
        self.verbose = verbose
        
        # Core data structures
        self.memblocks: Dict[str, List[MemBlock]] = {}  # function_name -> [MemBlock]
        
        # Identity tracking - the key improvement
        self.relax_var_to_memblock: Dict[relax.Var, MemBlock] = {}  # Direct var -> MemBlock mapping
        
        # Track per-context mappings
        self.tir_buffer_to_memblock: Dict[Tuple[tir.Buffer, int], MemBlock] = {}  # (Buffer, context_id) -> MemBlock mapping

        # Call context tracking
        self.call_tir_contexts: Dict[str, List['CallTirContext']] = {}  # tir_func_name -> [context]
        
        # For dependency resolution
        self.current_function = None
        
    def add_memblock(self, function_name: str, mb: MemBlock) -> None:
        """Add a MemBlock to the specified function."""
        if self.verbose:
            print(f"    [add_mb] adding {function_name}::{mb._id} ({mb.name} : {mb.shape}+{mb.dtype}+{mb.origin})")
        
        self.memblocks.setdefault(function_name, []).append(mb)

    def walk(self):
        """Walk the IR in dependency order: Relax first, then TIR."""
        # Pass 1: Process all Relax functions to establish the call graph
        for gv, func in self.mod_.functions.items():
            if isinstance(func, relax.Function):
                self.visit_relax_func(gv.name_hint, func)
        
        # Pass 2: Process TIR functions with full context
        for gv, func in self.mod_.functions_items():
            if isinstance(func, tvm.tir.PrimFunc):
                self.visit_tir_func(gv.name_hint, func)

    def visit_relax_func(self, name_hint: str, func: relax.Function):
        """Process Relax function - establish primary tensor identities."""
        self.current_function = name_hint

        # Process input parameters and create Relax MemBlocks
        for var, sinfo in zip(func.params, func.struct_info.params):
            mb = MemBlock.from_struct_info(str(var), sinfo, origin=f"relax.input.{name_hint}")
            self.add_memblock(name_hint, mb)
            self.relax_var_to_memblock[var] = mb  # Var to mb Direct mapping
            
        # Process function body
        for binding in func.body.blocks[0].bindings:
            if isinstance(binding.value, relax.Call) and binding.value.op.name == "relax.call_tir":
                self._process_call_tir(name_hint, binding)  # Create output memblock and get context

    def _process_call_tir(self, relax_func_name: str, binding):
        """Process a relax.call_tir operation."""
        tir_func_name = binding.value.args[0].name_hint
        output_sinfo = binding.value.sinfo_args[0]
        input_vars = [arg for arg in binding.value.args[1].fields]  # The Tuple of input vars
        
        # Create output MemBlock
        output_mb = MemBlock.from_struct_info(
            str(binding.var),
            output_sinfo,
            origin=f"relax.call_tir.{tir_func_name}"
        )
        self.add_memblock(relax_func_name, output_mb)
        self.relax_var_to_memblock[binding.var] = output_mb

        # Store call context for TIR processing
        context = CallTirContext(
            tir_func_name=tir_func_name,
            input_memblocks=[self.relax_var_to_memblock[var] for var in input_vars],
            output_memblock=output_mb,
            input_var_names=[str(var) for var in input_vars]
        )
        
        # Store multiple contexts per TIR function
        if tir_func_name not in self.call_tir_contexts:
            self.call_tir_contexts[tir_func_name] = []
        self.call_tir_contexts[tir_func_name].append(context)

    def visit_tir_func(self, name_hint: str, func: tvm.tir.PrimFunc):
        """Process TIR function using call context.
        Called after visit_relax_func"""
        self.current_function = name_hint
        if self.verbose:
            print(f"[visit_tir_func] Visiting {name_hint}")
        
        # Get all call contexts for this TIR function
        contexts = self.call_tir_contexts.get(name_hint, [])
        
        if not contexts:
            # Standalone TIR function - create MemBlocks for parameters
            for param in func.params:
                buffer = func.buffer_map.get(param)
                if buffer:
                    mb = MemBlock.from_tir_buffer(
                        buffer.name, 
                        buffer, 
                        origin=f"tir.param.{name_hint}"
                    )
                    self.add_memblock(name_hint, mb)
                    # Use context_id=0 for standalone functions
                    self.tir_buffer_to_memblock[(buffer, 0)] = mb
        else:
            # Process each context separately and create separate mappings
            for context_idx, context in enumerate(contexts):
                if self.verbose:
                    print(f"  [visit_tir_func] Processing context {context_idx + 1}/{len(contexts)} for {name_hint}")
                
                # First, determine which parameters are outputs by analyzing the function
                self._identify_output_parameters(func, context)

                # Map TIR parameters to existing Relax MemBlocks for this specific context
                for i, param in enumerate(func.params):
                    buffer = func.buffer_map.get(param)
                    if self.verbose:
                        print(f"    [visit_tir_func] func.buffer_map ({type(func.buffer_map)}) : {func.buffer_map}")

                    if buffer:
                        # Create context-specific mapping
                        buffer_key = (buffer, context_idx)
                        
                        if self.verbose:
                            print(f"    [visit_tir_func] {context.print()}")
                        if i in context.output_param_indices:
                            # Output parameter - map to the Relax call_tir output MemBlock
                            self.tir_buffer_to_memblock[buffer_key] = context.output_memblock
                            if self.verbose:
                                print(f"    [visit_tir_func] Mapped output buffer {buffer.name} (ctx {context_idx}) to Relax output {context.output_memblock.name}")
                        elif i < len(context.input_memblocks):
                            # Input parameter - map to existing input MemBlock
                            self.tir_buffer_to_memblock[buffer_key] = context.input_memblocks[i]
                            if self.verbose:
                                print(f"    [visit_tir_func] Mapped input buffer {buffer.name} (ctx {context_idx}) to Relax input {context.input_memblocks[i].name}")
        
        # FIXED: Process internal allocations and build dependency graph for each context
        for context_idx, context in enumerate(contexts if contexts else [None]):
            self._process_tir_body(name_hint, func.body, context, context_idx)
    
    def _identify_output_parameters(self, func: tvm.tir.PrimFunc, context: 'CallTirContext'):
        """
        Identify which TIR function parameters are outputs by analyzing buffer usage.
        In TVM, output parameters are typically those that are written to but not read from.
        """
        param_buffers = [func.buffer_map.get(param) for param in func.params]
        
        # Analyze all blocks to see which buffers are primarily written to
        def analyze_buffer_usage(stmt):
            if isinstance(stmt, tir.Block):
                written_buffers = {write.buffer for write in stmt.writes}
                read_buffers = {read.buffer for read in stmt.reads}
                
                for i, buffer in enumerate(param_buffers):
                    if buffer and buffer in written_buffers:
                        # Check if this buffer is primarily an output
                        # Heuristic: if it's written to but never read from in this block,
                        # or if it matches the expected output shape, it's likely an output
                        if (buffer not in read_buffers or 
                            _buffer_matches_output_shape(buffer, context.output_memblock)):
                            context.mark_output_param(i)
            return True
        
        tvm.tir.stmt_functor.pre_order_visit(func.body, analyze_buffer_usage)
        
        # Fallback: if no clear outputs identified, assume last parameter is output
        if not context.output_param_indices and param_buffers:
            context.mark_output_param(len(param_buffers) - 1)
            if self.verbose:
                print(f"    [tir] Fallback: assuming last parameter is output")

    def _process_tir_body(self, func_name: str, stmt, context: Optional['CallTirContext'], context_idx: int):
        """Process TIR function body to find allocations and dependencies."""
        
        def visit_allocations(node):
            if isinstance(node, tir.Block):
                # Process allocated buffers - these are shared across contexts
                for buf in node.alloc_buffers:
                    # FIXED: Only create MemBlock once per allocated buffer, not per context
                    existing_mb = None
                    for existing_key, mb in self.tir_buffer_to_memblock.items():
                        if existing_key[0] == buf and mb.origin == f"tir.alloc.{func_name}":
                            existing_mb = mb
                            break
                    
                    if not existing_mb:
                        mb = MemBlock.from_tir_buffer(
                            buf.name, 
                            buf, 
                            origin=f"tir.alloc.{func_name}"
                        )
                        self.add_memblock(func_name, mb)
                        # Use context_idx for the key, but the MemBlock is shared conceptually
                        self.tir_buffer_to_memblock[(buf, context_idx)] = mb
                    else:
                        # Reuse existing MemBlock for this context
                        self.tir_buffer_to_memblock[(buf, context_idx)] = existing_mb
                
                # Build dependencies for this specific context
                self._build_tir_dependencies(node, context_idx)
            return True
        
        tvm.tir.stmt_functor.pre_order_visit(stmt, visit_allocations)
    
    def _build_tir_dependencies(self, block: tir.Block, context_idx: int):
        """Build dependency relationships within a TIR block for a specific context."""
        # Get MemBlocks for read/write buffers using context-specific keys
        write_memblocks = []
        for write in block.writes:
            mb = self.tir_buffer_to_memblock.get((write.buffer, context_idx))
            if mb:
                write_memblocks.append(mb)

        read_memblocks = []
        for read in block.reads:
            mb = self.tir_buffer_to_memblock.get((read.buffer, context_idx))
            if mb:
                read_memblocks.append(mb)
        
        if self.verbose:
            print(f"    [build_tir_dep] ({block.name_hint}) ctx {context_idx} readers {[(x.name, x._id) for x in read_memblocks]} writers {[(x.name, x._id) for x in write_memblocks]}")
        
        # Establish dependencies: each write depends on all reads
        for writer in write_memblocks:
            for reader in read_memblocks:
                if writer._id != reader._id:  # Don't self-depend
                    if self.verbose:
                        print(f"    [_build_tir_dep] ctx {context_idx}: {writer.name}:{writer._id} <-> {reader.name}:{reader._id}")
                    # Avoid duplicate dependencies
                    if reader not in writer.depends_on:
                        writer.depends_on.append(reader)
                    if writer not in reader.links_to:
                        reader.links_to.append(writer)

    def print_elements(self):
        """Print all discovered MemBlocks."""
        if self.memblocks:
            for func, mbs in self.memblocks.items():
                print(f"Function: {func}")
                for mb in mbs:
                    print("   ", mb)
                    if mb.depends_on:
                        print(f"      depends_on: {[dep.name for dep in mb.depends_on]}")
                    if mb.links_to:
                        print(f"      links_to: {[link.name for link in mb.links_to]}")
        else:
            warnings.warn("No memblocks found. Did you call walk()?")


class CallTirContext:
    """Context information for a call_tir operation."""
    def __init__(self, tir_func_name: str, input_memblocks: List[MemBlock], 
                 output_memblock: MemBlock, input_var_names: List[str]):
        self.tir_func_name = tir_func_name
        self.input_memblocks = input_memblocks
        self.output_memblock = output_memblock
        self.input_var_names = input_var_names
        # Track which TIR parameters correspond to outputs
        self.output_param_indices: Set[int] = set()
        
    def mark_output_param(self, param_index: int):
        """Mark a TIR parameter as an output parameter."""
        self.output_param_indices.add(param_index)

    def print(self):
        """Debug"""
        return f"Context '{self.tir_func_name}' inputs ({self.input_var_names} -> {self.input_memblocks}) outputs {self.output_memblock} and output_param_indices {self.output_param_indices}"
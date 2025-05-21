# Library imports
import tvm
import warnings
from tvm import relax, tir

# Local imports
from MemBlock import MemBlock

INSIGHT = None

class AllocationFinder(relax.PyExprVisitor):
    """
    Traverses both Relax and TIR parts of a TVM IRModule to extract all tensors and their memory allocations.
    Creates MemBlocks for:
      - Relax call_tir outputs
      - TIR alloc_buffer / Allocate statements
      - TIR BufferStore targets
    """
    def __init__(self, mod: tvm.IRModule, verbose=False) -> None:
        super().__init__()
        self.mod_ = mod
        self.verbose = verbose
        self.memblocks = {}  # maps function name → List of MemBlock objects
        self.id_to_memblock = {}  # buffer name -> MemBlock (for dependency resolution)

    def add_memblock(self, name_hint:str, mb: MemBlock) -> None:
        """
        Adds a MemBlock by adding a MemBlock object to self.memblocks[name_hint], where name_hint is the current function
        and performs checks and needed side effects
            - Checks if name_hint key exists, if not, creates the empty list
            - Checks that the added MemBlock is not identical to the last one added, else it is a duplicate to be skipped
            - Adds the corresponding entry to 'id_to_memblock' dictionary.
        """
        if name_hint in self.memblocks.keys() and mb._id in [x._id for x in self.memblocks[name_hint]]:
            if self.memblocks[name_hint][-1]._id == mb._id:
                # Means that the last MemBlock was exactly the same one => We don't add it.
                # it's the case when buffers init (i.e at 0.0) then modify the value but the AST suggests 2 allocations
                return  # Skip adding this MemBlock as it's already in
            raise RuntimeError(
                "Two memblocks have the same id, and are not adjacent. Rethink your code."
                "This means the same (name + shape + dtype) occurs across at least two PrimFuncs/R.Function"
            )

        self.memblocks.setdefault(name_hint, []).append(mb)
        self.id_to_memblock[mb._id] = mb

    def walk(self):
        """
        Walk over all functions in the IRModule in two passes
        - First, relax functions
        - Then,  TIR PrimFunc
        """
        # 1st Pass: Relax.Function
        for gv, func in self.mod_.functions.items():
            if isinstance(func, relax.Function):
                self.visit_relax_func(gv.name_hint, func)
        
        # 2nd Pass: tir.PrimFunc
        for gv, func in self.mod_.functions_items():
            if isinstance(func, tvm.tir.PrimFunc):
                self.visit_tir_func(gv.name_hint, func)

    def visit_relax_func(self, name_hint: str, func: relax.Function):
        """
        Parses a Relax function to find all call_tir outputs, and wraps them in MemBlocks.
        """
        # Visit the input arguments to create MemBlocks
        for var, sinfo in zip(func.params, func.struct_info.params):
            in_mb = MemBlock.from_struct_info(str(var), sinfo, origin="relax.input")
            self.add_memblock(name_hint, mb=in_mb)

        # Visit the function body to create MemBlocks + Dependencies
        for binding in func.body.blocks[0].bindings:
            if isinstance(binding.value, relax.Call) and binding.value.op.name == "relax.call_tir":
                sinfo = binding.value.sinfo_args[0]  # StructInfo describing the output tensor
                origin = f"relax.call_tir.{binding.value.args[0].name_hint}"
                out_mb = MemBlock.from_struct_info(str(binding.var), sinfo, origin=origin)
                self.add_memblock(name_hint, out_mb)

                # Create linked dependencies
                # for r in binding.value.args[1]:
                #     r_memblock_id = MemBlock.compute_id_from_relax_sinfo(r.struct_info, origin="relax.input")
                #     reader: MemBlock = self.id_to_memblock.get(r_memblock_id)

                #     if reader and reader._id != out_mb._id:
                #         out_mb.depends_on.append(reader)
                #         reader.links_to.append(out_mb)
                    
    def visit_tir_func(self, name_hint: str, func: tvm.tir.PrimFunc):
        """
        Parses a TIR PrimFunc using TVM's stmt_functor to visit all nodes.
        Detects:
          - tir.Allocate or tir.AllocateConst
          - tir.BufferStore targets
        Creates MemBlocks accordingly.
        """

        def find_allocations(stmt):
            if isinstance(stmt, tir.Block):
                # If there are buffer allocations (T.alloc_buffer), then add it to the list
                if len(stmt.alloc_buffers) > 0:
                    for buf in stmt.alloc_buffers:
                        mb = MemBlock.from_tir_buffer(buf.name, buf, origin="tir." + name_hint + "." + buf.name)
                        self.add_memblock(name_hint, mb)
            return True

        def dependency_retrieval(stmt):
            # Match alloc statements: these are memory-allocated local buffers
            # Doesn't seem to happen in examples at hand after all
            if isinstance(stmt, tvm.tir.Allocate) or isinstance(stmt, tvm.tir.AllocateConst):
                raise NotImplementedError("Halfly implemented, sure to be bugged. Implement it if it happens")
                shape = tuple(int(x) for x in stmt.extents)
                dtype = stmt.dtype
                mb = MemBlock(name=f"{name_hint}_alloc", shape=shape, dtype=dtype, origin=name_hint)
                self.add_memblock(name_hint, mb)
            # Walks tir.Blocks to retrieve read/write buffer relations
            elif isinstance(stmt, tir.Block):
                # Dependency modeling: reads → writes
                write_buffers = [r.buffer for r in stmt.writes]
                read_buffers = [r.buffer for r in stmt.reads]

                for w in write_buffers:
                    # We have to recompute the id of the block to get it from self.id_to_memblock
                    w_memblock_id = MemBlock.compute_id_from_buffer(w, origin="tir." + name_hint + "." + w.name)
                    if w_memblock_id not in self.id_to_memblock.keys():
                        w_memblock_id = MemBlock.compute_id_from_buffer(w, origin="relax")

                    writer = self.id_to_memblock.get(w_memblock_id)
                    if writer:
                        for r in read_buffers:
                            # We have to recompute the id of the block to get it from self.id_to_memblock
                            r_memblock_id = MemBlock.compute_id_from_buffer(r, origin="tir." + name_hint + "." + r.name)  # Internal tensor of the tir func
                            if r_memblock_id not in self.id_to_memblock.keys():
                                r_memblock_id = MemBlock.compute_id_from_buffer(r, origin="tir." + name_hint)  # New param of the tir func
                            if r_memblock_id not in self.id_to_memblock.keys():
                                r_memblock_id = MemBlock.compute_id_from_buffer(r, origin="relax")  # Comes from relax func
                            reader = self.id_to_memblock.get(r_memblock_id)

                            if reader and reader._id != writer._id:
                                writer.depends_on.append(reader)
                                reader.links_to.append(writer)


        for var, sinfo in zip(func.params, func.struct_info.params):
            mb_id = MemBlock.compute_id_from_relax_sinfo(sinfo, origin="relax")

            if not mb_id in self.id_to_memblock.keys():
                in_mb = MemBlock.from_struct_info(str(var), sinfo, origin="tir." + name_hint)
                self.add_memblock(name_hint, mb=in_mb)

        # Create allocated buffer objects
        tvm.tir.stmt_functor.pre_order_visit(func.body, find_allocations)
        # Links the buffers with dependency relations
        tvm.tir.stmt_functor.post_order_visit(func.body, dependency_retrieval)


    def print_elements(self):
        if self.memblocks:
            for func, mbs in self.memblocks.items():
                print(f"Function: {func}")
                for mb in mbs:
                    print("   ", mb)
        else:
            warnings.warn("No memblocks in alloc_finder. Did you walk() ?")



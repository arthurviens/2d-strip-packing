import unittest
import tvm
from tvm import relax
from AllocationFinder import AllocationFinder
from MemBlock import MemBlock


# === Test Network ===
class RelaxMnist(relax.frontend.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = relax.frontend.nn.Conv2D(3, 32, kernel_size=5, stride=1, padding=2, bias=True)
        self.relu1 = relax.frontend.nn.ReLU()
        self.conv2 = relax.frontend.nn.Conv2D(32, 64, kernel_size=5, stride=1, padding=2, bias=True)
        self.relu2 = relax.frontend.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class AllocationFinderTester(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        # === 1. Export TVM Module ===
        input_shape = (1, 3, 128, 128)
        rconv_mod, __ = RelaxMnist().export_tvm(
            {"forward": {"x": relax.frontend.nn.spec.Tensor(input_shape, "float32")}}
        )

        transforms = [
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FoldConstant(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ]
        new_mod = rconv_mod
        for t in transforms:
            new_mod = t(new_mod)

        self.transformed_mod = new_mod

    def test_alloc_finder_memblocks(self):
        # === 2. Walk AST ===
        alloc_finder = AllocationFinder(self.transformed_mod)
        alloc_finder.walk()

        # === 3. Check memblock counts ===
        mb = alloc_finder.memblocks
        self.assertIn("fused_conv2d_add_relu", mb)
        self.assertIn("fused_conv2d1_add1_relu1", mb)
        self.assertIn("reshape", mb)
        self.assertIn("reshape1", mb)
        self.assertIn("forward", mb)

        self.assertEqual(len(mb["fused_conv2d_add_relu"]), 4)
        self.assertEqual(len(mb["fused_conv2d1_add1_relu1"]), 4)
        self.assertEqual(len(mb["reshape"]), 1)
        self.assertEqual(len(mb["reshape1"]), 1)
        self.assertEqual(len(mb["forward"]), 9)

        # === 4. Check shape, dtype, and origin for a few specific ones ===
        fwd = {mbi.name: mbi for mbi in mb["forward"]}
        self.assertEqual(fwd["x"].shape, (1, 3, 128, 128))
        self.assertEqual(fwd["x"].dtype, "float32")
        self.assertEqual(fwd["x"].origin, "relax.input")

        self.assertEqual(fwd["gv"].origin, "relax.call_tir.fused_conv2d1_add1_relu1")
        self.assertEqual(fwd["gv"].shape, (1, 64, 128, 128))

        # === 5. Dependency modeling: gv must depend on lv and lv3 ===
        gv = fwd["gv"]
        gv_deps = {d.name for d in gv.depends_on}
        self.assertIn("lv", gv_deps)
        self.assertIn("lv3", gv_deps)

        # === 6. Transitive backward check: lv3 must
        lv3 = fwd["lv3"]
        self.assertIn(gv, lv3.links_to)


class TestMemBlock(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.mb = MemBlock(name="x", shape=(1, 3, 128, 128), dtype="float32", origin="test_origin")

    def test_basic_init(self):
        self.assertEqual(self.mb.name, "x")
        self.assertEqual(self.mb.shape, (1, 3, 128, 128))
        self.assertEqual(self.mb.dtype, "float32")
        self.assertEqual(self.mb.origin, "test_origin")
        self.assertTrue(hasattr(self.mb, "_id"))
        self.assertIsInstance(self.mb._id, str)

    def test_empty_dependencies(self):
        mb = MemBlock(name="A", shape=(4, 4), dtype="float32")
        self.assertEqual(mb.depends_on, [])
        self.assertEqual(mb.links_to, [])

    def test_consistent_hashing(self):
        id1 = MemBlock.compute_id("a", (1, 2, 3), "float32")
        id2 = MemBlock.compute_id("a", (1, 2, 3), "float32")
        self.assertEqual(id1, id2)

    def test_id_changes_with_name(self):
        id1 = MemBlock.compute_id("a", (1, 2), "float32")
        id2 = MemBlock.compute_id("b", (1, 2), "float32")
        self.assertNotEqual(id1, id2)

    def test_id_changes_with_shape(self):
        id1 = MemBlock.compute_id("a", (1, 2), "float32")
        id2 = MemBlock.compute_id("a", (1, 6), "float32")
        self.assertNotEqual(id1, id2)

    def test_id_changes_with_dtype(self):
        id1 = MemBlock.compute_id("a", (1, 2), "float32")
        id2 = MemBlock.compute_id("a", (1, 2), "float64")
        self.assertNotEqual(id1, id2)

    def test_from_buffer(self):
        from tvm.tir import decl_buffer
        buf = decl_buffer((16, 16), dtype="float32", name="B")
        mb = MemBlock.from_tir_buffer("B", buf, origin="my_tir_func")
        self.assertEqual(mb.name, "B")
        self.assertEqual(mb.shape, (16, 16))
        self.assertEqual(mb.dtype, "float32")
        self.assertEqual(mb.origin, "my_tir_func")

    def test_from_relax_input_output(self):
        var = relax.Var("x")
        sinfo = relax.TensorStructInfo((1, 3, 128, 128), dtype="float32")
        mb_out = MemBlock.from_relax_output(var, sinfo)
        mb_in = MemBlock.from_relax_input(var, sinfo)

        self.assertEqual(mb_out.name, "x")
        self.assertEqual(mb_out.shape, (1, 3, 128, 128))
        self.assertEqual(mb_out.origin, "relax.output")
        self.assertEqual(mb_in.origin, "relax.input")

    def test_size_bytes_float32(self):
        mb = MemBlock("x", (1, 3, 128, 128), "float32")
        expected = 1 * 3 * 128 * 128 * 4  # float32 = 4 bytes
        self.assertEqual(mb.size_bytes, expected)

    def test_size_bytes_int8(self):
        mb = MemBlock("y", (4, 4), "int8")
        expected = 4 * 4 * 1  # int8 = 1 byte
        self.assertEqual(mb.size_bytes, expected)

    def test_recursive_dependency_traversal(self):
        A = MemBlock("A", (1,), "float32")
        B = MemBlock("B", (1,), "float32", depends_on=[A])
        C = MemBlock("C", (1,), "float32", depends_on=[B])
        visited = []

        def patched_walk(mb):
            visited.append(mb.name)
            for d in mb.depends_on:
                patched_walk(d)

        patched_walk(C)
        self.assertEqual(visited, ["C", "B", "A"])

    def test_repr_verbose(self):
        mb = MemBlock("Z", (2, 2), "float32", origin="test")
        out = repr(mb)
        self.assertIn("Z", out)
        self.assertIn("shape=(2, 2)", out)
        self.assertIn("float32", out)
        self.assertIn("origin=test", out)


if __name__ == "__main__":
    unittest.main()
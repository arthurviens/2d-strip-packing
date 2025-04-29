from tvm import relax
from AllocationFinder import AllocationFinder

class RelaxMnist(relax.frontend.nn.Module):
    def __init__(self):
        super(RelaxMnist, self).__init__()
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


def get_relax_mnist():
    input_shape = (1, 3, 128, 128)
    rconv_mod, rconv_params = RelaxMnist().export_tvm({"forward": {"x": relax.frontend.nn.spec.Tensor(input_shape, "float32")}})
    
    transforms = [
        # # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FoldConstant(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
    ]

    new_mod = rconv_mod
    for t in transforms:
        new_mod = t(new_mod)

    return new_mod
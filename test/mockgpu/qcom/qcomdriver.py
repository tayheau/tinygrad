from tinygrad.runtime.autogen import kgsl
from test.mockgpu.driver import VirtDriver, VirtFileDesc

class KGSLFileDesc(VirtFileDesc):
    def __init__(self, fd, driver):
        super().__init__(fd)
        self.driver = driver
    def ioctl(self, fd, req, argp): return self.driver.kgsl_ioctl(req, argp)
    def mmap(self, st, sz, prot, flags, fd, off): return off

class QCOMDriver(VirtDriver):
    def __init__(self): super().__init__()
    def kgsl_ioctl(req, argp): pass

# https://www.lei.chat/posts/android-linux-gpu-drivers-internals-and-resources/
# https://projectzero.google/2020/09/attacking-qualcomm-adreno-gpu.html
# https://github.com/qualcomm-linux/kgsl
# https://docs.kernel.org/gpu/index.html
# https://lwn.net/Articles/562199/ -> kgsl
#TODO check on discord the scope since ISARenderer sovereignty as far as i know ?


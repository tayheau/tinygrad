from __future__ import annotations
import functools
from tinygrad.runtime.autogen import kgsl
from test.mockgpu.driver import VirtDriver, VirtFileDesc, VirtFile
from test.mockgpu.gpu import VirtGPU

class KGSLFileDesc(VirtFileDesc):
    def __init__(self, fd, driver):
        super().__init__(fd)
        self.driver = driver
    def ioctl(self, fd, req, argp): return self.driver.kgsl_ioctl(req, argp)
    def mmap(self, st, sz, prot, flags, fd, off): return off

def _get_ioctl_nr(ioctl:functools.partial):return ioctl.args[2]
kgsl_ioctl_info = {_get_ioctl_nr(partial): (n, partial.args[3]) for n, partial in vars(kgsl).items()
                  if n.startswith("IOCTL_KGSL_") and isinstance(partial, functools.partial)}

class QCOMDriver(VirtDriver):
    def __init__(self):
        super().__init__()
        self.next_fd = 1<<30; self.next_drawctx = 0
        self.tracked_files.append(VirtFile("/dev/kgsl-3d0", functools.partial(KGSLFileDesc, driver=self)))
        #make sense to have multiple ones given the nature of snapdragon ? 
        self.gpus = {}
        for i in range(1): self._prepare_gpu(i)
    def kgsl_ioctl(self, req, argp):
        nr = req & 0xFF
        name, struct_type = kgsl_ioctl_info[nr]
        struct = struct_type.from_address(argp)
        if nr == _get_ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY):
            if struct.type == kgsl.KGSL_PROP_DEVICE_INFO:
                kgsl.struct_kgsl_devinfo.from_address(struct.value).chip_id = 0x6030000
        elif nr == _get_ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE):
            #TODO init drawctxt
            struct.drawctxt_id = self.next_drawctx; self.next_drawctx += 1
        elif nr == _get_ioctl_nr(kgsl.IOCTL_KGSL_SETPROPERTY):
            #check which prop to handle
            if struct.type == kgsl.KGSL_PROP_PWR_CONSTRAINT: pass
            else: pass
        elif nr == _get_ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC):
            print("ALLLOOOCCC")
            print(struct)
            # #TODO 
            # pass

        # for k, v in vars(kgsl).items():
        #     if hasattr(v, "args") and v.args == req:
        #         print(k,v)

    def _alloc_fd(self): self.next_fd = (cur:=self.next_fd) + 1; return cur
    def open(self, name, flags, mode, fdcls): return fdcls.fdcls(self._alloc_fd())
    def _prepare_gpu(self, gpu_id): self.gpus[gpu_id] = QCOMGPU(gpu_id)
    #does it make sens to track multiple files ? 

#TODO to move to qcomgpu.py
class QCOMGPU(VirtGPU):
    def __init__(self, gpuid):
        super().__init__(gpuid)
    def map_range(self, vaddr, size): raise NotImplementedError()
    def unmap_range(self, vaddr, size): raise NotImplementedError()

# https://www.lei.chat/posts/android-linux-gpu-drivers-internals-and-resources/
# https://projectzero.google/2020/09/attacking-qualcomm-adreno-gpu.html
# https://github.com/qualcomm-linux/kgsl
# https://docs.kernel.org/gpu/index.html
# https://lwn.net/Articles/562199/ -> kgsl
#TODO check on discord the scope since ISARenderer sovereignty as far as i know ?


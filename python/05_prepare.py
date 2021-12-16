import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

# 雙精度的變數須轉型為單精度(Single)浮點數
a = numpy.random.randn(4,4)
a = a.astype(numpy.float32)
# 配置GPU記憶體
d_a = cuda.mem_alloc(a.nbytes)

# 複製到 GPU 上
cuda.memcpy_htod(d_a, a)

mod = SourceModule("""
  __global__ void square(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)
  
# Python 呼叫 C 程式  
func = mod.get_function("square")

# 編譯程式碼
func.prepare("P")
grid = (1, 1)
block = (4, 4, 1)
func.prepared_call(grid, block, d_a)

print("\n平方：")
print(a)
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

# 雙精度的變數須轉型為單精度(Single)浮點數
a = numpy.random.randn(4,4)

mod = SourceModule("""
  __global__ void square(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)
  
# Python 呼叫 C 程式  
func = mod.get_function("square")
func(cuda.InOut(a), block=(4,4,1))  

print("\n平方：")
print(a)

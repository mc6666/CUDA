import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include <stdio.h>

     __global__ void GPU_function()
       {
        printf("Hello PyCUDA!!!");
      }
""")
 
function = mod.get_function("GPU_function")
function(block=(1,1,1))

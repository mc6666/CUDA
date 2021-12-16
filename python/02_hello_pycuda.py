import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule(open('./c_code.cu', encoding='utf8').read())
 
function = mod.get_function("GPU_function")
function(block=(1,1,1))

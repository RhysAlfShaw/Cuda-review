#import "@preview/diatypst:0.9.1": *

#show: slides.with(
  title: "Please No more CUDA!", 
  subtitle: "Nvidia C++ CUDA fundamentals Course Review",
  date: "27.02.2026",
  authors: ("Rhys Shaw"),
  title-color: blue.darken(50%),
  ratio: 4/3, 
  layout: "medium", 
  toc: false,
  count: none, 
  footer: true,
  theme: "normal"
)

= Wins?

= Issues?

= News?

= Introduction to CUDA
== What is CUDA?

Cuda is a language created for Nvidia only (booo!) GPUs. 


It stands for Compute Unified Device Architecture. 


- Closed Source -- proprietary.
- Nvidia is an evil company.

// #image("imgs/cuda-meme.png",width:33%)

== What is CUDA?

Cuda is a language created for Nvidia only (booo!) GPUs. 


It stands for Compute Unified Device Architecture. 


- Closed Source -- proprietary.
- Nvidia is an evil company.

#image("imgs/cuda-meme.png",width:33%)


== Why would you want to Use CUDA?

- #emph("You need every bit of performance possible.")

== Why would you want to Use CUDA?

- You need every bit of performance possible. 
- #emph("You are insane.") 

== Why would you want to Use CUDA?

- You need every bit of performance possible. 
- You are insane. 
- #emph("You are being blackmailed.")

== Why would you want to Use CUDA?

- You need every bit of performance possible. 
- You are insane. 
- You are being blackmailed. 
- #emph("You are being held at gunpoint by a leatherclad Jensen Huang.")


== Why would you want to Use CUDA?

- You need every bit of performance possible. 
- You are insane. 
- You are being blackmailed. 
- #emph("You are being held at gunpoint by a leatherclad Jensen Huang.")

#image("imgs/huang.png", width:50%)


== GPU vs CPU
A car might be quicker if you’re moving four people, but a bus will probably get forty people there faster. 

#image("imgs/cpu-vs-gpu-memory.svg", width:70%)

Copying a single byte is five times faster on a CPU.
But when it comes to copying gigabytes of data, the GPU can do it ten times faster than the CPU


== The Software Stack

Rests upon CUDA runtime -- Interface with GPU.
Libraries (thrust, Cub, libcu++, cuDNN etc..) simplify programming.

#image("imgs/stack.png")


== The Course.

Exercises on implementing GPU versions of given C++ code snippest.

And detailed introducitons on alot of parts of CUDA programming.

Mostly focuses around using Thrust to make GPU implementations of CPU C++ code.

== What is Thrust?

Thrust is a CUDA c++ template library that provides a high level interface to make parallel programs easier?

Has alot of parallel functions for standard C++ functions e.g. scan, sort, reduce.

Thrust claims it will pick the most efficient CUDA implementation for your problem.


== Thrust Example (CPU)

```cpp
#include <algorithm>
#include <cstdio>
#include <vector>

int main() {
    float k = 0.5;
    float ambient_temp = 20;
    std::vector<float> temp{ 42, 24, 50 };
    auto transformation = [=] (float temp) { return temp + k * (ambient_temp - temp); };

    std::printf("step  temp[0]  temp[1]  temp[2]\n");
    for (int step = 0; step < 3; step++) {
        std::transform(temp.begin(), temp.end(), temp.begin(), transformation);
        std::printf("%d     %.2f    %.2f    %.2f\n", step, temp[0], temp[1], temp[2]);
    }
}
```
== Thrust Example (GPU)

```cpp
#include <thrust/execution_policy.h>
#include <thrust/universal_vector.h>
#include <thrust/transform.h>
#include <cstdio>

int main() {
    float k = 0.5;
    float ambient_temp = 20;
    thrust::universal_vector<float> temp{ 42, 24, 50 };
    auto transformation = [=] __host__ __device__ (float temp) { return temp + k * (ambient_temp - temp); };
    std::printf("step  temp[0]  temp[1]  temp[2]\n");
    for (int step = 0; step < 3; step++) {
        thrust::transform(thrust::device, temp.begin(), temp.end(), temp.begin(), transformation);
        std::printf("%d     %.2f    %.2f    %.2f\n", step, temp[0], temp[1], temp[2]);
    }
} 
}
```

== Pythonic Alternatives to CUDA

- **CuPy** - Numpy-like package for Nvidia GPUs. 
```Python
import cupy as cpp

a = cp.array([1,2,3]) # puts array in GPU memory
b = cp.array([3,4,5]) 
result = cp.dot(a,b) # compute done on GPU.
```

- **PyCUDA** - Direct access to the CUDA API from Python.

```python
import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

dest = numpy.zeros_like(a)
multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400,1,1), grid=(1,1))

print(dest-a*b)
```


== The Massive Power hungry GPU in the Room.

Whilst CUDA and Nvidia GPUs are fast and currently the most performative parallel hardware options at the moment, we should really be trying to move towards a more open nature to GPU programming. 

Nvidia #emph("should not") have a monopoly on GPU and Machine learning futures.

If Science starts to rely on this software then it should be open!

Open standard options of include:
- Vulkan - used for gaming and video renders.
- OpenGL - The grandparnet of Gaphic APIs.
- OpenCL - Compute orientated design? ML alternative to CUDA?


== My final thoughts.

#link("https://youtu.be/OF_5EKNX0Eg?si=3RVqLseY8SoqvlDI&t=7")


= AoB

= PuB?
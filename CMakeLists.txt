#------------------------------------------------------------------------------#
# Copyright 2023 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------#

set(CPU_SOURCES
                        src/CudaHelp.hh
  src/Driver.cc         src/Driver.hh
  src/ExportGold.cc     src/ExportGold.hh
  src/GenMesh.cc        src/GenMesh.hh
  src/Hydro.cc          src/Hydro.hh
  src/HydroBC.cc        src/HydroBC.hh
  src/InputFile.cc      src/InputFile.hh
                        src/Memory.hh
  src/Mesh.cc           src/Mesh.hh
                        src/MyLegion.hh
  src/PennantMapper.cc  src/PennantMapper.hh
  src/PolyGas.cc        src/PolyGas.hh
  src/QCS.cc            src/QCS.hh
  src/TTS.cc            src/TTS.hh
                        src/Vec2.hh
  src/WriteXY.cc        src/WriteXY.hh
  src/main.cc
)
if(Legion_USE_CUDA)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)
  set(GPU_SOURCES
    src/Hydro.cu
    src/HydroBC.cu
    src/Mesh.cu
    src/PolyGas.cu
    src/QCS.cu
    src/TTS.cu
  )
  cuda_add_executable(pennant ${CPU_SOURCES} ${GPU_SOURCES})
elseif(Legion_USE_HIP)
  set(GPU_SOURCES
    src/Hydro.cu
    src/HydroBC.cu
    src/Mesh.cu
    src/PolyGas.cu
    src/QCS.cu
    src/TTS.cu
  )
  if(Legion_HIP_TARGET STREQUAL "CUDA")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)
    cuda_add_executable(pennant ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(pennant PRIVATE __HIP_PLATFORM_NVIDIA__)
  elseif (Legion_HIP_TARGET STREQUAL "ROCM")
    hip_add_executable(pennant ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(pennant PRIVATE __HIP_PLATFORM_AMD__)
  endif()
else()
  add_executable(pennant ${CPU_SOURCES})
endif()
target_compile_definitions(pennant PRIVATE ENABLE_GATHER_COPIES)
target_compile_options(pennant PRIVATE -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-but-set-variable -Wno-unused-variable)
target_link_libraries(pennant Legion::Legion ResilienceStatic)
set_target_properties(pennant PROPERTIES CXX_STANDARD 17)
add_test(NAME pennant COMMAND ${PROJECT_SOURCE_DIR}/tests/runner.py $<TARGET_FILE:pennant> -n 4 -f ${CMAKE_CURRENT_SOURCE_DIR}/test/sedovsmall/sedovsmall.pnt -ll:cpu 4 -lg:no_tracing)

# Copyright 2013 Stanford University
# Copyright 2015 Los Alamos National Laboratory 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#setenv GCC mpicxx

SRCDIR := src

#LG_RT_DIR := ${HOME}/Documents/legion/legion-fork/runtime
#LG_RT_DIR := ${HOME}/local/src/legion/runtime
ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
#DEBUG=1                   # Include debugging symbols
DEBUG=0
OUTPUT_LEVEL=LEVEL_DEBUG  # Compile time print level
SHARED_LOWLEVEL=0
USE_CUDA=0
USE_OPENMP=0
HDF_ROOT = ${HOME}/local
USE_HDF   =0
USE_GASNET=0

# Put the binary file name here
OUTFILE         := pennant
# List all the application source files here
GEN_SRC         := $(wildcard $(SRCDIR)/*.cc)
GEN_GPU_SRC	:= $(wildcard $(SRCDIR)/*.cu) # .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:= -I$(SRCDIR) 
#CC_FLAGS	:=
CC_FLAGS	:= -std=c++11 -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-but-set-variable -Wno-unused-variable #-DLEGION_SPY
CC_FLAGS	+= -DENABLE_GATHER_COPIES #-DPRECOMPACTED_RECT_POINTS
ifeq ($(strip $(USE_OPENMP)),1)
CC_FLAGS	+= -fopenmp
endif
#CC_FLAGS	+= -DENABLE_MAX_CYCLE_PREDICATION
#CC_FLAGS	+= -DBOUNDS_CHECKS
#CC_FLAGS	+= -DLEGION_SPY
NVCC_FLAGS	:= -std=c++11
GASNET_FLAGS	:=
#LD_FLAGS	:=
LD_FLAGS := 

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################


include $(LG_RT_DIR)/runtime.mk



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
DEBUG=1
OUTPUT_LEVEL=LEVEL_DEBUG  # Compile time print level
#OUTPUT_LEVEL=LEVEL_INFO   # Compile time print level
#SHARED_LOWLEVEL=1	  # Use the shared low level
SHARED_LOWLEVEL=0
USE_CUDA=0
GASNET_ROOT= ${HOME}/local
HDF_ROOT = ${HOME}/local
#GASNET_ROOT=/users/cferenba/gasnet/1.24.0/build-mustang-gnu48
#GASNET_ROOT=/users/cferenba/gasnet/1.24.0/build-mustang-intel14
GASNET=$(GASNET_ROOT)
#CONDUIT=udp
CONDUIT=ibv
#ALT_MAPPERS=1		  # Compile the alternative mappers
USE_HDF   =0
USE_GASNET=0

# Put the binary file name here
OUTFILE         := pennant
# List all the application source files here
GEN_SRC         := $(wildcard $(SRCDIR)/*.cc)
GEN_GPU_SRC	:=				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:= -I$(SRCDIR) 
#CC_FLAGS	:=
CC_FLAGS	:= -std=c++11 -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-but-set-variable -Wno-unused-variable
#CC_FLAGS	+= -DENABLE_MAX_CYCLE_PREDICATION
#CC_FLAGS	+= -DBOUNDS_CHECKS
#CC_FLAGS	+= -DLEGION_SPY
NVCC_FLAGS	:=
GASNET_FLAGS	:=
#LD_FLAGS	:=
LD_FLAGS := 

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################


include $(LG_RT_DIR)/runtime.mk



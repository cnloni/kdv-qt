# All Target
all: ./bin/kdv-qt

# Add inputs and outputs from these tool invocations to the build variables
CU_SRCS += \
./src/KdV.cu \
./src/KdVDevice.cu

CPP_SRCS += \
./src/gb-kdv.cpp

OBJS += \
./bin/KdV.o \
./bin/KdVDevice.o \
./bin/gb-kdv.o

CU_DEPS += \
./bin/KdV.d \
./bin/KdVDevice.d

CPP_DEPS += \
./bin/gb-kdv.d


# Each subdirectory must supply rules for building sources it contributes
./bin/%.o: ./src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -I"/home/oni/git/cuda/kdv-qt/src" -I/usr/local/cuda/samples/common/inc -O3 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda/bin/nvcc -I"/home/oni/git/cuda/kdv-qt/src" -I/usr/local/cuda/samples/common/inc -O3 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

./bin/%.o: ./src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda/bin/nvcc -I"/home/oni/git/cuda/kdv-qt/src" -I/usr/local/cuda/samples/common/inc -O3 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda/bin/nvcc -I"/home/oni/git/cuda/kdv-qt/src" -I/usr/local/cuda/samples/common/inc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


OS_SUFFIX := $(subst Linux,linux,$(subst Darwin/x86_64,darwin,$(shell uname -s)/$(shell uname -m)))

bin/:
		@mkdir -p bin

# Tool invocations
./bin/kdv-qt: bin/ $(OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC Linker'
	/usr/local/cuda/bin/nvcc --cudart static --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30 -link -o "bin/kdv-qt" $(OBJS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(CC_DEPS)$(C++_DEPS)$(EXECUTABLES)$(C_UPPER_DEPS)$(CXX_DEPS)$(OBJS)$(CU_DEPS)$(CPP_DEPS)$(C_DEPS) ./bin/kdv-qt
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

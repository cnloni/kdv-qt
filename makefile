# All Target
all: ./bin/kdv-qt

CU_BASE = KdV KdVDevice
CPP_BASE = gb-kdv

DEPS = $(CU_BASE:%=./bin/%.d) $(CPP_BASE:%=./bin/%.d)
OBJS = $(CU_BASE:%=./bin/%.o) $(CPP_BASE:%=./bin/%.o)

PROCESSOR = $(shell which nvidia-smi && echo G || echo C)


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

./bin/:
	-mkdir -p $@

./bin/kdv-qt: ./bin/ $(OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC Linker'
	/usr/local/cuda/bin/nvcc --cudart static --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30 -link -o "bin/kdv-qt" $(OBJS)
	@echo 'Finished building target: $@'
	@echo ' '

TIMES = 0 1 3.6
NPYS = $(TIMES:%=./results/kdv_256_1e-05_%.npy)

test: $(NPYS)
	./results/fig1.py $^

./results/kdv_256_1e-05_0.npy: ./bin/kdv-qt
	$< -N 256 -d 1e-5 -T 0 -$(PROCESSOR)

./results/kdv_256_1e-05_1.npy: ./bin/kdv-qt
	$< -N 256 -d 1e-5 -T 1 -$(PROCESSOR)

./results/kdv_256_1e-05_3.6.npy: ./bin/kdv-qt
	$< -N 256 -d 1e-5 -T 3.6 -$(PROCESSOR)

clean:
	-$(RM) $(OBJS) $(DEPS) $(NPYS) ./bin/kdv-qt
	-@echo ' '

.PHONY: all clean test

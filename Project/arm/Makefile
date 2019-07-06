.PHONY: build clean rebuild

CFLAGS += -O3 -march=native -mtune=native --std=c++14 -Imnist/include
LDFLAGS += -lcnpy

BATCH_SIZE ?= 64
USE_OMP ?= 1
USE_OMP_FOR ?= 1
USE_OMP_SIMD ?= 1

DEFS := -DBATCH_SIZE=$(BATCH_SIZE)

ifeq ($(DEBUG), 1)
CFLAGS += -g
DEFS += -DDEBUG
endif

ifeq ($(USE_OMP), 1)
CFLAGS += -fopenmp
LDFLAGS += -fopenmp
ifeq ($(USE_OMP_FOR), 1)
DEFS += -DUSE_OMP_FOR
endif
ifeq ($(USE_OMP_SIMD), 1)
DEFS += -DUSE_OMP_SIMD
endif
endif

ifeq ($(USE_VEC), 1)
ifeq ($(USE_OMP_SIMD), 1)
$(error can only use either openmp SIMD or explicit vector instructions)
endif
DEFS += -DUSE_VEC
endif

OBJ := main.o mat.o sparse.o
SRC_DEP := $(patsubst %.o,%.d,$(OBJ))
DEP_DEP := $(patsubst %.o,%.d.d,$(OBJ))

build: main

rebuild: clean build

clean:
	rm -f main $(OBJ)
ifeq ($(MAKECMDGOALS),clean)
	rm -f $(SRC_DEP) $(DEP_DEP)
endif

main: $(OBJ)

%:
	@echo "LINK $^ ==> $@"
	@$(CXX) -o $@ $^ $(LDFLAGS)

%.d: %.cpp
	@echo "scanning $< dependencies"
	@$(CXX) $(CFLAGS) $(DEFS) -MM $< > $@
	@$(CXX) $(CFLAGS) $(DEFS) -MM -MT $@ $< > $@.d

%.o: %.cpp
	@echo "CXX $< ==> $@"
	@$(CXX) $(CFLAGS) $(DEFS) -c $<

ifneq ($(MAKECMDGOALS),clean)
-include $(SRC_DEP) $(DEP_DEP)
endif

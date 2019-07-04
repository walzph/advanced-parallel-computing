.PHONY: build clean rebuild run

CFLAGS += -O3 -march=native -mtune=native -fopenmp -Wall --std=c++14 -Imnist/include
#CFLAGS += -O0 -fopenmp -Wall --std=c++14 -Imnist/include
LDFLAGS += -fopenmp -lcnpy

OBJ := test.o mat.o sparse.o
SRC_DEP := $(patsubst %.o,%.d,$(OBJ))
DEP_DEP := $(patsubst %.o,%.d.d,$(OBJ))

build: test

rebuild: clean build

clean:
	rm -f main main_avx test $(OBJ)
ifeq ($(MAKECMDGOALS),clean)
	rm -f $(SRC_DEP) $(DEP_DEP)
endif

test: $(OBJ)

%:
	@echo "LINK $^ ==> $@"
	@$(CXX) -o $@ $^ $(LDFLAGS)

%.d: %.cpp
	@echo "scanning $< dependencies"
	@$(CXX) $(CFLAGS) -MM $< > $@
	@$(CXX) $(CFLAGS) -MM -MT $@ $< > $@.d

%.o: %.cpp
	@echo "CXX $< ==> $@"
	@$(CXX) $(CFLAGS) -c $<

ifneq ($(MAKECMDGOALS),clean)
-include $(SRC_DEP) $(DEP_DEP)
endif

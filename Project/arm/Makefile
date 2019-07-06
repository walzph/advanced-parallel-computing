.PHONY: build clean rebuild run

CFLAGS += -O3 -march=native -mtune=native -fopenmp --std=c++14 -Imnist/include
LDFLAGS += -fopenmp -lcnpy

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
	@$(CXX) $(CFLAGS) -MM $< > $@
	@$(CXX) $(CFLAGS) -MM -MT $@ $< > $@.d

%.o: %.cpp
	@echo "CXX $< ==> $@"
	@$(CXX) $(CFLAGS) -c $<

ifneq ($(MAKECMDGOALS),clean)
-include $(SRC_DEP) $(DEP_DEP)
endif

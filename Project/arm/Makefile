.PHONY: build clean rebuild run

CFLAGS += -Wall --std=c++14 -Imnist/include
LDFLAGS += -lcnpy

OBJ := test.o mat.o sparse.o
DEP := $(patsubst %.o,%.d,$(OBJ))

#build: main main_avx test
build: test

rebuild: clean build

clean:
	rm -f main main_avx test $(OBJ)
ifeq ($(MAKECMDGOALS),clean)
	rm -f $(DEP)
endif

run: build
#	./main_avx
#	./main
	./test

main_avx: main.cpp sparse_ops.h
	$(CXX) $(CFLAGS) $(LDFLAGS) -mavx2 -mfma -o $@ $<

main: main.cpp mat.o sparse.o
	$(CXX) $(CFLAGS) $(LDFLAGS) -D__NEW_SPARSE__ -o $@ $^

test: $(OBJ)

%:
	@echo "LINK $^ ==> $@"
	@$(CXX) $(LDFLAGS) -o $@ $^

%.d: %.cpp
	@echo "scanning $< dependencies"
	@$(CXX) $(CFLAGS) -MM $< > $@

%.o: %.cpp
	@echo "CXX $< ==> $@"
	@$(CXX) $(CFLAGS) -c $<

ifneq ($(MAKECMDGOALS),clean)
-include $(DEP)
endif

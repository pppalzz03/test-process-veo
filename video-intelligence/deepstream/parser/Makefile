# Build inside the DeepStream container:  make -C /opt/vi/parser
DS_ROOT   ?= /opt/nvidia/deepstream/deepstream
CUDA_ROOT ?= /usr/local/cuda
CXX       ?= g++
LIB       := libnvdsinfer_custom_impl_ultralytics.so

$(LIB): nvdsparsebbox_ultralytics.cpp yolo_decode.h
	$(CXX) -std=c++17 -O2 -fPIC -shared -Wall -o $@ nvdsparsebbox_ultralytics.cpp \
	  -I$(DS_ROOT)/sources/includes -I$(CUDA_ROOT)/include

# Host-side unit test of the decode logic (no DeepStream needed).
test: test_decode.cpp yolo_decode.h
	$(CXX) -std=c++17 -Wall -o test_decode test_decode.cpp && ./test_decode

clean:
	rm -f $(LIB) test_decode

.PHONY: test clean

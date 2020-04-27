CC=g++
CXXFLAGS= -I. -L. -std=c++11
OBJ = face_ssd.o main_tflite.o 

OPENCV_LIBS = $(shell pkg-config --libs opencv)
OPENCV_LIBPATH := /usr/loacl/lib
OPENCV_INC    := /usr/loacl/include


all: example

example: $(OBJ)
	$(CC) -o $@ $^ $(CXXFLAGS) -Wl,-rpath=./ dl_interface.so \
	-Wl,-rpath,$(OPENCV_LIBPATH) \
	-L$(OPENCV_LIBPATH) $(OPENCV_LIBS)
	
%.o: %.cpp 
	$(CC) -c -o $@ $< $(CXXFLAGS) \
	$(patsubst %,-I%,$(OPENCV_INC) ) \
	$(OPENCV_LIBS) 

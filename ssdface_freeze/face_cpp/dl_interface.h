#ifndef dl_interface_h
#define dl_interface_h
void* dl_new(const char* modelPath);
bool dl_run(void* ptr, float *pDataIn, int size_i, float *pDataOut, int size_o);
bool dl_delete(void* ptr);


#endif

arch = sm_86

main: clean
	nvcc -o cublas_FP16.x -lcublas -arch=${arch} cublas_FP16.cu
	nvcc -o cublas_FP32.x  -lcublas -arch=${arch} cublas_FP32.cu
	nvcc -o cublas_TF32.x  -lcublas -arch=${arch} cublas_TF32.cu
	nvcc -o tensor_custom.x -arch=${arch} tensor_custom.cu

clean:
	rm -f *.x *.o
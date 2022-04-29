#include <stdio.h>
#include <sys/time.h>

#include <cuda.h>
#include <cublas_v2.h>

#define valueA 1.57f
#define valueB 5.36f

static struct timeval tv0;
double get_time()
{
	double t;
	gettimeofday(&tv0, (struct timezone*)0);
	t = ((tv0.tv_usec) + (tv0.tv_sec)*1000000);

	return (t);
}

int main(int argc, char* argv[]) {
    
    double t0, t1;
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);

    float *a, *a_GPU;
    float *b, *b_GPU;
    float *c, *c_GPU;

    int N;

    if (argc != 2) {
        printf("Usage ./exec N\n");
        return 1;
    }

    else N = atoi(argv[1]);

    // Mallocs
    a = (float*) malloc(N*N*sizeof(float));
    cudaMalloc((void**) &a_GPU, N*N*sizeof(float));
  
    b = (float*) malloc(N*N*sizeof(float));
    cudaMalloc((void**) &b_GPU, N*N*sizeof(float));

    c = (float*) malloc(N*N*sizeof(float));
    cudaMalloc((void**) &c_GPU, N*N*sizeof(float));

    // Inicializar a y b
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            a[i*N + j] = valueA;
            b[i*N + j] = valueB;
        }

    status = cublasSetMatrix(N, N, sizeof(float), a, N, a_GPU, N);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("Data download failed 1\n");
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    status = cublasSetMatrix(N, N, sizeof(float), b, N, b_GPU, N);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("Data download failed 2\n");
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    float mult = 1.0f;

    status = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    // ---------------- Calentamiento ----------------------
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &mult,
                            a_GPU, N,
                            b_GPU, N,
                            &mult, c_GPU, N);
    cudaDeviceSynchronize();
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("Mult failed\n");
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    	
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed \n");
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    // ----------------------------------------------------------

    free(c);
    cudaFree(c_GPU);
    c = (float*) malloc(N*N*sizeof(float));
    cudaMalloc((void**) &c_GPU, N*N*sizeof(float));

    
    t0 = get_time();	
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &mult,
                            a_GPU, N,
                            b_GPU, N,
                            &mult, c_GPU, N);

    cudaDeviceSynchronize();
    t1 = get_time();
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("Mult failed\n");
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    	
    status = cublasGetMatrix(N, N, sizeof(float), c_GPU, N, c, N);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed \n");
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    printf("Tiempo ejecución: %fms.\n", (t1 - t0) / 1000.0f);
    printf("Error: %f%%\n", fabs(valueA * valueB * N - c[0]) / (valueA * valueB * N));
    // ----------------------------------------------------------
    
    free(a); free(b); free(c);
    cudaFree(a_GPU); cudaFree(b_GPU); cudaFree(c_GPU);

    return 0;
}
#include <stdio.h>
#include <mma.h>
#include <sys/time.h>

using namespace nvcuda;

const int TILE_SIZE = 16;   // Cada warp calcula sub-matriz de TILE_SIZE x TILE_SIZE

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

__global__ void matrix_mul(half *a, half *b, half *c, int N) {

    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, half> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int i = 0; i < N; i += TILE_SIZE) {

        int aRow = warpM * TILE_SIZE;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * TILE_SIZE;

        if (aRow < N && aCol < N && bRow < N && bCol < N) {

            wmma::load_matrix_sync(a_frag, a + aRow * N + aCol, N);       
            wmma::load_matrix_sync(b_frag, b + bRow * N + bCol, N);
        }
        
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    int cRow = warpM * TILE_SIZE;
    int cCol = warpN * TILE_SIZE;

    if (cRow < N && cCol < N) 
        wmma::store_matrix_sync(c + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
}

int main(int argc, char* argv[]) {

    double t0, t1;
    half *a, *b;
    half *c;

    half *a_GPU, *b_GPU;
    half *c_GPU;

    int N;
    if (argc != 2) {
        printf("Usage ./exec N\n");
        return 1;
    }

    else N = atoi(argv[1]);

    if (N % 16 != 0) {
        printf("N must be a multiple of 16\n");
        return 1;
    }

    a = (half*) malloc(N*N*sizeof(half));
    b = (half*) malloc(N*N*sizeof(half));
    c = (half*) malloc(N*N*sizeof(half));

    for (int i = 0; i < N; i++) 
        for (int j = 0; j < N; j++) {
            a[i*N + j] = valueA;
            b[i*N + j] = valueB;
        }
            
    cudaMalloc((void**) &a_GPU, N*N*sizeof(half));
    cudaMalloc((void**) &b_GPU, N*N*sizeof(half));
    cudaMalloc((void**) &c_GPU, N*N*sizeof(half));
    
    cudaMemcpy(a_GPU, a, N*N*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b_GPU, b, N*N*sizeof(half), cudaMemcpyHostToDevice);

    dim3 block_size(256, 2); // 128 * 2 = 128 hilos = 4 warps => Cada bloque calcula sub-matriz de 32x32
    dim3 grid_size(N/128, N/32);

    matrix_mul<<< grid_size, block_size >>>(a_GPU, b_GPU, c_GPU, N);
    cudaDeviceSynchronize();

    t0 = get_time();
    matrix_mul<<< grid_size, block_size >>>(a_GPU, b_GPU, c_GPU, N);
    cudaDeviceSynchronize();
    t1 = get_time();

    cudaMemcpy(c, c_GPU, N*N*sizeof(half), cudaMemcpyDeviceToHost);
    

    printf("Tiempo ejecución: %fms.\n", (t1 - t0) / 1000.0f);
    printf("Error: %f%%\n", fabs(valueA * valueB * N - __half2float(c[0])) / (valueA * valueB * N));


    free(a); free(b); free(c);
    cudaFree(a_GPU); cudaFree(b_GPU); cudaFree(c_GPU);

    return 0;
}
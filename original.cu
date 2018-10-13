#include "original.h"
#include <chrono>

int main(int argc, char ** argv) {

    ///  The A and B matrix
    int * A;
    int * B; 

    ///  The output C matrix
    int * TiledGPUC;
    int * UntiledGPUC; 
    int * CPUC;

	/// Matrix in GPU
    int * GPUA;
    int * GPUB;
    int * GPUC;

    ///  Allocate Inputs
    A = (int *) malloc(sizeof(int)*N*N);
    B = (int *) malloc(sizeof(int)*N*N);

    ///  Allocate Results
    TiledGPUC = (int *) malloc(sizeof(int)*N*N);
    UntiledGPUC = (int *) malloc(sizeof(int)*N*N);
    CPUC = (int *) malloc(sizeof(int)*N*N);

    srand(time(NULL));
    ///  Matrix Initialization
    #pragma omp parallel for default(none) shared(N,A,B)
    for (int i = 0; i < N*N; i++)
    {
        A[i]=rand() % 5;
        B[i]=rand() % 5;
    }

    ///  Allocating GPU memory
    cudaMalloc((void **)&GPUA, sizeof(int)*N*N);
    cudaMalloc((void **)&GPUB, sizeof(int)*N*N);
    cudaMalloc((void **)&GPUC, sizeof(int)*N*N);

    ///  Copy memory to the GPU
    cudaMemcpy(GPUA, A, sizeof(int)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(GPUB, B, sizeof(int)*N*N, cudaMemcpyHostToDevice);

    ///  Initialize the grid and block dimensions
    ///  Number of Blocks required
    dim3 Grid((N/Tile) + 1, (N/Tile) + 1, 1);
    /// Number of threads in each block
    dim3 Block(Tile, Tile, 1);

    // Initialize timer
    auto start = std::chrono::high_resolution_clock::now();

    /// Launch the GPU Tiled Kernel
    matMultiplyTiled<<<Grid, Block>>>(N,GPUA, GPUB, GPUC);

    // Finish timer
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration_ms = end - start;
    printf("Tiled GPU Test, N: %d duration %f ms\n",N,duration_ms.count());
    fflush(stdout); 


    cudaDeviceSynchronize(); 
    ///  Copy the results in GPU memory back to the CPU
    cudaMemcpy(TiledGPUC, GPUC, sizeof(int)*N*N, cudaMemcpyDeviceToHost);

    //reset GPUC
    cudaFree(GPUC);
    cudaMalloc((void **)&GPUC, sizeof(int)*N*N);

    // Initialize timer
    start = std::chrono::high_resolution_clock::now();

    /// Launch the GPU UnTiled Kernel
    matMultiplyGPU<<<Grid, Block>>>(N,GPUA, GPUB, GPUC);

    // Finish timer
    end = std::chrono::high_resolution_clock::now();

    duration_ms = end - start;
    printf("Untiled GPU Test, N: %d duration %f ms\n",N,duration_ms.count());
    fflush(stdout); 

    cudaDeviceSynchronize(); 
    ///  Copy the results in GPU memory back to the CPU
    cudaMemcpy(UntiledGPUC, GPUC, sizeof(int)*N*N, cudaMemcpyDeviceToHost);

    // Initialize timer
    start = std::chrono::high_resolution_clock::now();

    /// CPU mat Multiply
    matMultiplyCPUOMP(N, A, B, CPUC);

    // Finish timer
    end = std::chrono::high_resolution_clock::now();

    duration_ms = end - start;
    printf("CPU Test, N: %d duration %f ms\n",N,duration_ms.count());
    fflush(stdout); 

    /// Verify both
    printf("1 ");
    fflush(stdout); 
    printf("Untiled GPU vs CPU %s \n", 
        verifyMatrix(N,UntiledGPUC,CPUC) ? "true" : "false");
    printf("2 ");
    fflush(stdout); 
    printf("Tiled GPU vs CPU %s \n",  
        verifyMatrix(N,TiledGPUC,CPUC) ? "true" : "false");

    ///  Free the GPU memory
    cudaFree(GPUA);
    cudaFree(GPUB);
    cudaFree(GPUC);

    /// Free the Pointer Memory
    free(A);
    free(B);
    free(TiledGPUC);
    free(UntiledGPUC);
    free(CPUC);

    return 0;
}

bool verifyMatrix(int N, int * C1,int * C2){
    for (int i=0; i < N*N; i++) {
        if (C1[i]  != C2[i] ) {
            printf("%d %d %d \n",i,C1[i],C2[i]);
            return false; 
        }
    }
    return true;
}

void matMultiplyCPUOMP(int N,int*a,int*b,int*c){
    int result; 
    #pragma omp parallel for default(none) shared(N,a,b,c) private(result)
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            result= 0;
            for(int n=0;n < N;n++){
                result+=a[i*N+n]*b[n*N+j]; 
            }
            c[i*N+j]=result;
        }
    }
}

__global__
void matMultiplyGPU(int N, int * a, int * b, int * c){
    // We get the index of the current data 
    unsigned int threadx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int thready = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int threadxy = thready * N + threadx;

    // Then we get the col and row
    int row = threadxy / N;
    int col = threadxy % N;

    if(row < N && col < N){

        // Then we multiply and add each one of them
        int result = 0;
        for(int i=0;i<N;i++){
            result +=a[row*N+i]*b[i*N+col];
        }

        c[row*N + col]=result;

    }
}

__global__
void matMultiplyTiled(int N, int * a, int * b, int * c){
    ///  Tile size to store elements in shared memory
    __shared__ int sharedA[Tile][Tile];   
    __shared__ int sharedB[Tile][Tile];

    /// To generate ids of threads.
    int Row = blockDim.y*blockIdx.y + threadIdx.y; 
    int Col = blockDim.x*blockIdx.x + threadIdx.x;

    int cvalue = 0.0;

    sharedA[threadIdx.y][threadIdx.x] = 0.0;
    sharedB[threadIdx.y][threadIdx.x] = 0.0;

    /// copy into shared and then calculate 
    for (int k = 0; k < (((N - 1)/ Tile) + 1); k++){
		/// copy Data to Tile from Matrix (Global Memory to Shared Memory)
		if ( (Row < N) && (threadIdx.x + (k*Tile)) < N) {
			sharedA[threadIdx.y][threadIdx.x] =
				 a[(N*N) + threadIdx.x + (k*Tile)];
        }
        /// due to the matrix not always being a multiple
        else
        {
            sharedA[threadIdx.y][threadIdx.x] = 0.0;
        }

        /// copy Data to Tile from Matrix (Global Memory to Shared Memory)
        if ( Col < N && (threadIdx.y + k*Tile) < N) {
            sharedB[threadIdx.y][threadIdx.x] =
				b[(threadIdx.y + k*Tile)*N + Col];
        }
        /// due to the matrix not always being a multiple
        else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0;
        }

        /// Wait for all partials to be calculated
        __syncthreads();

        /// Multiplying Elements present in tile
        for (int j = 0; j < Tile; ++j)
        {
            cvalue += sharedA[threadIdx.y][j] * sharedB[j][threadIdx.x];
        }
    }

    /// Saving Final result into Matrix c
    if (Row < N && Col < N) {
        c[Row*N + Col] = cvalue;
    }
}

/** Tiled Matrix multiplication program
 * @author Jose Enrique Estremadoyro A01018990 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

// Tile size
#define Tile 2 

/** dimension to be squared for the matrix */
int N = 200;   

/** Verify that two matrix have the same values */
bool verifyMatrix(int N, int * C1,int * C2);

/** Normal CPU Matrix Multiplication */
void matMultiplyCPUOMP(int N, int * a, int * b, int * c);

/** Tiles GPU Matrix Multiplication - Kernel */
__global__ 
void matMultiplyTiled(int N, int * a, int * b, int * c);

/** GPU Matrix Multiplication - Kernel */
__global__
void matMultiplyGPU(int N, int * a, int * b, int * c);

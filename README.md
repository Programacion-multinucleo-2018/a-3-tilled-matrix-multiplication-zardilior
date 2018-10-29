# Assignment 3: Tilled Matrix Multiplication

Assignment No 3 for the multi-core programming course. Modify previous matrix multiplication kernels to integrate a tilled multiplication using shared memory.

The program has to do the following:

1. Multiply 2 NxN matrices. N has to be set to 2000. Perform the multiplication with and without tilling.
2. Fill the matrices with random floats between 1 and 10.
3. Validate that the result from the matrix multiplication in GPU with a CPU version. The CPU version does not have to be tilled.
4. Compare the processing time of the matrix multiplication in GPU with and without tilling, and report the speedup obtained.

Execute the kernel at least 20 times, and measure average time spent for calculating the matrix multiplication, and report both the processing times and the speedups within the readme. Test performance varying the number of threads, and the tile window. Test with the following sizes: 8x8, 16x16, 32x32.

Rubric:

1. Matrices are properly initialized.
2. Matrices are properly multiplied in GPU, and the result is validated in CPU.
3. GPU code is initialized correctly, and the device memory is deallocated.
4. Implement matrix multiplication using shared memory and tiling.
5. Report the average processing time and speedup for the different tile sizes.

**Grade: 100**
******
Results for 
N 2000
Tile Size 8
Tiled GPU Test, N: 2000 average duration 0.003995 ms
Untiled GPU Test, N: 2000 average duration 0.003324 ms
CPU Test, N: 2000 duration 63208.203125 ms
1 Untiled GPU vs CPU true 
2 Tiled GPU vs CPU true 
******
Results for 
N 2000
Tile Size 16
Tiled GPU Test, N: 2000 average duration 0.003536 ms
Untiled GPU Test, N: 2000 average duration 0.003194 ms
CPU Test, N: 2000 duration 47384.800781 ms
1 Untiled GPU vs CPU true 
2 0 8112 8072 
Tiled GPU vs CPU false 
Results for 
******
N 2000
Tile Size 32
Tiled GPU Test, N: 2000 average duration 0.003414 ms
Untiled GPU Test, N: 2000 average duration 0.003310 ms
CPU Test, N: 2000 duration 36977.363281 ms
1 Untiled GPU vs CPU true 
2 0 8316 8325 
Tiled GPU vs CPU false 

## Analisis
COmo podemos ver los tiempos son muy similares siendo el tiled siempre menor. Dado que se lleva una operacion compleja en el codigo del tiled, se checa si el tile cabe en lo que queda de la matriz original, cosa que es solo util cuando las dimensiones no son multiplos del tile, esto podria estar alentando mucho el codigo. Tambien una de las optimizaciones en ejercicios anteriores se daba con reducir el numero de bloques y aumentar el de threads, con tan pocos threads es un problema. Por otro lado no sabemos si por debajo ya se utilice algun tipo de caching dentro de cuda.

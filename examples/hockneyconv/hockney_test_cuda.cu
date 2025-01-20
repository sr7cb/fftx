#include <iostream>
#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#undef __CUDACC__
#include "interface.hpp"
#include "hockneyconvObj.hpp"
// #include "spiral_generated_hockney.cu"

static void checkOutputBuffers( double *spiral_Y, double *devfft_Y, long arrsz )
{
    bool correct = true;
    double maxdelta = 0.0;

    for ( int indx = 0; indx < arrsz; indx++ ) {
        double s = spiral_Y[indx];
        double c = devfft_Y[indx];

        double deltar = abs ( s - c );
        bool   elem_correct = ( deltar < 1e-7 );
        maxdelta = maxdelta < deltar ? deltar : maxdelta ;
        correct &= elem_correct;
    }
    
    printf ( "Correct: %s\tMax delta = %E\n", (correct ? "True" : "False"), maxdelta );
    fflush ( stdout );

    return;
}

template <typename T>
void print(T input) {
    for(int i = 0; i < 10; i++)
        std::cout << input[i] << std::endl;
}

__global__ void complexMultiplyKernel(double* a, double* b, double* result, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        double realA = a[2 * tid];
        double imagA = a[2 * tid + 1];
        double realB = b[2 * tid];
        double imagB = b[2 * tid + 1];

        result[2 * tid] = realA * realB - imagA * imagB;  // Real part of the result
        result[2 * tid + 1] = realA * imagB + imagA * realB; // Imaginary part of the result

    }
}

void buildInput(std::vector<double>& input) {
   for(int i = 0; i < input.size(); i++) {
        input[i] = 1;
   }
}

void buildInput(std::vector<std::complex<double>>& input) {
    for(int i = 0; i < input.size(); i++) {
        input[i] = std::complex<double>(1,0);
    }
}


__global__ void zeroPadKernel(double* d_output, const double* d_input, 
                              int x_old, int y_old, int z_old, 
                              int x, int y, int z) {
    // Calculate thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Ensure within bounds of the smaller cube
    if (i < x_old && j < y_old && k < z_old) {
        // Calculate the index in the smaller input array
        int oldIndex = i * y_old * z_old + j * z_old + k;
        
        // Calculate the index in the larger output array
        int newIndex = i * y * z + j * z + k;

        // Copy the value from input to output
        d_output[newIndex] = d_input[oldIndex];
    }
}


void zeroPad(double* d_output, const std::vector<double> input, 
             int x_old, int y_old, int z_old, int x, int y, int z) {
    // Step 1: Initialize the larger cube with zeros on the GPU
    cudaMemset(d_output, 0, x * y * z * sizeof(double));
    double *d_input;
    cudaMalloc(&d_input, x_old*y_old*z_old*sizeof(double));
    cudaMemcpy(d_input, input.data(), x_old*y_old*z_old*sizeof(double), cudaMemcpyHostToDevice);

    // Step 2: Define block dimensions for 256 threads per block
    dim3 blockDim(8, 8, 4);

    // Step 3: Calculate grid dimensions
    dim3 gridDim((x_old + blockDim.x - 1) / blockDim.x,
                 (y_old + blockDim.y - 1) / blockDim.y,
                 (z_old + blockDim.z - 1) / blockDim.z);

    // Step 4: Launch the kernel
    zeroPadKernel<<<gridDim, blockDim>>>(d_output, d_input, x_old, y_old, z_old, x, y, z);
    cudaFree(d_input);
}

__global__ void extractKernel(double *d_input, double *d_output, 
                              int x, int y, int z, 
                              int x_small, int y_small, int z_small) {
    // Calculate thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if within bounds of the smaller cube
    if (i < x_small && j < y_small && k < z_small) {
        // Calculate the index in the larger input vector
        int inputIndex = i * y * z + j * z + k;
        
        // Calculate the index in the smaller output vector
        int outputIndex = i * y_small * z_small + j * z_small + k;

        // Copy the value from input to output
        d_output[outputIndex] = d_input[inputIndex];
    }
}
void extract(std::vector<double>& output, double *d_input, 
             int x, int y, int z, int x_small, int y_small, int z_small) {
    // Resize the output vector to hold the smaller cube
    output.resize(x_small * y_small * z_small);

    // Allocate memory for the smaller cube on the device
    double *d_output;
    cudaMalloc(&d_output, x_small * y_small * z_small * sizeof(double));

    // Define block dimensions for 256 threads per block
    dim3 blockDim(8, 8, 4);

    // Calculate grid dimensions
    dim3 gridDim((x_small + blockDim.x - 1) / blockDim.x,
                 (y_small + blockDim.y - 1) / blockDim.y,
                 (z_small + blockDim.z - 1) / blockDim.z);

    // Launch the kernel with the updated configuration
    extractKernel<<<gridDim, blockDim>>>(d_input, d_output, x, y, z, x_small, y_small, z_small);

    // Copy the extracted smaller cube back to the host
    cudaMemcpy(output.data(), d_output, x_small * y_small * z_small * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_output);
}

int main() {
    int nx = 32;
    int ny = 32;
    int nz = 128;
    int Nx = nx * 2;
    int Ny = ny * 2;
    int Nz = nz * 2;
    int mx = 32;
    int my = 32;
    int mz = 128;

    std::vector<double> input(nx * ny * nz);
    std::vector<double> output(mx * my * mz);
    std::vector<double> spiral_output(mx * my * mz);
    std::vector<std::complex<double>> input2(Nx * Ny * ((Nz/2)+1));

    double* d_extended_input;
    double* d_out;
    double* d_temp;
    double* d_out2;

    cudaMalloc(&d_extended_input, Nx * Ny * Nz * sizeof(double));
    cudaMalloc(&d_out, 2* Nx * Ny * ((Nz/2)+1) * sizeof(double));
    cudaMalloc(&d_temp, 2* Nx * Ny * ((Nz/2)+1) * sizeof(double));
    cudaMalloc(&d_out2, Nx*Ny*Nz * sizeof(double));

    buildInput(input);
    buildInput(input2);

    // Copy `input` to `d_extended_input`
    zeroPad(d_extended_input, input, nx, ny, nz, Nx, Ny, Nz);

    // Create cuFFT plan for forward FFT (real-to-complex)
    cufftHandle plan;
    cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_D2Z);
    cufftExecD2Z(plan, (cufftDoubleReal *)d_extended_input, (cufftDoubleComplex *)d_out);

    // Copy `input2` to `d_temp` for element-wise complex multiplication
    cudaMemcpy(d_temp, input2.data(), 2 * Nx * Ny * ((Nz/2)+1) * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel for element-wise complex multiplication
    int blockSize = 256;
    int gridSize = (Nx * Ny * ((Nz/2)+1) + blockSize - 1) / blockSize;
    complexMultiplyKernel<<<gridSize, blockSize>>>(d_out, d_temp, d_temp, Nx * Ny * ((Nz/2)+1));

    cufftHandle plan2;
    cufftPlan3d(&plan2, Nx, Ny, Nz, CUFFT_Z2D);
    cufftExecZ2D(plan2, (cufftDoubleComplex *)d_temp, (cufftDoubleReal *)d_out2);
    double * h_out2 = new double[Nx*Ny*Nz];
    cudaMemcpy(h_out2, d_out2, Nx*Ny*Nz*sizeof(double), cudaMemcpyDeviceToHost);
    
    // Copy the result back to the host
    extract(output, d_out2, Nx, Ny, Nz, mx, my, mz);
    
    double *d_input, *d_sym, *d_output;
    double* check;
    cudaMallocManaged(&check, 2*Nx*Ny*(Nz/2+1) *sizeof(double));
    cudaMalloc(&d_input, nx*ny*nz*sizeof(double));
    cudaMalloc(&d_sym, 2* Nx * Ny * ((Nz/2)+1)*sizeof(double));
    cudaMalloc(&d_output, mx*my*mz*sizeof(double));
    cudaMemcpy(d_input, input.data(), nx*ny*nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sym, input2.data(), 2* Nx * Ny * ((Nz/2)+1)*sizeof(double), cudaMemcpyHostToDevice);
    // init_hockney();
    // hockney(d_output, d_input, d_sym, check);
    // destroy_hockney();
    

    std::vector<void*>args{&d_output, &d_input, &d_sym};
    std::vector<int> sizes{nx, ny, nz};
    HOCKNEYCONVProblem hcp(args, sizes, "hockney");
    hcp.transform();

    // for (int i = 0; i < 10; i++) {
    //     std::cout << output[i] << "\t";
    //     std::cout << spiral_output[i] << std::endl;
    // }

    cudaMemcpy(spiral_output.data(), d_output, mx*my*mz*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkOutputBuffers(output.data(), spiral_output.data(), mx*my*mz);

    // Clean up
    cudaFree(d_extended_input);
    cudaFree(d_out);
    cudaFree(d_temp);
    cudaFree(d_out2);
    cufftDestroy(plan);
    cufftDestroy(plan2);

    return 0;
}
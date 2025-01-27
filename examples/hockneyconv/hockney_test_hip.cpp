#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <complex>
#include <hip/hip_runtime.h>
#include <rocfft/rocfft.h>
#include <hipfft/hipfft.h>
#include "interface.hpp"
#include "hockneyconvObj.hpp"

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
    std::cout << std::endl;
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
        input[i] = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
   }
}

void buildInput(std::vector<std::complex<double>>& input) {
    for(int i = 0; i < input.size(); i++) {
        double real = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
        double imag = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
        input[i] = std::complex<double>(real, imag);
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
    hipMemset(d_output, 0, x * y * z * sizeof(double));
    double *d_input;
    hipMalloc(&d_input, x_old*y_old*z_old*sizeof(double));
    hipMemcpy(d_input, input.data(), x_old*y_old*z_old*sizeof(double), hipMemcpyHostToDevice);

    // Step 2: Define block dimensions for 256 threads per block
    dim3 blockDim(8, 8, 4);

    // Step 3: Calculate grid dimensions
    dim3 gridDim((x_old + blockDim.x - 1) / blockDim.x,
                 (y_old + blockDim.y - 1) / blockDim.y,
                 (z_old + blockDim.z - 1) / blockDim.z);

    // Step 4: Launch the kernel
    hipLaunchKernelGGL(zeroPadKernel, gridDim, blockDim, 0, 0, d_output, d_input, x_old, y_old, z_old, x, y, z);
    hipFree(d_input);
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
    hipMalloc(&d_output, x_small * y_small * z_small * sizeof(double));

    // Define block dimensions for 256 threads per block
    dim3 blockDim(8, 8, 4);

    // Calculate grid dimensions
    dim3 gridDim((x_small + blockDim.x - 1) / blockDim.x,
                 (y_small + blockDim.y - 1) / blockDim.y,
                 (z_small + blockDim.z - 1) / blockDim.z);

    // Launch the kernel with the updated configuration
    hipLaunchKernelGGL(extractKernel, gridDim, blockDim, 0, 0, d_input, d_output, x, y, z, x_small, y_small, z_small);

    // Copy the extracted smaller cube back to the host
    hipMemcpy(output.data(), d_output, x_small * y_small * z_small * sizeof(double), 
    hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_output);
}

int main() {
    std::srand(std::time(nullptr));
    int nx = 32;
    int ny = 32;
    int nz = 128;
    int Nx = nx * 2;
    int Ny = ny * 2;
    int Nz = nz * 2;
    int mx = 32;
    int my = 32;
    int mz = 128;

    /*Variable Setup*/

    std::vector<double> input(nx * ny * nz);
    std::vector<double> output(mx * my * mz);
    std::vector<double> spiral_output(mx * my * mz);
    std::vector<std::complex<double>> input2(Nx * Ny * ((Nz/2)+1));

    double* d_extended_input;
    double* d_out;
    double* d_temp;
    double* d_out2;

    hipMalloc(&d_extended_input, Nx * Ny * Nz * sizeof(double));
    hipMalloc(&d_out, 2* Nx * Ny * ((Nz/2)+1) * sizeof(double));
    hipMalloc(&d_temp, 2* Nx * Ny * ((Nz/2)+1) * sizeof(double));
    hipMalloc(&d_out2, Nx*Ny*Nz * sizeof(double));

    hipMemset(d_extended_input, 0, Nx * Ny * Nz * sizeof(double));
    double *d_input;
    hipMalloc(&d_input, nx*ny*nz*sizeof(double));

    double *d_output;
    hipMalloc(&d_output, mx*my*mz*sizeof(double));

    dim3 blockDim(8, 8, 4);

    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 (nz + blockDim.z - 1) / blockDim.z);

    dim3 gridDimout((mx + blockDim.x - 1) / blockDim.x,
                 (my + blockDim.y - 1) / blockDim.y,
                 (mz + blockDim.z - 1) / blockDim.z);

    buildInput(input);
    buildInput(input2);

    hipMemcpy(d_input, input.data(), nx*ny*nz*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_temp, input2.data(), 2 * Nx * Ny * ((Nz/2)+1) * sizeof(double), hipMemcpyHostToDevice);
 
    rocfft_plan forward_plan;
    size_t work_size = 0;
    size_t lengths[3] = {static_cast<size_t>(Nz), static_cast<size_t>(Ny), static_cast<size_t>(Nx)};
    rocfft_plan_create(&forward_plan, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_double,
                       3, lengths, 1, nullptr);

    rocfft_plan_get_work_buffer_size(forward_plan, &work_size);
    void* work_buffer;
    hipMalloc(&work_buffer, work_size);
    rocfft_execution_info forward_info;
    rocfft_execution_info_create(&forward_info);
    rocfft_execution_info_set_work_buffer(forward_info, work_buffer, work_size);

    rocfft_plan inverse_plan;
    rocfft_plan_create(&inverse_plan, rocfft_placement_notinplace, rocfft_transform_type_real_inverse, rocfft_precision_double,
                       3, lengths, 1, nullptr);

    rocfft_execution_info inverse_info;
    rocfft_execution_info_create(&inverse_info);
    rocfft_execution_info_set_work_buffer(inverse_info, work_buffer, work_size);

    int blockSize = 256;
    int gridSize = (Nx * Ny * ((Nz/2)+1) + blockSize - 1) / blockSize;
    /*Variable Setup*/

    /*Hockney Computation*/
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);

    hipLaunchKernelGGL(zeroPadKernel, gridDim, blockDim, 0, 0, d_extended_input, d_input, nx, ny, nz, Nx, Ny, Nz);

    rocfft_execute(forward_plan, (void**)&d_extended_input, (void**)&d_out, forward_info);
 
    hipLaunchKernelGGL(complexMultiplyKernel, gridSize, blockSize, 0, 0, d_out, d_temp, d_temp, Nx * Ny * ((Nz/2)+1));
    
    rocfft_execute(inverse_plan, (void**)&d_temp, (void**)&d_out2, inverse_info);
    
    hipLaunchKernelGGL(extractKernel, gridDimout, blockDim, 0, 0, d_out2, d_output, Nx, Ny, Nz, mx, my, mz);
    
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float milliseconds;
    hipEventElapsedTime(&milliseconds, start, stop);
    /*Hockney Computation*/

    hipMemcpy(output.data(), d_output, mx * my * mz *sizeof(double), hipMemcpyDeviceToHost);
    
     /*FFTX*/
    double *fftx_input, *fftx_sym, *fftx_output;
    hipMalloc(&fftx_input, nx*ny*nz*sizeof(double));
    hipMalloc(&fftx_sym, 2* Nx * Ny * ((Nz/2)+1)*sizeof(double));
    hipMalloc(&fftx_output, mx*my*mz*sizeof(double));
    hipMemcpy(fftx_input, input.data(), nx*ny*nz*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(fftx_sym, input2.data(), 2* Nx * Ny * ((Nz/2)+1)*sizeof(double), hipMemcpyHostToDevice);
    
    std::vector<void*>args{fftx_output, fftx_input, fftx_sym};
    std::vector<int> sizes{nx, ny, nz};
    HOCKNEYCONVProblem hcp(args, sizes, "hockney");
    hcp.transform();
    float time = hcp.getTime();

     /*FFTX*/

    hipMemcpy(spiral_output.data(), fftx_output, mx*my*mz*sizeof(double), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    checkOutputBuffers(output.data(), spiral_output.data(), mx*my*mz);

    std::cout << "The execution time for\nvendor: " << milliseconds << " ms\n" << "FFTX: " << time << " ms" << std::endl; 

    // Clean up
    hipFree(d_extended_input);
    hipFree(d_out);
    hipFree(d_temp);
    hipFree(d_out2);
    rocfft_plan_destroy(forward_plan);
    rocfft_plan_destroy(inverse_plan);

    return 0;
}
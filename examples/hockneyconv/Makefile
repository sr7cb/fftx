FFTW_HOME=/home/sanilr/fftw-3.3.10
CC=g++
NVCC=nvcc
HIPCC=hipcc
INCLUDE=${FFTX_HOME}/include
LIB=${FFTX_HOME}/lib

all: test

# test:
# 	$(CC) -std=c++17 -O3 -w -I$(INCLUDE) hockney_test.cpp -o cpu_hockney -lfftw3

cuda:
	$(NVCC) -O3 -w -I$(INCLUDE) -DFFTX_CUDA hockney_test_cuda.cu -o cuda_hockney -L$(LIB) -lcuda -lnvrtc -lcufft -lfftx_dftbat_gpu -lfftx_idftbat_gpu -lfftx_mddft_gpu -lfftx_imddft_gpu -lfftx_mdprdft_gpu -lfftx_imdprdft_gpu -lfftx_rconv_gpu -lfftx_prdftbat_gpu -lfftx_iprdftbat_gpu

hip:
	$(HIPCC) -O3 -w -I$(INCLUDE_SRC) -I$(INCLUDE_LIB) -DFFTX_HIP hockney_test_hip.cpp -o hip_hockney -lrocfft

clean:
	rm *_hockney
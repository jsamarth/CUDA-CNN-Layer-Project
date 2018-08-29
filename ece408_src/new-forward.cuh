#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 16
#include <mxnet/base.h>
#include <stdio.h>
#include <math.h>

// #define CMEMSIZE 150
// #define CMEMSIZE2 



namespace mxnet
{
namespace op
{

__constant__ float constant_kernel2[6*25*16]; //for the second kernel
__constant__ float constant_kernel[6*25]; //for the first kernel

__global__ void forward_kernel1(float *y, const float *x, const float *k, const int B, const int M, const int H, const int W, const int K, const int W_out, const int H_out, const int W_grid) {
	//macros
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define k4d(i3, i2, i1, i0) k[(i3) * (K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	int b, m, h, w, c, p, q, ty, tx, i, j, hbase, wbase;
	int X_tile_width = TILE_WIDTH + K-1;
	ty = threadIdx.y;
	tx = threadIdx.x;

	//Shared memory and atomics optimization
	//Attempting to index on the z dimension with C to do multiple channels at once along with shared memory
	b = blockIdx.x;
	m = blockIdx.y;
	c = 0;

	hbase = ((blockIdx.z) / W_grid) * TILE_WIDTH;
	wbase = ((blockIdx.z) % W_grid) * TILE_WIDTH;
	h = hbase + ty;
	w = wbase + tx;

	extern __shared__ float shmem[];
	float* X_s = &shmem[0];
	// float* W_s = &shmem[X_tile_width*X_tile_width];

	float acc = 0.0;

	// if ((tx < K) && (ty < K)) {
	// 	W_s[ty*K + tx] = k4d(m, c, ty, tx);
	// }
	// __syncthreads();

	//loading X to shared memory
	for (i = h; i < hbase + X_tile_width; i += TILE_WIDTH) {
		for (j = w; j < wbase + X_tile_width; j += TILE_WIDTH) {
			if (i < H && j < W) {
				X_s[(i - hbase)*X_tile_width + (j - wbase)] = x4d(b, c, i, j);
			} else {
				X_s[(i - hbase)*X_tile_width + (j - wbase)] = 0.0;
			}
		}
	}
	__syncthreads();

	//accumulating value
	for (p = 0; p < K; p++) {
		for (q = 0; q < K; q++) {
			acc += X_s[(ty + p)*X_tile_width + (tx + q)] * constant_kernel[m*25*1 + c*25 + p*K + q];
		}
	}

	__syncthreads();

	if (w < W_out && h < H_out) { 
		y4d(b, m, h, w) = acc;
	}

	#undef y4d
	#undef x4d
	#undef k4d
}

__global__ void forward_kernel2(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int Z, const int W_out, const int H_out, const int W_grid) {   //New parameters added

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	//#define y4d(b, m, h, w) y[(b) * (M * H_out * W_out) + (m) * (H_out * W_out) + (h) * (W_out) + w]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	//#define x4d(b, c, h + p, w + q) x[(b) * (C * H * W) + (c) * (H * W) + (h + p) * (W) + (w + q)]
	#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
	//#define k4d(m, c, p, q) k[(m) * (C * K * K) + (c) * (K * K) + (p) * (K) + q]

	int b, m, h, w, c, p, q, ty, tx, i, j, hbase, wbase;
	int X_tile_width = TILE_WIDTH + K-1;
	ty = threadIdx.y;
	tx = threadIdx.x;

	//Shared memory and atomics optimization
	//Attempting to index on the z dimension with C to do multiple channels at once along with shared memory
	b = blockIdx.x;
	m = blockIdx.y;
	// c = blockIdx.z / Z; // gives the channel to impliment

	hbase = ((blockIdx.z) / W_grid) * TILE_WIDTH;
	wbase = ((blockIdx.z) % W_grid) * TILE_WIDTH;
	h = hbase + ty;
	w = wbase + tx;

	extern __shared__ float shmem[];
	float* X_s = &shmem[0];
	// float* W_s = &shmem[X_tile_width*X_tile_width];

	float acc = 0.0;

	#pragma unroll 6
	for (c = 0; c < C; c++) {
	
		// if ((tx < K) && (ty < K)) {
		// 	W_s[ty*K + tx] = k4d(m, c, ty, tx);
		// }
		// __syncthreads();

		//loading X to shared memory
		for (i = h; i < hbase + X_tile_width; i += TILE_WIDTH) {
			for (j = w; j < wbase + X_tile_width; j += TILE_WIDTH) {
				if (i < H && j < W) {
					X_s[(i - hbase)*X_tile_width + (j - wbase)] = x4d(b, c, i, j);
				} else {
					X_s[(i - hbase)*X_tile_width + (j - wbase)] = 0.0;
				}
			}
		}
		__syncthreads();

		//accumulating value
		for (p = 0; p < K; p++) {
			for (q = 0; q < K; q++) {
				acc += X_s[(ty + p)*X_tile_width + (tx + q)] * constant_kernel2[m*25*6 + c*25 + p*K + q];
			}
		}

		__syncthreads();

	}

	if (w < W_out && h < H_out) { 
		// if (C == 1) {
			y4d(b, m, h, w) = acc;
		// } else {
		// 	atomicAdd(&(y4d(b, m, h, w)), acc);
		// }
	}


	#undef y4d
	#undef x4d
	#undef k4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!

    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    // cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0]; // input batch
    const int M = w.shape_[0]; // Number of feature maps
    const int C = x.shape_[1]; // Number of Input Channels
    const int H = x.shape_[2]; // input height
    const int W = x.shape_[3]; // input width
    const int K = w.shape_[2]; // size of the mask, it is a square mask

    // Set the kernel dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = (W_out-1)/TILE_WIDTH + 1;
    const int H_grid = (H_out-1)/TILE_WIDTH + 1;

    printf("Size of (H, W): %d, %d\n",H, W);
    printf("Size of (H_out, W_out): %d, %d\n",H_out, W_out);
    printf("Size of (H_grid, W_grid): %d, %d\n",H_grid, W_grid);
    printf("Number of Input Channels: %d\n", C);
    printf("Number of Feature Maps: %d\n", M);

    const int Z = W_grid*H_grid;
    // const int BLK_C = (Z-1)/TILE_WIDTH + 1; //using ceil to calculate how many blocks there are for the output image (perchannel)
    const int Z_c = Z*C; //gives to total Z_dim for the grid;

    printf("Number of Blocks per channel: %d\n", Z);
    printf("Number of Blocks for all channels: %d\n", Z_c);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z); //batches are mapped to the grid dimensions, blockx corresponds the the input batch, blocky corresponds to the feature map,
    //blockz which index(s) w and h is/should it be outputing to.

   // __constant__ float cmem1[CMEMSIZE];
    // allocate constant_kernel
    //cudaMemcpyToSymbol(constant_kernel, w.dptr_, sizeof(float) * 50 * 25, 0, cudaMemcpyDeviceToDevice);

    // Call the kernel
    // forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    // .dptr_ gives the data pointer? the address of the data?

    //loading the constant memory
    if (M == 16) { //6 channels, K*K Dim, 16 feature maps
    	cudaMemcpyToSymbol(constant_kernel2, w.dptr_, sizeof(float)*6*25*16, 0, cudaMemcpyDeviceToDevice);
    }
    else if (M == 6) { //K*K Dim, 6 feature maps, 1 channel
    	cudaMemcpyToSymbol(constant_kernel, w.dptr_, sizeof(float)*25*6, 0, cudaMemcpyDeviceToDevice);
    }

    size_t smsize = sizeof(float)*((TILE_WIDTH+K-1)*(TILE_WIDTH+K-1));

    if (C == 1) {
    	forward_kernel1<<<gridDim, blockDim, smsize>>>(y.dptr_, x.dptr_, w.dptr_, B,M,H,W,K,W_out,H_out,W_grid);
    } else {
    	forward_kernel2<<<gridDim, blockDim, smsize>>>(y.dptr_, x.dptr_, w.dptr_, B,M,C,H,W,K,Z, W_out, H_out, W_grid); //extra parameters to reduce redundant integer calculations
    }

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}

}
}

#endif


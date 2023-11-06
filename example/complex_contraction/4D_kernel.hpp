// created by tc_code_include() in tc_code_include.py
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <locale.h>
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>


using namespace std;

// created by tc_gen_definition_new()
#define SIZE_SLICE_1_E 16
#define SIZE_SLICE_1_F 1
#define SIZE_SLICE_1_A 16
#define SIZE_SLICE_1_B 6
#define SIZE_SLICE_1_D 16
#define SIZE_SLICE_1_C 6

#define SIZE_INT_UNIT_1 SIZE_SLICE_1_E * SIZE_SLICE_1_F

#define SIZE_TB_1_X 	SIZE_SLICE_1_A
#define SIZE_TB_1_Y 	SIZE_SLICE_1_D
#define SIZE_REG_1_X 	SIZE_SLICE_1_B
#define SIZE_REG_1_Y 	SIZE_SLICE_1_C

#define NUM_INDEX 		4
#define CEIL(a, b) 		(((a) + (b) - 1) / (b))

// Not Yet: Multiple Tensor Contractions.
// |Constant Memory| = 64KB, 16K Words(Integer), which means |K| <= 8192
#define MAX_CONST_LEN 		8192
__constant__ int const_internal_t2_offset[MAX_CONST_LEN];
__constant__ int const_internal_v2_offset[MAX_CONST_LEN];



struct Complex
{
	/* data */
	float re;
	float im;
};


// using Complex_t = _BitInt(64) ;




__device__ Complex ComplexMul(Complex a, Complex b)
{
	Complex c;

	c.re = a.re * b.re - a.im * b.im ;
	c.im = a.re * b.im + a.im * b.re ;

	return c;
}


__device__ Complex ComplexAdd(Complex a, Complex b)
{
	Complex c;

	c.re = a.re + b.re ;
	c.im = a.im + b.im ;

	return c;

}


__global__ void kernel__1_1(Complex* dev_t3, 
Complex* dev_t2, 
Complex* dev_v2, 
int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, 
int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_d, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ Complex sm_a[16][96];
	__shared__ Complex sm_b[16][96];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x;
	int idx_d = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + (blk_idx_d * SIZE_SLICE_1_D + idx_d) * size_c) * size_b) * size_a;


	Complex temp_av;
	Complex temp_bv[6];
	Complex reg_tile[6][6];

	for (int i = 0; i < 6; i++){
		for (int j = 0; j < 6; j++){

			reg_tile[i][j].re = 0.0;
			reg_tile[i][j].im = 0.0;
		}
	}

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'e', 'b', 'f']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['d', 'f', 'c', 'e']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		// No Need to Put Boundary-Checks before For-Statement: : 
		for (int ll = 0; ll < 6; ll++)
		{
			// ['a', 'e', 'b', 'f']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + ((blk_idx_b * SIZE_SLICE_1_B + ll) * size_e) * size_a + const_internal_t2_offset[threadIdx.y + l]];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		// No Need to Put Boundary-Checks before For-Statement: : 
		for (int ll = 0; ll < 6; ll++)
		{
			// ['d', 'f', 'c', 'e']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_d
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_d * SIZE_SLICE_1_D + idx_a + ((blk_idx_c * SIZE_SLICE_1_C + ll) * size_f) * size_d + const_internal_v2_offset[threadIdx.y + l]];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d + 0];
			temp_bv[1] = sm_b[ll][idx_d + 16];
			temp_bv[2] = sm_b[ll][idx_d + 32];
			temp_bv[3] = sm_b[ll][idx_d + 48];
			temp_bv[4] = sm_b[ll][idx_d + 64];
			temp_bv[5] = sm_b[ll][idx_d + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (xx * 16)];

				// reg_tile[0][xx] += temp_av * temp_bv[0];
				// reg_tile[1][xx] += temp_av * temp_bv[1];
				// reg_tile[2][xx] += temp_av * temp_bv[2];
				// reg_tile[3][xx] += temp_av * temp_bv[3];
				// reg_tile[4][xx] += temp_av * temp_bv[4];
				// reg_tile[5][xx] += temp_av * temp_bv[5];

				reg_tile[0][xx] = ComplexAdd(reg_tile[0][xx] , ComplexMul(temp_av, temp_bv[0] )) ;
				reg_tile[1][xx] = ComplexAdd(reg_tile[1][xx] , ComplexMul(temp_av, temp_bv[1] )) ;
				reg_tile[2][xx] = ComplexAdd(reg_tile[2][xx] , ComplexMul(temp_av, temp_bv[2] )) ;
				reg_tile[3][xx] = ComplexAdd(reg_tile[3][xx] , ComplexMul(temp_av, temp_bv[3] )) ;
				reg_tile[4][xx] = ComplexAdd(reg_tile[4][xx] , ComplexMul(temp_av, temp_bv[4] )) ;
				reg_tile[5][xx] = ComplexAdd(reg_tile[5][xx] , ComplexMul(temp_av, temp_bv[5] )) ;
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	#pragma unroll 6
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// Tensor Contraction Kernel
__global__ void kernel__2_1(Complex* dev_t3, 
Complex* dev_t2, 
Complex* dev_v2, 
int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, 
int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_d, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ Complex sm_a[16][96];
	__shared__ Complex sm_b[16][96];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x;
	int idx_d = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + (blk_idx_d * SIZE_SLICE_1_D + idx_d) * size_c) * size_b) * size_a;


	Complex temp_av;
	Complex temp_bv[6];
	Complex reg_tile[6][6];

	for (int i = 0; i < 6; i++){
		for (int j = 0; j < 6; j++){

			reg_tile[i][j].re = 0.0;
			reg_tile[i][j].im = 0.0;
		}
	}

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'e', 'b', 'f']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['d', 'f', 'c', 'e']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < 6; ll++)
		{
			// ['a', 'e', 'b', 'f']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + ((blk_idx_b * SIZE_SLICE_1_B + ll) * size_e) * size_a + const_internal_t2_offset[threadIdx.y + l]];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < 6; ll++)
		{
			// ['d', 'f', 'c', 'e']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_d
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_d * SIZE_SLICE_1_D + idx_a + ((blk_idx_c * SIZE_SLICE_1_C + ll) * size_f) * size_d + const_internal_v2_offset[threadIdx.y + l]];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d + 0];
			temp_bv[1] = sm_b[ll][idx_d + 16];
			temp_bv[2] = sm_b[ll][idx_d + 32];
			temp_bv[3] = sm_b[ll][idx_d + 48];
			temp_bv[4] = sm_b[ll][idx_d + 64];
			temp_bv[5] = sm_b[ll][idx_d + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (xx * 16)];

				// reg_tile[0][xx] += temp_av * temp_bv[0];
				// reg_tile[1][xx] += temp_av * temp_bv[1];
				// reg_tile[2][xx] += temp_av * temp_bv[2];
				// reg_tile[3][xx] += temp_av * temp_bv[3];
				// reg_tile[4][xx] += temp_av * temp_bv[4];
				// reg_tile[5][xx] += temp_av * temp_bv[5];

				reg_tile[0][xx] = ComplexAdd(reg_tile[0][xx] , ComplexMul(temp_av, temp_bv[0] )) ;
				reg_tile[1][xx] = ComplexAdd(reg_tile[1][xx] , ComplexMul(temp_av, temp_bv[1] )) ;
				reg_tile[2][xx] = ComplexAdd(reg_tile[2][xx] , ComplexMul(temp_av, temp_bv[2] )) ;
				reg_tile[3][xx] = ComplexAdd(reg_tile[3][xx] , ComplexMul(temp_av, temp_bv[3] )) ;
				reg_tile[4][xx] = ComplexAdd(reg_tile[4][xx] , ComplexMul(temp_av, temp_bv[4] )) ;
				reg_tile[5][xx] = ComplexAdd(reg_tile[5][xx] , ComplexMul(temp_av, temp_bv[5] )) ;
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	#pragma unroll 6
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// Tensor Contraction Kernel
__global__ void kernel__3_1(Complex* dev_t3, 
Complex* dev_t2, 
Complex* dev_v2, 
int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, 
int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_d, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ Complex sm_a[16][96];
	__shared__ Complex sm_b[16][96];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x;
	int idx_d = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + (blk_idx_d * SIZE_SLICE_1_D + idx_d) * size_c) * size_b) * size_a;

	// need to support partial tiles
	int rng_a, rng_b, rng_c, rng_d;
	if ((size_a - (blk_idx_a * SIZE_SLICE_1_A)) >= SIZE_SLICE_1_A)
	{
		rng_a = SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % SIZE_SLICE_1_A;
	}
	if ((size_b - (blk_idx_b * SIZE_SLICE_1_B)) >= SIZE_SLICE_1_B)
	{
		rng_b = SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % SIZE_SLICE_1_B;
	}
	if ((size_c - (blk_idx_c * SIZE_SLICE_1_C)) >= SIZE_SLICE_1_C)
	{
		rng_c = SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % SIZE_SLICE_1_C;
	}
	if ((size_d - (blk_idx_d * SIZE_SLICE_1_D)) >= SIZE_SLICE_1_D)
	{
		rng_d = SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % SIZE_SLICE_1_D;
	}

	Complex temp_av;
	Complex temp_bv[6];
	Complex reg_tile[6][6];

	for (int i = 0; i < 6; i++){
		for (int j = 0; j < 6; j++){

			reg_tile[i][j].re = 0.0;
			reg_tile[i][j].im = 0.0;
		}
	}
		

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'e', 'b', 'f']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['d', 'f', 'c', 'e']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		
		if (idx_a < rng_a)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'e', 'b', 'f']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + ((blk_idx_b * SIZE_SLICE_1_B + ll) * size_e) * size_a + const_internal_t2_offset[threadIdx.y + l]];
		}
		
		// This Part is for Loading Input-Right
		
		if (idx_a < rng_d)
		for (int ll = 0; ll < rng_c; ll++)
		{
			// ['d', 'f', 'c', 'e']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_d
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_d * SIZE_SLICE_1_D + idx_a + ((blk_idx_c * SIZE_SLICE_1_C + ll) * size_f) * size_d + const_internal_v2_offset[threadIdx.y + l]];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d + 0];
			temp_bv[1] = sm_b[ll][idx_d + 16];
			temp_bv[2] = sm_b[ll][idx_d + 32];
			temp_bv[3] = sm_b[ll][idx_d + 48];
			temp_bv[4] = sm_b[ll][idx_d + 64];
			temp_bv[5] = sm_b[ll][idx_d + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (xx * 16)];

				// reg_tile[0][xx] += temp_av * temp_bv[0];
				// reg_tile[1][xx] += temp_av * temp_bv[1];
				// reg_tile[2][xx] += temp_av * temp_bv[2];
				// reg_tile[3][xx] += temp_av * temp_bv[3];
				// reg_tile[4][xx] += temp_av * temp_bv[4];
				// reg_tile[5][xx] += temp_av * temp_bv[5];


				reg_tile[0][xx] = ComplexAdd(reg_tile[0][xx] , ComplexMul(temp_av, temp_bv[0] )) ;
				reg_tile[1][xx] = ComplexAdd(reg_tile[1][xx] , ComplexMul(temp_av, temp_bv[1] )) ;
				reg_tile[2][xx] = ComplexAdd(reg_tile[2][xx] , ComplexMul(temp_av, temp_bv[2] )) ;
				reg_tile[3][xx] = ComplexAdd(reg_tile[3][xx] , ComplexMul(temp_av, temp_bv[3] )) ;
				reg_tile[4][xx] = ComplexAdd(reg_tile[4][xx] , ComplexMul(temp_av, temp_bv[4] )) ;
				reg_tile[5][xx] = ComplexAdd(reg_tile[5][xx] , ComplexMul(temp_av, temp_bv[5] )) ;

			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_d < rng_d)
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			if(i < rng_c && j < rng_b)
			{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
			}
		}
	}
}

// Tensor Contraction Kernel
__global__ void kernel__4_1(Complex* dev_t3, 
Complex* dev_t2, 
Complex* dev_v2, 
int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, 
int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_d, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ Complex sm_a[16][96];
	__shared__ Complex sm_b[16][96];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x;
	int idx_d = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + (blk_idx_d * SIZE_SLICE_1_D + idx_d) * size_c) * size_b) * size_a;

	// need to support partial tiles
	int rng_a, rng_b, rng_c, rng_d;
	if ((size_a - (blk_idx_a * SIZE_SLICE_1_A)) >= SIZE_SLICE_1_A)
	{
		rng_a = SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % SIZE_SLICE_1_A;
	}
	if ((size_b - (blk_idx_b * SIZE_SLICE_1_B)) >= SIZE_SLICE_1_B)
	{
		rng_b = SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % SIZE_SLICE_1_B;
	}
	if ((size_c - (blk_idx_c * SIZE_SLICE_1_C)) >= SIZE_SLICE_1_C)
	{
		rng_c = SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % SIZE_SLICE_1_C;
	}
	if ((size_d - (blk_idx_d * SIZE_SLICE_1_D)) >= SIZE_SLICE_1_D)
	{
		rng_d = SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % SIZE_SLICE_1_D;
	}

	Complex temp_av;
	Complex temp_bv[6];
	Complex reg_tile[6][6];

	for (int i = 0; i < 6; i++){
		for (int j = 0; j < 6; j++){

			reg_tile[i][j].re = 0.0;
			reg_tile[i][j].im = 0.0;
		}
	}

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'e', 'b', 'f']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['d', 'f', 'c', 'e']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_a && threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'e', 'b', 'f']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + ((blk_idx_b * SIZE_SLICE_1_B + ll) * size_e) * size_a + const_internal_t2_offset[threadIdx.y + l]];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_d && threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_c; ll++)
		{
			// ['d', 'f', 'c', 'e']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_d
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_d * SIZE_SLICE_1_D + idx_a + ((blk_idx_c * SIZE_SLICE_1_C + ll) * size_f) * size_d + const_internal_v2_offset[threadIdx.y + l]];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d + 0];
			temp_bv[1] = sm_b[ll][idx_d + 16];
			temp_bv[2] = sm_b[ll][idx_d + 32];
			temp_bv[3] = sm_b[ll][idx_d + 48];
			temp_bv[4] = sm_b[ll][idx_d + 64];
			temp_bv[5] = sm_b[ll][idx_d + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (xx * 16)];

				// reg_tile[0][xx] += temp_av * temp_bv[0];
				// reg_tile[1][xx] += temp_av * temp_bv[1];
				// reg_tile[2][xx] += temp_av * temp_bv[2];
				// reg_tile[3][xx] += temp_av * temp_bv[3];
				// reg_tile[4][xx] += temp_av * temp_bv[4];
				// reg_tile[5][xx] += temp_av * temp_bv[5];

				reg_tile[0][xx] = ComplexAdd(reg_tile[0][xx] , ComplexMul(temp_av, temp_bv[0] )) ;
				reg_tile[1][xx] = ComplexAdd(reg_tile[1][xx] , ComplexMul(temp_av, temp_bv[1] )) ;
				reg_tile[2][xx] = ComplexAdd(reg_tile[2][xx] , ComplexMul(temp_av, temp_bv[2] )) ;
				reg_tile[3][xx] = ComplexAdd(reg_tile[3][xx] , ComplexMul(temp_av, temp_bv[3] )) ;
				reg_tile[4][xx] = ComplexAdd(reg_tile[4][xx] , ComplexMul(temp_av, temp_bv[4] )) ;
				reg_tile[5][xx] = ComplexAdd(reg_tile[5][xx] , ComplexMul(temp_av, temp_bv[5] )) ;

			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_d < rng_d)
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			if(i < rng_c && j < rng_b)
			{
				dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
			}
		}
	}
}

// Tensor Contraction Kernel
__global__ void kernel__1_tex_1(Complex* dev_t3, 
Complex* dev_t2, 
Complex* dev_v2, 
int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, 
int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_d, 
int* dev_internal_offset_t2, int* dev_internal_offset_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ Complex sm_a[16][96];
	__shared__ Complex sm_b[16][96];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x;
	int idx_d = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + (blk_idx_d * SIZE_SLICE_1_D + idx_d) * size_c) * size_b) * size_a;


	Complex temp_av;
	Complex temp_bv[6];
	Complex reg_tile[6][6];

	for (int i = 0; i < 6; i++){
		for (int j = 0; j < 6; j++){

			reg_tile[i][j].re = 0.0;
			reg_tile[i][j].im = 0.0;
		}
	}

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'e', 'b', 'f']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['d', 'f', 'c', 'e']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		// No Need to Put Boundary-Checks before For-Statement: : 
		for (int ll = 0; ll < 6; ll++)
		{
			// ['a', 'e', 'b', 'f']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + ((blk_idx_b * SIZE_SLICE_1_B + ll) * size_e) * size_a + dev_internal_offset_t2[threadIdx.y + l]];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		// No Need to Put Boundary-Checks before For-Statement: : 
		for (int ll = 0; ll < 6; ll++)
		{
			// ['d', 'f', 'c', 'e']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_d
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_d * SIZE_SLICE_1_D + idx_a + ((blk_idx_c * SIZE_SLICE_1_C + ll) * size_f) * size_d + dev_internal_offset_v2[threadIdx.y + l]];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d + 0];
			temp_bv[1] = sm_b[ll][idx_d + 16];
			temp_bv[2] = sm_b[ll][idx_d + 32];
			temp_bv[3] = sm_b[ll][idx_d + 48];
			temp_bv[4] = sm_b[ll][idx_d + 64];
			temp_bv[5] = sm_b[ll][idx_d + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (xx * 16)];

				// reg_tile[0][xx] += temp_av * temp_bv[0];
				// reg_tile[1][xx] += temp_av * temp_bv[1];
				// reg_tile[2][xx] += temp_av * temp_bv[2];
				// reg_tile[3][xx] += temp_av * temp_bv[3];
				// reg_tile[4][xx] += temp_av * temp_bv[4];
				// reg_tile[5][xx] += temp_av * temp_bv[5];

				reg_tile[0][xx] = ComplexAdd(reg_tile[0][xx] , ComplexMul(temp_av, temp_bv[0] )) ;
				reg_tile[1][xx] = ComplexAdd(reg_tile[1][xx] , ComplexMul(temp_av, temp_bv[1] )) ;
				reg_tile[2][xx] = ComplexAdd(reg_tile[2][xx] , ComplexMul(temp_av, temp_bv[2] )) ;
				reg_tile[3][xx] = ComplexAdd(reg_tile[3][xx] , ComplexMul(temp_av, temp_bv[3] )) ;
				reg_tile[4][xx] = ComplexAdd(reg_tile[4][xx] , ComplexMul(temp_av, temp_bv[4] )) ;
				reg_tile[5][xx] = ComplexAdd(reg_tile[5][xx] , ComplexMul(temp_av, temp_bv[5] )) ;
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	#pragma unroll 6
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}


__global__ void kernel__2_tex_1(Complex* dev_t3, 
Complex* dev_t2, 
Complex* dev_v2, 
int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, 
int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_d, 
int* dev_internal_offset_t2, int* dev_internal_offset_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ Complex sm_a[16][96];
	__shared__ Complex sm_b[16][96];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x;
	int idx_d = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + (blk_idx_d * SIZE_SLICE_1_D + idx_d) * size_c) * size_b) * size_a;

	Complex temp_av;
	Complex temp_bv[6];
	Complex reg_tile[6][6];

	for (int i = 0; i < 6; i++){
		for (int j = 0; j < 6; j++){

			reg_tile[i][j].re = 0.0;
			reg_tile[i][j].im = 0.0;
		}
	}


	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'e', 'b', 'f']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['d', 'f', 'c', 'e']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left

		if (threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < 6; ll++)
		{
			// ['a', 'e', 'b', 'f']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + ((blk_idx_b * SIZE_SLICE_1_B + ll) * size_e) * size_a + dev_internal_offset_t2[threadIdx.y + l]];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < 6; ll++)
		{
			// ['d', 'f', 'c', 'e']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_d
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_d * SIZE_SLICE_1_D + idx_a + ((blk_idx_c * SIZE_SLICE_1_C + ll) * size_f) * size_d + dev_internal_offset_v2[threadIdx.y + l]];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d + 0];
			temp_bv[1] = sm_b[ll][idx_d + 16];
			temp_bv[2] = sm_b[ll][idx_d + 32];
			temp_bv[3] = sm_b[ll][idx_d + 48];
			temp_bv[4] = sm_b[ll][idx_d + 64];
			temp_bv[5] = sm_b[ll][idx_d + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (xx * 16)];

				// reg_tile[0][xx] += temp_av * temp_bv[0];
				// reg_tile[1][xx] += temp_av * temp_bv[1];
				// reg_tile[2][xx] += temp_av * temp_bv[2];
				// reg_tile[3][xx] += temp_av * temp_bv[3];
				// reg_tile[4][xx] += temp_av * temp_bv[4];
				// reg_tile[5][xx] += temp_av * temp_bv[5];

				reg_tile[0][xx] = ComplexAdd(reg_tile[0][xx] , ComplexMul(temp_av, temp_bv[0] )) ;
				reg_tile[1][xx] = ComplexAdd(reg_tile[1][xx] , ComplexMul(temp_av, temp_bv[1] )) ;
				reg_tile[2][xx] = ComplexAdd(reg_tile[2][xx] , ComplexMul(temp_av, temp_bv[2] )) ;
				reg_tile[3][xx] = ComplexAdd(reg_tile[3][xx] , ComplexMul(temp_av, temp_bv[3] )) ;
				reg_tile[4][xx] = ComplexAdd(reg_tile[4][xx] , ComplexMul(temp_av, temp_bv[4] )) ;
				reg_tile[5][xx] = ComplexAdd(reg_tile[5][xx] , ComplexMul(temp_av, temp_bv[5] )) ;
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	#pragma unroll 6
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
		}
	}
}

// Tensor Contraction Kernel
__global__ void kernel__3_tex_1(Complex* dev_t3, 
Complex* dev_t2, 
Complex* dev_v2, 
int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, 
int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_d, 
int* dev_internal_offset_t2, int* dev_internal_offset_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ Complex sm_a[16][96];
	__shared__ Complex sm_b[16][96];


	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x;
	int idx_d = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + (blk_idx_d * SIZE_SLICE_1_D + idx_d) * size_c) * size_b) * size_a;

	// need to support partial tiles
	int rng_a, rng_b, rng_c, rng_d;
	if ((size_a - (blk_idx_a * SIZE_SLICE_1_A)) >= SIZE_SLICE_1_A)
	{
		rng_a = SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % SIZE_SLICE_1_A;
	}
	if ((size_b - (blk_idx_b * SIZE_SLICE_1_B)) >= SIZE_SLICE_1_B)
	{
		rng_b = SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % SIZE_SLICE_1_B;
	}
	if ((size_c - (blk_idx_c * SIZE_SLICE_1_C)) >= SIZE_SLICE_1_C)
	{
		rng_c = SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % SIZE_SLICE_1_C;
	}
	if ((size_d - (blk_idx_d * SIZE_SLICE_1_D)) >= SIZE_SLICE_1_D)
	{
		rng_d = SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % SIZE_SLICE_1_D;
	}

	Complex temp_av;
	Complex temp_bv[6];
	Complex reg_tile[6][6];

	for (int i = 0; i < 6; i++){
		for (int j = 0; j < 6; j++){

			reg_tile[i][j].re = 0.0;
			reg_tile[i][j].im = 0.0;
		}
	}

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'e', 'b', 'f']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['d', 'f', 'c', 'e']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_a)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'e', 'b', 'f']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + ((blk_idx_b * SIZE_SLICE_1_B + ll) * size_e) * size_a + dev_internal_offset_t2[threadIdx.y + l]];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_d)
		for (int ll = 0; ll < rng_c; ll++)
		{
			// ['d', 'f', 'c', 'e']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_d
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_d * SIZE_SLICE_1_D + idx_a + ((blk_idx_c * SIZE_SLICE_1_C + ll) * size_f) * size_d + dev_internal_offset_v2[threadIdx.y + l]];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d + 0];
			temp_bv[1] = sm_b[ll][idx_d + 16];
			temp_bv[2] = sm_b[ll][idx_d + 32];
			temp_bv[3] = sm_b[ll][idx_d + 48];
			temp_bv[4] = sm_b[ll][idx_d + 64];
			temp_bv[5] = sm_b[ll][idx_d + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (xx * 16)];

				// reg_tile[0][xx] += temp_av * temp_bv[0];
				// reg_tile[1][xx] += temp_av * temp_bv[1];
				// reg_tile[2][xx] += temp_av * temp_bv[2];
				// reg_tile[3][xx] += temp_av * temp_bv[3];
				// reg_tile[4][xx] += temp_av * temp_bv[4];
				// reg_tile[5][xx] += temp_av * temp_bv[5];

				reg_tile[0][xx] = ComplexAdd(reg_tile[0][xx] , ComplexMul(temp_av, temp_bv[0] )) ;
				reg_tile[1][xx] = ComplexAdd(reg_tile[1][xx] , ComplexMul(temp_av, temp_bv[1] )) ;
				reg_tile[2][xx] = ComplexAdd(reg_tile[2][xx] , ComplexMul(temp_av, temp_bv[2] )) ;
				reg_tile[3][xx] = ComplexAdd(reg_tile[3][xx] , ComplexMul(temp_av, temp_bv[3] )) ;
				reg_tile[4][xx] = ComplexAdd(reg_tile[4][xx] , ComplexMul(temp_av, temp_bv[4] )) ;
				reg_tile[5][xx] = ComplexAdd(reg_tile[5][xx] , ComplexMul(temp_av, temp_bv[5] )) ;

			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_d < rng_d)
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			if(i < rng_c && j < rng_b)
			{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
			}
		}
	}
}

// Tensor Contraction Kernel
__global__ void kernel__4_tex_1(Complex* dev_t3, 
Complex* dev_t2, 
Complex* dev_v2, 
int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, 
int numBlk_a, int numBlk_b, int numBlk_c, int numBlk_d, 
int* dev_internal_offset_t2, int* dev_internal_offset_v2, 
int stride_reg_x, int stride_reg_y, 
int size_internal)
{
	// For Shared Memory,
	__shared__ Complex sm_a[16][96];
	__shared__ Complex sm_b[16][96];


	int internal_upperbound   = 0;
	int internal_offset;

	// when opt_pre_computed == -1, all indices will be calculated manually
	// # of indices mapped on TB_X: 1
	// # of indices mapped on TB_Y: 1
	int idx_a = threadIdx.x;
	int idx_d = threadIdx.y;

	int tmp_blkIdx;
	int blk_idx_d = blockIdx.x / (numBlk_c * numBlk_b * numBlk_a);
	tmp_blkIdx = blockIdx.x % (numBlk_c * numBlk_b * numBlk_a);

	int blk_idx_c = tmp_blkIdx / (numBlk_b * numBlk_a);
	tmp_blkIdx = tmp_blkIdx % (numBlk_b * numBlk_a);

	int blk_idx_b = tmp_blkIdx / numBlk_a;
	tmp_blkIdx = tmp_blkIdx % (numBlk_a);

	int  blk_idx_a = tmp_blkIdx;

	int t3_base_thread = blk_idx_a * SIZE_SLICE_1_A + idx_a + (blk_idx_b * SIZE_SLICE_1_B + (blk_idx_c * SIZE_SLICE_1_C + (blk_idx_d * SIZE_SLICE_1_D + idx_d) * size_c) * size_b) * size_a;

	// need to support partial tiles
	int rng_a, rng_b, rng_c, rng_d;
	if ((size_a - (blk_idx_a * SIZE_SLICE_1_A)) >= SIZE_SLICE_1_A)
	{
		rng_a = SIZE_SLICE_1_A;
	}
	else
	{
		rng_a = size_a % SIZE_SLICE_1_A;
	}
	if ((size_b - (blk_idx_b * SIZE_SLICE_1_B)) >= SIZE_SLICE_1_B)
	{
		rng_b = SIZE_SLICE_1_B;
	}
	else
	{
		rng_b = size_b % SIZE_SLICE_1_B;
	}
	if ((size_c - (blk_idx_c * SIZE_SLICE_1_C)) >= SIZE_SLICE_1_C)
	{
		rng_c = SIZE_SLICE_1_C;
	}
	else
	{
		rng_c = size_c % SIZE_SLICE_1_C;
	}
	if ((size_d - (blk_idx_d * SIZE_SLICE_1_D)) >= SIZE_SLICE_1_D)
	{
		rng_d = SIZE_SLICE_1_D;
	}
	else
	{
		rng_d = size_d % SIZE_SLICE_1_D;
	}

	Complex temp_av;
	Complex temp_bv[6];
	Complex reg_tile[6][6];

	for (int i = 0; i < 6; i++){
		for (int j = 0; j < 6; j++){

			reg_tile[i][j].re = 0.0;
			reg_tile[i][j].im = 0.0;
		}
	}

	// tensor contraction: [[16, 'STR_SD2_T2_H7', 'x', 't2', ['a', 'e', 'b', 'f']], [16, 'STR_SD2_V2_H7', 'y', 'v2', ['d', 'f', 'c', 'e']], '+=']
	#pragma unroll 1
	for (int l = 0; l < size_internal; l += SIZE_INT_UNIT_1)
	{
		// Part: Generalized Contraction Index (p7b)
		internal_offset = (l + SIZE_INT_UNIT_1) - size_internal;
		if (internal_offset > 0) internal_upperbound = internal_offset;

		//---------------------------------------------------------------------------------------------------
		// This is for the new version
		// This Part is for Loading Input-Left
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_a && threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_b; ll++)
		{
			// ['a', 'e', 'b', 'f']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_a
			sm_a[threadIdx.y][threadIdx.x + ll * 16] = dev_t2[blk_idx_a * SIZE_SLICE_1_A + idx_a + ((blk_idx_b * SIZE_SLICE_1_B + ll) * size_e) * size_a + dev_internal_offset_t2[threadIdx.y + l]];
		}
		
		// This Part is for Loading Input-Right
		// tc_gen_code_Kernel_Load_Inputs_Abstracts()
		if (idx_a < rng_d && threadIdx.y < SIZE_INT_UNIT_1 - internal_upperbound)
		for (int ll = 0; ll < rng_c; ll++)
		{
			// ['d', 'f', 'c', 'e']
			// Exception: Temp. version!: threadIdx.y + l
			// Exception: Temp. version!: idx_a < rng_d
			sm_b[threadIdx.y][threadIdx.x + ll * 16] = dev_v2[blk_idx_d * SIZE_SLICE_1_D + idx_a + ((blk_idx_c * SIZE_SLICE_1_C + ll) * size_f) * size_d + dev_internal_offset_v2[threadIdx.y + l]];
		}
		__syncthreads();
		//---------------------------------------------------------------------------------------------------
		

		// Part: Generalized Threads
		for (int ll = 0; ll < SIZE_INT_UNIT_1 - internal_upperbound; ll++)
		{
			temp_bv[0] = sm_b[ll][idx_d + 0];
			temp_bv[1] = sm_b[ll][idx_d + 16];
			temp_bv[2] = sm_b[ll][idx_d + 32];
			temp_bv[3] = sm_b[ll][idx_d + 48];
			temp_bv[4] = sm_b[ll][idx_d + 64];
			temp_bv[5] = sm_b[ll][idx_d + 80];

			for (int xx = 0; xx < 6; xx++) // (1)
			{
				temp_av = sm_a[ll][idx_a + (xx * 16)];

				// reg_tile[0][xx] += temp_av * temp_bv[0];
				// reg_tile[1][xx] += temp_av * temp_bv[1];
				// reg_tile[2][xx] += temp_av * temp_bv[2];
				// reg_tile[3][xx] += temp_av * temp_bv[3];
				// reg_tile[4][xx] += temp_av * temp_bv[4];
				// reg_tile[5][xx] += temp_av * temp_bv[5];

				reg_tile[0][xx] = ComplexAdd(reg_tile[0][xx] , ComplexMul(temp_av, temp_bv[0] )) ;
				reg_tile[1][xx] = ComplexAdd(reg_tile[1][xx] , ComplexMul(temp_av, temp_bv[1] )) ;
				reg_tile[2][xx] = ComplexAdd(reg_tile[2][xx] , ComplexMul(temp_av, temp_bv[2] )) ;
				reg_tile[3][xx] = ComplexAdd(reg_tile[3][xx] , ComplexMul(temp_av, temp_bv[3] )) ;
				reg_tile[4][xx] = ComplexAdd(reg_tile[4][xx] , ComplexMul(temp_av, temp_bv[4] )) ;
				reg_tile[5][xx] = ComplexAdd(reg_tile[5][xx] , ComplexMul(temp_av, temp_bv[5] )) ;
			}
		}
		__syncthreads();
	}


	// Store Results (Registers) to Global Memory
	// Part: Generalized Threads
	// Part: Generalized Register-Tiling
	if (idx_a < rng_a && idx_d < rng_d)
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			if(i < rng_c && j < rng_b)
			{
			dev_t3[t3_base_thread + (i * stride_reg_y) + (j * stride_reg_x)] = reg_tile[i][j];
			}
		}
	}
}


extern "C"
void sd_t_d2_fusion(int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, Complex* t3, Complex* host_t2, Complex* host_v2, int cond_kernel_1, int opt_register_transpose)
{
	int num_thread_blocks_kernel_1;

	Complex* dev_t3;
	Complex* dev_t2;
	Complex* dev_v2;

	int* host_internal_left_offset;
	int* host_internal_right_offset;

	num_thread_blocks_kernel_1 = CEIL(size_a, SIZE_SLICE_1_A) * CEIL(size_b, SIZE_SLICE_1_B) * CEIL(size_c, SIZE_SLICE_1_C) * CEIL(size_d, SIZE_SLICE_1_D);
	// hipMalloc()
	hipMalloc((void**) &dev_t3, sizeof(Complex) * size_a * size_b * size_c * size_d);
	hipMalloc((void**) &dev_t2, sizeof(Complex) * size_f * size_b * size_e * size_a);
	hipMalloc((void**) &dev_v2, sizeof(Complex) * size_e * size_c * size_f * size_d);

	// hipMemcpy()
	hipMemcpy(dev_t3, t3, sizeof(Complex) * size_a * size_b * size_c * size_d, hipMemcpyHostToDevice);
	hipMemcpy(dev_t2, host_t2, sizeof(Complex) * size_f * size_b * size_e * size_a, hipMemcpyHostToDevice);
	hipMemcpy(dev_v2, host_v2, sizeof(Complex) * size_e * size_c * size_f * size_d, hipMemcpyHostToDevice);

	// Related to Kernels
	// There are 1 Basic Kernels
	long long int tmp_operations = (long long int)((long long int)(size_a * size_b * size_c * size_d) * size_e) * size_f;
	printf ("========================================= fusedKernels =============================================\n");
	printf ("		Grid Size  : %6d (1D)\n", num_thread_blocks_kernel_1);
	printf ("		Block-size : %2d, %2d (2D)\n", SIZE_TB_1_X, SIZE_TB_1_Y);
	printf ("		Reg.-size  : %2d, %2d (2D)\n", SIZE_REG_1_X, SIZE_REG_1_Y);
	printf ("		A thread deals with (%d x %d) elements (basically)\n", SIZE_TB_1_X * SIZE_REG_1_X, SIZE_TB_1_Y * SIZE_REG_1_Y);
	printf ("		# of Operations: %lld\n", tmp_operations);
	printf ("====================================================================================================\n");
	dim3 gridsize_1(num_thread_blocks_kernel_1);
	dim3 blocksize_1(SIZE_TB_1_X, SIZE_TB_1_Y);

	int stride_output_a = 1;
	int stride_output_b = stride_output_a * size_a;
	int stride_output_c = stride_output_b * size_b;
	int stride_output_d = stride_output_c * size_c;

	int stride_reg_x_1 = stride_output_b;
	int stride_reg_y_1 = stride_output_c;

	int size_internal = size_e * size_f;

	// (manually) ['e', 'f']
	host_internal_left_offset 	= (int*)malloc(sizeof(int) * size_internal);
	host_internal_right_offset 	= (int*)malloc(sizeof(int) * size_internal);
	for (int idx_f = 0; idx_f < size_f; idx_f++)
	for (int idx_e = 0; idx_e < size_e; idx_e++)
	{
		host_internal_left_offset[idx_e + (idx_f) * size_e] 	= (idx_e + ((idx_f) * size_b) * size_e) * size_a;
		host_internal_right_offset[idx_e + (idx_f) * size_e] 	= (idx_f + ((idx_e) * size_c) * size_f) * size_d;
	}

	hipMemcpyToSymbol(const_internal_t2_offset, host_internal_left_offset, sizeof(int) * size_internal);
	hipMemcpyToSymbol(const_internal_v2_offset, host_internal_right_offset, sizeof(int) * size_internal);

	int* dev_internal_offset_t2;
	int* dev_internal_offset_v2;
	// hipMalloc()
	hipMalloc((void**) &dev_internal_offset_t2, sizeof(int) * size_internal);
	hipMalloc((void**) &dev_internal_offset_v2, sizeof(int) * size_internal);

	// hipMemcpy()
	hipMemcpy(dev_internal_offset_t2, host_internal_left_offset, sizeof(int) * size_internal, hipMemcpyHostToDevice);
	hipMemcpy(dev_internal_offset_v2, host_internal_right_offset, sizeof(int) * size_internal, hipMemcpyHostToDevice);

	// Decision Tree for Kernel Types
	// No Chance to Utilize the Register Transpose
	if (size_a % SIZE_SLICE_1_A == 0 && size_b % SIZE_SLICE_1_B == 0 && size_c % SIZE_SLICE_1_C == 0 && size_d % SIZE_SLICE_1_D == 0)
	{
		// [2] Extenral Index: Full
		if (size_e % SIZE_SLICE_1_E == 0 && size_f % SIZE_SLICE_1_F == 0)
		{
			// [3] Internal Index: Full
			// >>> External: Full && Internal: Full
			printf ("External: Full, Internal: Full\n");
			if (size_internal > MAX_CONST_LEN)
			{
				kernel__1_tex_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_d, size_e, size_f, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), CEIL(size_d, SIZE_SLICE_1_D), dev_internal_offset_t2, dev_internal_offset_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
			}
			else
			{
				kernel__1_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_d, size_e, size_f, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), CEIL(size_d, SIZE_SLICE_1_D), stride_reg_x_1, stride_reg_y_1, size_internal);
			}
		}
		else
		{
			// [4] Internal Index: Partial
			// >>> External: Full && Internal: Partial
			printf ("External: Full, Internal: Partial\n");
			if (size_internal > MAX_CONST_LEN)
			{
				kernel__2_tex_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_d, size_e, size_f, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), CEIL(size_d, SIZE_SLICE_1_D), dev_internal_offset_t2, dev_internal_offset_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
			}
			else
			{
				kernel__2_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_d, size_e, size_f, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), CEIL(size_d, SIZE_SLICE_1_D), stride_reg_x_1, stride_reg_y_1, size_internal);
			}
		}
	}
	else
	{
		// [2] Extenral Index: Partial
		if (size_e % SIZE_SLICE_1_E == 0 && size_f % SIZE_SLICE_1_F == 0)
		{
			// [3] Internal Index: Full
			// >>> External: Partial && Internal: Full
			printf ("External: Partial, Internal: Full\n");
			if (size_internal > MAX_CONST_LEN)
			{
				kernel__3_tex_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_d, size_e, size_f, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), CEIL(size_d, SIZE_SLICE_1_D), dev_internal_offset_t2, dev_internal_offset_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
			}
			else
			{
				kernel__3_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_d, size_e, size_f, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), CEIL(size_d, SIZE_SLICE_1_D), stride_reg_x_1, stride_reg_y_1, size_internal);
			}
		}
		else
		{
			// [4] Internal Index: Partial
			// >>> External: Partial && Internal: Partial
			printf ("External: Partial, Internal: Partial\n");
			if (size_internal > MAX_CONST_LEN)
			{
				kernel__4_tex_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_d, size_e, size_f, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), CEIL(size_d, SIZE_SLICE_1_D), dev_internal_offset_t2, dev_internal_offset_v2, stride_reg_x_1, stride_reg_y_1, size_internal);
			}
			else
			{
				kernel__4_1<<<gridsize_1, blocksize_1>>>(dev_t3, dev_t2, dev_v2, size_a, size_b, size_c, size_d, size_e, size_f, CEIL(size_a, SIZE_SLICE_1_A), CEIL(size_b, SIZE_SLICE_1_B), CEIL(size_c, SIZE_SLICE_1_C), CEIL(size_d, SIZE_SLICE_1_D), stride_reg_x_1, stride_reg_y_1, size_internal);
			}
		}
	}

	// Copy the Result from Device to Host
	hipMemcpy(t3, dev_t3, sizeof(Complex) * (size_a * size_b * size_c * size_d), hipMemcpyDeviceToHost);

	// hipFree()
	hipFree(dev_t3);	hipFree(dev_t2);	hipFree(dev_v2);

	// Shoule be Fixed
	// HostFree

}

// This is written by tc_interface.tc_gen_code_interface()
// This Interface Should be Called to Run the Kernels
extern "C"
void sd_t_d2_fusion_(int size_a, int size_b, int size_c, int size_d, int size_e, int size_f, Complex* t3, Complex* t2, Complex* v2, int cond_kernel_1, int opt_register_transpose)
{
	// Pre-Processing for Split
	// Based on Tile-Sizes and Problem-Size
	// Currently, one index can be split into two indices

	// Call An Application
	sd_t_d2_fusion(size_a, size_b, size_c, size_d, size_e, size_f, t3, t2, v2, cond_kernel_1, opt_register_transpose);
}
//
//	Sample Code:
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#define DEBUG_CORRECTNESS
//#define DEBUG_SIMPLE_CORRECTNESS

void pre_Initializing_Input_Tensors();
void post_Correctness();

//
//  t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [a,e,b,f] * v2 [d,f,c,e];
//
int main(int argc, char** argv)
{
	// for sd2
	float *host_C, *host_C_chk;
	float *host_A;
	float *host_B;
	int size_idx_a, size_idx_b, size_idx_c, size_idx_d, size_idx_e, size_idx_f;

	// Problem Size
	size_idx_a = 16;
	size_idx_b = 16;
	size_idx_c = 16;
	size_idx_d = 16;
	size_idx_e = 16;
	size_idx_f = 16;
	
	//
	if (argc == 7)
	{
		size_idx_a = atoi(argv[1]);
		size_idx_b = atoi(argv[2]);
		size_idx_c = atoi(argv[3]);
		size_idx_d = atoi(argv[4]);
		size_idx_e = atoi(argv[5]);
		size_idx_f = atoi(argv[6]);
	}
	
	int size_C;
	int size_A;
	int size_B;
	int size_internal;

	//  t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [a,e,b,f] * v2 [d,f,c,e];
	size_internal 	= size_idx_e * size_idx_f;
	size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	size_A = size_idx_a * size_idx_e * size_idx_b * size_idx_f;
	size_B = size_idx_d * size_idx_f * size_idx_c * size_idx_e;

    //
	host_C 		= (float*)malloc(sizeof(float) * size_C);
	host_C_chk 	= (float*)malloc(sizeof(float) * size_C);
	host_A 		= (float*)malloc(sizeof(float) * size_A);
	host_B 		= (float*)malloc(sizeof(float) * size_B);
	
	printf ("==========================================================================================================\n");
    printf (">>> abcd-aebf-dfce\n");
    printf (">>> t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [a,e,b,f] * v2 [d,f,c,e];\n");
    printf (">>> Problem Size (a,b,c,d) and (e,f): (%2d,%2d,%2d,%2d) and (%2d,%2d)\n", size_idx_a, size_idx_b, size_idx_c, size_idx_d, size_idx_e, size_idx_f);
	printf ("==========================================================================================================\n");
	
	// Initialze "1" Output and "2 x 9" Inputs
	pre_Initializing_Input_Tensors(host_C, host_C_chk, size_C, host_A, size_A, host_B, size_B);
	
    // Run the Kernels
	sd_t_d2_fusion_(size_idx_a, size_idx_b, size_idx_c, size_idx_d, size_idx_e, size_idx_f, host_C, host_A, host_B, 1, -1);

#ifdef DEBUG_CORRECTNESS
	// Correctness-Check
	post_Correctness(host_C, host_C_chk,  host_A, host_B, size_idx_a, size_idx_b, size_idx_c, size_idx_d, size_idx_e, size_idx_f);
#endif

	// Free
	free(host_C);   free(host_C_chk);
	free(host_A);
	free(host_B);

	return 0;
}

// Initialize t3 (t3_temp), 9 t2 and 9 v2.
void pre_Initializing_Input_Tensors(float* h_C, float* h_C_chk, int size_C, float* h_A, int size_A, float* h_B, int size_B)
{
	// t3
	int i, j;
	for (i = 0; i < size_C; i++)
	{
		h_C[i] 	= 0.0;
		h_C_chk[i] = 0.0;
	}

	for (j = 0; j < size_A; j++)
	{
		h_A[j] = ((float)rand() / RAND_MAX);
	}

	for (j = 0; j < size_B; j++)
	{
		h_B[j] = ((float)rand() / RAND_MAX);
	}
}

//
void post_Correctness(float* h_C, float* h_C_chk, float* h_A, float* h_B, int size_idx_a, int size_idx_b, int size_idx_c, int size_idx_d, int size_idx_e, int size_idx_f)
{
    //  t3 [a,16,b,16,c,16,d,16] += sum(e,16,f,16) * t2 [a,e,b,f] * v2 [d,f,c,e];
    int size_C = size_idx_a * size_idx_b * size_idx_c * size_idx_d;
	
	long long int    tmp_ops = 0;
	int              ops     = 0;
    int idx_a, idx_b, idx_c, idx_d, idx_e, idx_f;
    for (idx_a = 0; idx_a < size_idx_a; idx_a++)
    for (idx_b = 0; idx_b < size_idx_b; idx_b++)
    for (idx_c = 0; idx_c < size_idx_c; idx_c++)
    for (idx_d = 0; idx_d < size_idx_d; idx_d++)
	{
		int tmp_r_idx = idx_a + (idx_b + (idx_c + (idx_d) * size_idx_c) * size_idx_b) * size_idx_a;

#ifdef DEBUG_SIMPLE_CORRECTNESS
        if (tmp_r_idx > 1024)
        break;
#endif

		for (idx_e = 0; idx_e < size_idx_e; idx_e++, ops = 0)
        {
            for (idx_f = 0; idx_f < size_idx_f; idx_f++)
            {   
                h_C_chk[tmp_r_idx] += 	h_A[idx_a + (idx_e + (idx_b + (idx_f) * size_idx_b) * size_idx_e) * size_idx_a] * 
                                        h_B[idx_d + (idx_f + (idx_c + (idx_e) * size_idx_c) * size_idx_f) * size_idx_d];
                ops++;
            }
            tmp_ops = tmp_ops + ops;
        }
	}

	printf ("======================================= Correctness Check ==========================================\n");
	float   epsilon = 0.00000001;
	int      diff    = 0;
	int      same    = 0;
	int 	 i;
	for (i = 0; i < size_C; i++)
	{
		float check = h_C_chk[i] - h_C[i];
		if (check < 0) check *= -1;
		if (check > epsilon)
		{
			diff++;
			if (diff < 8)
			printf ("Index: %5d, (Host) %8.4f, (Dev.) %8.4f >> (Diff.) %8.4f\n", i, h_C_chk[i], h_C[i], check);
		}
		else
		{
			same++;
		}
	}

	printf (" >>> PASSED: %'10d among %'10d in t3\n", same, size_C);
	printf (" >>> ERROR : %'10d among %'10d in t3\n", diff, size_C);
	printf (" >>> Total Operations: %'lld\n", tmp_ops * 2);
	printf ("====================================================================================================\n");
}

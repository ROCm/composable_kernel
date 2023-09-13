import jinja2

EXTRA_SHAPE_TEMPLATE = jinja2.Template(
    """
{{indent}}const int64_t stride_a = *a_dim1;
{{indent}}const int64_t stride_b = *b_dim1;
{{indent}}const int64_t stride_c = *c_dim1;
    ck::index_t M0 = M / G1 / G2;
    ck::index_t M1 = G1;
    ck::index_t M2 = G2;
    ck::index_t N0 = G3;
    ck::index_t N1 = N / G3;
    // GEMM shape
    //ck::index_t M = M0 * M1 * M2;
    //ck::index_t N = N0 * N1;
    //ck::index_t K = 128;
    //ck::index_t stride_A = K;
    //ck::index_t stride_B = K;
    // E = [M0, N0, M1, N1, M2]
    /* 0, 3, 1, 4, 2
    ck::index_t stride_E_M0 = N0 * M1 * N1 * M2;
    ck::index_t stride_E_M1 = N1 * M2;
    ck::index_t stride_E_M2 = 1;
    ck::index_t stride_E_N0 = M1 * N1 * M2;
    ck::index_t stride_E_N1 = M2;
    */
    // E = [M2, M0, N0, M1, N1] 2, 0, 3, 1, 4
    ck::index_t stride_E_M0 = N0* M1* N1;
    ck::index_t stride_E_M1 = N1;
    ck::index_t stride_E_M2 = M0* N0* M1* N1;
    ck::index_t stride_E_N0 = M1 * N1;
    ck::index_t stride_E_N1 = 1;
    // D = [0, N0, 0, N1, 0]
    ck::index_t stride_D_M0 = 0;
    ck::index_t stride_D_M1 = 0;
    ck::index_t stride_D_M2 = 0;
    ck::index_t stride_D_N0 = N1;
    ck::index_t stride_D_N1 = 1;
"""
)

output = EXTRA_SHAPE_TEMPLATE.render(indent=" ");
print (output)
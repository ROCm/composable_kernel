import re

def extract_template_parameters(template_str):
    # Extract everything inside the outermost <>
    match = re.search(r"<(.*)>", template_str, re.DOTALL)
    if not match:
        return []

    inside = match.group(1).strip()

    params = []
    current = []
    depth = 0  # track nested < >

    for char in inside:
        if char == '<':
            depth += 1
            current.append(char)
        elif char == '>':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            param = ''.join(current).strip()
            if param:
                params.append(param)
            current = []
        else:
            current.append(char)

    # Append last parameter if any
    if current:
        params.append(''.join(current).strip())

    return params


input_path = "inputkernel.txt"
output_path = "outputkernel_bwd_data.txt"

with open(input_path, 'r') as f:
    lines = f.readlines()

for line in lines:

    # Example usage
    #input_str = "        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,    64,    32,    64,    32,   8,   8,   32,   32,    1,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 4>,               8>"

    params = extract_template_parameters(line)

    NDimSpatial     = params[0]
    ALayout         = params[1]
    BLayout         = params[2]
    DsLayout        = params[3]
    ELayout         = params[4]

    ADataType       = params[5]
    BDataType       = params[6]
    AccDataType     = params[7]
    CshuffleDataType= params[8]
    DsDataTypes     = params[9]
    EDataType       = params[10]

    AElementwiseOp  = params[11]
    BElementwiseOp  = params[12]
    CElementwiseOp  = "PassThrough"#params[13]
    ConvFwdSpec     = params[14]

    DoPadGemmM      = params[15]
    DoPadGemmN      = params[16]
    NumGemmK        = params[17]

    BlockSize       = params[18]
    MPerBlock       = params[19]
    NPerBlock       = params[20]
    KPerBlock       = params[21]
    AK1             = params[22]
    BK1             = params[23]
    MPerXDL         = params[24]
    NPerXDL         = params[25]
    MXdlPerWave     = params[26]
    NXdlPerWave     = params[27]
    ABlockTransferClusterLengths        = params[28]
    ABlockTransferArrangeOrder          = params[29]
    ABlockTransferSrcAccessOrder        = params[30]
    ABlockTransferSrcVectorDim          = params[31]
    ABlockTransferSrcScalarPerVector    = params[32]
    ABlockTransferDstScalarPerVector_K1 = params[33]
    ABlockLdsAddExtraM                  = params[34]
    BBlockTransferClusterLengths        = params[35]
    BBlockTransferArrangeOrder          = params[36]
    BBlockTransferSrcVectorDim          = params[37]
    BBlockTransferSrcAccessOrder        = params[38]
    BBlockTransferSrcScalarPerVector    = params[39]
    BBlockTransferDstScalarPerVector_K1 = params[40]
    BBlockLdsAddExtraM                  = params[41]
    CShuffleMXdlPerwave                 = params[42]
    CShuffleNXdlPerWavePerShuffle       = params[43]
    CBlockTransferClusterLengths        = params[44]
    CBlockTransferScalarPerVector       = params[45]


    KBlockPerCu = 1
    MWarp = int(MPerBlock) // (int(MPerXDL) * int(MXdlPerWave))
    NWarp = int(NPerBlock) // (int(NPerXDL) * int(NXdlPerWave))
    KWarp = 1
    KPerXdl = 16 if MPerXDL == "32" else 32
    DoubleSMemBuffer = 'false'
    GemmPipelineVersion = "CK_TILE_PIPELINE_COMPUTE_V3"

    print(MPerBlock, NPerBlock, KPerBlock)

    pipelines = ["CK_TILE_PIPELINE_MEMORY", "CK_TILE_PIPELINE_COMPUTE_V3", "CK_TILE_PIPELINE_COMPUTE_V4"]

    for pipeline in pipelines:
        DoubleSMemBuffer = 'false' if pipeline != 'CK_TILE_PIPELINE_COMPUTE_V4' else 'true'
        with open(output_path, 'a') as f:
            f.write(f'GroupedConvolutionBackwardDataInvoker<{NDimSpatial},   {ALayout},   {BLayout},     {ELayout},   {ADataType},'
            f'{BDataType},   {EDataType},    {AElementwiseOp},       {BElementwiseOp},       {CElementwiseOp},'
            f'{KBlockPerCu},     {MPerBlock},      {NPerBlock},     {KPerBlock},     {MWarp},    {NWarp},    {KWarp},'
            f'{MPerXDL},     {NPerXDL},      {KPerXdl},      {ABlockTransferSrcScalarPerVector},    {BBlockTransferSrcScalarPerVector},'
            f'{CBlockTransferScalarPerVector}, {DoubleSMemBuffer}, {pipeline}>,\n')


# print(params[0])

# # Print each parameter as a separate variable
# for i, p in enumerate(params, start=1):
#     print(f"param_{i} = '{p}'")
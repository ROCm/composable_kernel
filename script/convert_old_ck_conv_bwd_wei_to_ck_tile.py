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
output_path = "outputkernel_bwd_wei.txt"

with open(input_path, 'r') as f:
    lines = f.readlines()

for line in lines:

    # Example usage
    #input_str = "        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,    64,    32,    64,    32,   8,   8,   32,   32,    1,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 4>,               8>"

    params = extract_template_parameters(line)

    NDimSpatial     = params[0]
    ALayout         = params[1]
    BLayout         = params[2]
    ELayout         = params[3]

    ADataType       = params[4]
    BDataType       = params[5]
    EDataType       = params[6]
    AccDataType     = params[7]

    AElementwiseOp  = params[8]
    BElementwiseOp  = params[9]
    CElementwiseOp  = "PassThrough"#params[13]
    ConvFwdSpec     = params[11]

    BlockSize       = params[12]
    MPerBlock       = params[13]
    NPerBlock       = params[14]
    KPerBlock       = params[15]
    K1              = params[16]
    MPerXDL         = params[17]
    NPerXDL         = params[18]
    MXdlPerWave     = params[19]
    NXdlPerWave     = params[20]
    ABlockTransferClusterLengths        = params[21]
    ABlockTransferArrangeOrder          = params[22]
    ABlockTransferSrcAccessOrder        = params[23]
    ABlockTransferSrcVectorDim          = params[24]
    ABlockTransferSrcScalarPerVector    = params[25]
    ABlockTransferDstScalarPerVector_K1 = params[26]
    ABlockLdsAddExtraM                  = params[27]
    BBlockTransferClusterLengths        = params[28]
    BBlockTransferArrangeOrder          = params[29]
    BBlockTransferSrcVectorDim          = params[30]
    BBlockTransferSrcAccessOrder        = params[31]
    BBlockTransferSrcScalarPerVector    = params[32]
    BBlockTransferDstScalarPerVector_K1 = params[33]
    BBlockLdsAddExtraM                  = params[34]
    CShuffleMXdlPerwave                 = params[35]
    CShuffleNXdlPerWavePerShuffle       = params[36]
    CBlockTransferClusterLengths        = params[37]
    CBlockTransferScalarPerVector       = params[38]


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
            f.write(f'GroupedConvolutionBackwardWeightInvoker<{NDimSpatial},   {ALayout},   {BLayout},     {ELayout},   {ADataType},'
            f'{BDataType},   {EDataType},    {AElementwiseOp},       {BElementwiseOp},       {CElementwiseOp},'
            f'{KBlockPerCu},     {MPerBlock},      {NPerBlock},     {KPerBlock},     {MWarp},    {NWarp},    {KWarp},'
            f'{MPerXDL},     {NPerXDL},      {KPerXdl},      {ABlockTransferSrcScalarPerVector},    {BBlockTransferSrcScalarPerVector},'
            f'{CBlockTransferScalarPerVector}, {DoubleSMemBuffer}, {pipeline}>,\n')


# print(params[0])

# # Print each parameter as a separate variable
# for i, p in enumerate(params, start=1):
#     print(f"param_{i} = '{p}'")
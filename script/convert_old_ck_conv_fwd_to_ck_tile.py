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


input_path = "inputkernel_fwd.txt"
output_path = "outputkernel_fwd.txt"

with open(input_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    print(1)
    params = extract_template_parameters(line)
    print(1)
    NDimSpatial     = params[0]
    ALayout         = params[1]
    BLayout         = params[2]
    DsLayout        = params[3]
    ELayout         = params[4]
    ADataType       = params[5]
    BDataType       = params[6]
    AccDataType     = params[7]
    CShuffleDataType= params[8]
    DsDataTypes     = params[9]
    EDataType       = params[10]
    AElementwiseOp  = params[11]
    BElementwiseOp  = params[12]
    CElementwiseOp  = "PassThrough"#params[13]
    ConvFwdSpec     = params[14]
    GemmSpec        = params[15]
    NummGemmKPref   = params[16]
    BlockSize       = params[17]
    MPerBlock       = params[18]
    NPerBlock       = params[19]
    KPerBlock       = params[20]
    AK1             = params[21]
    BK1             = params[22]
    MPerXDL         = params[23]
    NPerXDL         = params[24]
    MXdlPerWave     = params[25]
    NXdlPerWave     = params[26]
    ABlockTransferClusterLengths        = params[27]
    ABlockTransferArrangeOrder          = params[28]
    ABlockTransferSrcAccessOrder        = params[29]
    ABlockTransferSrcVectorDim          = params[30]
    ABlockTransferSrcScalarPerVector    = params[31]
    ABlockTransferDstScalarPerVector_K1 = params[32]
    ABlockLdsAddExtraM                  = params[33]
    BBlockTransferClusterLengths        = params[34]
    BBlockTransferArrangeOrder          = params[35]
    BBlockTransferSrcVectorDim          = params[36]
    BBlockTransferSrcAccessOrder        = params[37]
    BBlockTransferSrcScalarPerVector    = params[38]
    BBlockTransferDstScalarPerVector_K1 = params[39]
    BBlockLdsAddExtraM                  = params[40]
    CShuffleMXdlPerwave                 = params[41]
    CShuffleNXdlPerWavePerShuffle       = params[42]
    CBlockTransferClusterLengths        = params[43]
    CBlockTransferScalarPerVector       = params[44]

    print(1)
    KBlockPerCu = 1
    MWarp = int(MPerBlock) // (int(MPerXDL) * int(MXdlPerWave))
    NWarp = int(NPerBlock) // (int(NPerXDL) * int(NXdlPerWave))
    KWarp = 1
    KPerXdl = 16 if MPerXDL == "32" else 32
    DoubleSMemBuffer = 'false'
    GemmPipelineVersion = "CK_TILE_PIPELINE_COMPUTE_V3"

    pipelines = ["CK_TILE_PIPELINE_MEMORY", "CK_TILE_PIPELINE_COMPUTE_V3", "CK_TILE_PIPELINE_COMPUTE_V4"]
    convspecs = ["Filter1x1Stride1Pad0", "Filter1x1Pad0", "Filter3x3", "Default"]

    for pipeline in pipelines:
        print(1)
        for convSpec in convspecs:
            DoubleSMemBuffer = 'false' if pipeline != 'CK_TILE_PIPELINE_COMPUTE_V4' else 'true'
            with open(output_path, 'a') as f:
                f.write(f'GroupedConvolutionForwardInvoker<{NDimSpatial},   {ALayout},   {BLayout},     {ELayout},   {ADataType},'
                f'{BDataType},   {EDataType},    {AElementwiseOp},       {BElementwiseOp},       {CElementwiseOp}, ConvolutionSpecialization::{convSpec}'
                f'{KBlockPerCu},     {MPerBlock},      {NPerBlock},     {KPerBlock},     {MWarp},    {NWarp},    {KWarp},'
                f'{MPerXDL},     {NPerXDL},      {KPerXdl},      {ABlockTransferSrcScalarPerVector},    {BBlockTransferSrcScalarPerVector},'
                f'{CBlockTransferScalarPerVector}, {DoubleSMemBuffer}, {pipeline}>,\n')

            print(1)


# print(params[0])

# # Print each parameter as a separate variable
# for i, p in enumerate(params, start=1):
#     print(f"param_{i} = '{p}'")
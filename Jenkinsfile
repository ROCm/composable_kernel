// Composable Kernel Jenkins Pipeline
//
// SMART BUILD SYSTEM:
// This pipeline uses intelligent dependency analysis to speed up PR builds while
// maintaining full validation on nightly runs.
//
// How it works:
// 1. PR Builds (Selective):
//    - Configure: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON (~30s)
//    - Analyze: Parse compile_commands.json + clang -MM for dependencies (~2min)
//    - Select: git diff to find affected tests (~1s)
//    - Build: ninja <affected-tests> only (minutes vs hours)
//    - Test: ctest -R <affected-pattern>
//
// 2. Nightly Builds (Full):
//    - FORCE_CI=true from cron triggers full build
//    - All targets built and tested for validation
//
// 3. Safety Checks:
//    - Forces full build if CMake configuration changes
//    - Forces full build if dependency cache stale (>7 days)
//    - Manual override: set DISABLE_SMART_BUILD=true
//
// Benefits: PR builds 5h -> 30min (typical), nightly builds unchanged
// See: script/dependency-parser/README.md for details
//
def rocmnode(name) {
    return '(rocmtest || miopen) && (' + name + ')'
}

def loadCk() {
    def branch = (params.USE_CURRENT_BRANCH_FOR_CK_GROOVY
        ? (env.CHANGE_BRANCH ?: env.BRANCH_NAME)
        : 'develop')
    library("ck@${branch}")
}

//launch develop branch daily jobs
CRON_SETTINGS = BRANCH_NAME == "develop" ? '''0 23 * * * % RUN_FULL_QA=true;RUN_CK_TILE_FMHA_TESTS=true;RUN_PERFORMANCE_TESTS=true;FORCE_CI=true
                                              0 22 * * * % RUN_FULL_QA=true;DISABLE_DL_KERNELS=true;RUN_TILE_ENGINE_BASIC_TESTS=true;RUN_TILE_ENGINE_GEMM_TESTS=true;RUN_PERFORMANCE_TESTS=true;RUN_ALL_UNIT_TESTS=true;FORCE_CI=true
                                              0 21 * * * % RUN_GROUPED_CONV_LARGE_CASES_TESTS=true;hipTensor_test=true;BUILD_GFX101=false;BUILD_GFX908=false;BUILD_GFX942=true;BUILD_GFX950=true;RUN_PERFORMANCE_TESTS=true;RUN_ALL_UNIT_TESTS=true;FORCE_CI=true;BUILD_PACKAGES=true
                                              0 19 * * * % BUILD_DOCKER=true;COMPILER_VERSION=develop;BUILD_COMPILER=/llvm-project/build/bin/clang++;USE_SCCACHE=false;NINJA_BUILD_TRACE=true;RUN_ALL_UNIT_TESTS=true;FORCE_CI=true
                                              0 17 * * * % BUILD_DOCKER=true;COMPILER_VERSION=therock;USE_SCCACHE=false;NINJA_BUILD_TRACE=true;RUN_ALL_UNIT_TESTS=true;FORCE_CI=true
                                              0 15 * * * % BUILD_DOCKER=true;COMPILER_VERSION=amd-staging;BUILD_COMPILER=/llvm-project/build/bin/clang++;USE_SCCACHE=false;NINJA_BUILD_TRACE=true;RUN_ALL_UNIT_TESTS=true;FORCE_CI=true
                                              0 13 * * * % BUILD_INSTANCES_ONLY=true;USE_SCCACHE=false;NINJA_BUILD_TRACE=true;FORCE_CI=true
                                              0 11 * * * % RUN_FULL_CONV_TILE_TESTS=true;RUN_AITER_TESTS=true;RUN_FA_TESTS=true;USE_SCCACHE=false;RUN_PERFORMANCE_TESTS=false;FORCE_CI=true
                                              0 9 * * * % RUN_PYTORCH_TESTS=true;USE_SCCACHE=false;RUN_PERFORMANCE_TESTS=false;BUILD_GFX101=false;BUILD_GFX103=false;BUILD_GFX11=false;BUILD_GFX12=false;BUILD_GFX90A=false;FORCE_CI=true''' : ""
CURRENT_BRANCH_NAME = env.CHANGE_ID ? "refs/pull/${env.CHANGE_ID}/head" : (env.CHANGE_BRANCH ? env.CHANGE_BRANCH : env.BRANCH_NAME)

POLL_SPEC = BRANCH_NAME == "develop" ? 'H H/6 * * *' : ''

pipeline {
    agent none
    triggers {
        parameterizedCron(CRON_SETTINGS)
        pollSCM(POLL_SPEC)
    }
    options {
        skipDefaultCheckout()
        parallelsAlwaysFailFast()
    }
    parameters {
        booleanParam(
            name: "BUILD_DOCKER",
            defaultValue: false,
            description: "Force building docker image (default: false), set to true if docker image needs to be updated.")
        string(
            name: 'USE_CUSTOM_DOCKER',
            defaultValue: '',
            description: 'If you want to use a custom docker image, please specify it here (default: leave blank).')
        string(
            name: 'ROCMVERSION',
            defaultValue: '7.13',
            description: 'Specify which ROCM version to use: 7.13 (default).')
        string(
            name: 'COMPILER_VERSION',
            defaultValue: '',
            description: 'Specify which version of compiler to use: develop, amd-staging, therock, or leave blank (default).')
        string(
            name: 'COMPILER_COMMIT',
            defaultValue: '',
            description: 'Specify which commit of compiler branch to use: leave blank to use the latest commit (default), or use some specific commit of llvm-project branch.')
        string(
            name: 'BUILD_COMPILER',
            defaultValue: '/opt/rocm/llvm/bin/clang++',
            description: 'Build CK with /opt/rocm/bin/hipcc, /llvm-project/build/bin/clang++, or with /opt/rocm/llvm/bin/clang++ (default).')
        booleanParam(
            name: "RUN_FULL_QA",
            defaultValue: false,
            description: "Select whether to run small set of performance tests (default) or full QA")
        booleanParam(
            name: "DISABLE_DL_KERNELS",
            defaultValue: false,
            description: "Select whether to build DL kernels (default: OFF)")
        booleanParam(
            name: "hipTensor_test",
            defaultValue: false,
            description: "Use the CK build to verify hipTensor build and tests (default: OFF)")
        string(
            name: 'hipTensor_branch',
            defaultValue: 'develop',
            description: 'Specify which branch of hipTensor to use (default: develop)')
        booleanParam(
            name: "USE_SCCACHE",
            defaultValue: true,
            description: "Use the sccache for building CK (default: ON)")
        booleanParam(
            name: "DISABLE_SMART_BUILD",
            defaultValue: false,
            description: "Disable smart build system and force full build/test (default: OFF). Smart build uses pre-build dependency analysis for selective testing on PRs, full builds on nightly runs.")
        booleanParam(
            name: "RUN_CPPCHECK",
            defaultValue: false,
            description: "Run the cppcheck static analysis (default: OFF)")
        booleanParam(
            name: "RUN_PERFORMANCE_TESTS",
            defaultValue: false,
            description: "Run the performance tests (default: OFF)")
        booleanParam(
            name: "RUN_GROUPED_CONV_LARGE_CASES_TESTS",
            defaultValue: false,
            description: "Run the grouped conv large cases tests (default: OFF)")
        booleanParam(
            name: "RUN_CONV_COMPREHENSIVE_DATASET",
            defaultValue: false,
            description: "Run comprehensive convolution dataset tests before important changes (default: OFF)")
        booleanParam(
            name: "RUN_CK_TILE_FMHA_TESTS",
            defaultValue: false,
            description: "Run the ck_tile FMHA tests (default: OFF)")
        booleanParam(
            name: "RUN_TILE_ENGINE_BASIC_TESTS",
            defaultValue: true,
            description: "Run the tile_engine_basic tests (default: ON)")
        booleanParam(
            name: "RUN_TILE_ENGINE_GEMM_TESTS",
            defaultValue: false,
            description: "Run the tile_engine_gemm tests (default: OFF)")
        booleanParam(
            name: "BUILD_INSTANCES_ONLY",
            defaultValue: false,
            description: "Test building instances for various architectures simultaneously (default: OFF)")
        booleanParam(
            name: "BUILD_PACKAGES",
            defaultValue: false,
            description: "Build packages for the libraries and/or ckProfiler (default: OFF)")
        booleanParam(
            name: "BUILD_GFX908",
            defaultValue: false,
            description: "Build CK and run tests on gfx908 (default: OFF)")
        booleanParam(
            name: "BUILD_GFX90A",
            defaultValue: true,
            description: "Build CK and run tests on gfx90a (default: ON)")
        booleanParam(
            name: "BUILD_GFX942",
            defaultValue: true,
            description: "Build CK and run tests on gfx942 (default: ON)")
        booleanParam(
            name: "BUILD_GFX950",
            defaultValue: true,
            description: "Build CK and run tests on gfx950 (default: ON)")
        booleanParam(
            name: "BUILD_GFX101",
            defaultValue: false,
            description: "Build CK and run tests on gfx101 (default: OFF)")
        booleanParam(
            name: "BUILD_GFX103",
            defaultValue: false,
            description: "Build CK and run tests on gfx103 (default: OFF)")
        booleanParam(
            name: "BUILD_GFX11",
            defaultValue: true,
            description: "Build CK and run tests on gfx11 (default: ON)")
        booleanParam(
            name: "BUILD_GFX12",
            defaultValue: true,
            description: "Build CK and run tests on gfx12 (default: ON)")
        booleanParam(
            name: "BUILD_GFX1250",
            defaultValue: true,
            description: "Build CK for gfx1250 (default: ON)")
        booleanParam(
            name: "NINJA_BUILD_TRACE",
            defaultValue: true,
            description: "Generate a ninja build trace (default: ON)")
        booleanParam(
            name: "NINJA_FTIME_TRACE",
            defaultValue: false,
            description: "Generate a detailed time trace (default: OFF)")
        booleanParam(
            name: "RUN_INDUCTOR_TESTS",
            defaultValue: true,
            description: "Run inductor codegen tests (default: ON)")
        booleanParam(
            name: "RUN_CODEGEN_TESTS",
            defaultValue: true,
            description: "Run codegen tests (default: ON)")
        booleanParam(
            name: "RUN_BUILDER_TESTS",
            defaultValue: false,
            description: "Run CK_BUILDER tests (default: OFF)")
        booleanParam(
            name: "RUN_ROCM_CK_TESTS",
            defaultValue: true,
            description: "Run rocm_ck tests (default: ON)")
        booleanParam(
            name: "RUN_ALL_UNIT_TESTS",
            defaultValue: false,
            description: "Run all unit tests (default: OFF)")
        booleanParam(
            name: "RUN_PYTORCH_TESTS",
            defaultValue: false,
            description: "Try building PYTORCH with latest CK develop branch (default: OFF)")
        string(
            name: 'ck_pytorch_branch',
            defaultValue: CURRENT_BRANCH_NAME,
            description: 'Specify which branch of CK to test with Pytorch (default: current branch)')
        booleanParam(
            name: "RUN_AITER_TESTS",
            defaultValue: false,
            description: "Run AITER tests with latest CK develop branch (default: OFF)")
        booleanParam(
            name: "RUN_FULL_CONV_TILE_TESTS",
            defaultValue: false,
            description: "Run CK Tile grouped convolution tests with latest CK develop branch (default: OFF)")
        string(
            name: 'aiter_branch',
            defaultValue: 'main',
            description: 'Specify which branch of AITER to use (default: main)')
        string(
            name: 'ck_aiter_branch',
            defaultValue: CURRENT_BRANCH_NAME,
            description: 'Specify which branch of CK to test with AITER (default: current branch)')
        booleanParam(
            name: "RUN_FA_TESTS",
            defaultValue: false,
            description: "Run Flash Attention tests with latest CK develop branch (default: OFF)")
        string(
            name: 'fa_base_docker',
            defaultValue: 'rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1',
            description: 'Specify which base docker image to use for flash-attention tests')
        string(
            name: 'fa_branch',
            defaultValue: 'ck_improve_main',
            description: 'Specify which branch of flash-attention to use (default: ck_improve_main)')
        string(
            name: 'ck_fa_branch',
            defaultValue: CURRENT_BRANCH_NAME,
            description: 'Specify which branch of CK to test with flash-attention (default: current branch)')
        booleanParam(
            name: "FORCE_CI",
            defaultValue: false,
            description: "Force CI to run even when only non-relevant files are changed (default: OFF)")
        booleanParam(
            name: 'USE_CURRENT_BRANCH_FOR_CK_GROOVY',
            defaultValue: false,
            description: 'Load ck.groovy from the current branch instead of develop. Enable when testing pipeline changes (default: OFF).')
    }
    environment{
        dbuser = "${dbuser}"
        dbpassword = "${dbpassword}"
        dbsship = "${dbsship}"
        dbsshport = "${dbsshport}"
        dbsshuser = "${dbsshuser}"
        dbsshpassword = "${dbsshpassword}"
        gerrit_cred="${gerrit_cred}"
        DOCKER_BUILDKIT = "1"
        BUILD_GFX103 = "${env.BRANCH_NAME == 'develop' ? true : false}"
    }
    stages{
        stage("Determine CI Execution") {
            agent{ label rocmnode("nogpu") }
            steps {
                script {
                    loadCk()
                    ck.checkoutComposableKernel()
                    env.SHOULD_RUN_CI = String.valueOf(params.FORCE_CI.toBoolean() || ck.shouldRunCICheck())
                    echo "SHOULD_RUN_CI: ${env.SHOULD_RUN_CI}"
                }
            }
        }
        stage("Build Docker"){
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel{
                stage('Docker /opt/rocm'){
                    agent{ label rocmnode("nogpu") }
                    steps{
                        deleteDir()
                        script {
                            loadCk()
                            ck.buildDocker('/opt/rocm')
                        }
                        cleanWs()
                    }
                }
            }
        }
        stage("Static checks") {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel{
                stage('Clang Format and Cppcheck') {
                    when {
                        beforeAgent true
                        expression { params.RUN_CPPCHECK.toBoolean() }
                    }
                    agent{ label rocmnode("nogpu") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runClangFormatAndCppcheck() }
                        archiveArtifacts "build/ck_cppcheck.log"
                        cleanWs()
                    }
                }
                stage('Clang Format') {
                    when {
                        beforeAgent true
                        expression { !params.RUN_CPPCHECK.toBoolean() }
                    }
                    agent{ label rocmnode("nogpu") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runClangFormat() }
                        cleanWs()
                    }
                }
                stage('ASCII Only Check') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_cmd = """cd .. && \
                                find . -type f \\( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.h.in' -o -name '*.hpp.in' -o -name '*.cpp.in' -o -name '*.inc' -o -name '*.cl' \\) \
                                -not -path '*/build/*' -not -path '*/include/rapidjson/*' \
                                -print0 | xargs -0 -P 8 -n 64 script/check_ascii_only.sh"""
                    }
                    steps{
                        deleteDir()
                        script {
                            loadCk();
                            ck.buildHipClangJobAndReboot(setup_args:setup_args, setup_cmd: "", build_cmd: "", execute_cmd: execute_cmd)
                        }
                        cleanWs()
                    }
                }
            }
        }
         stage("Run Downstream Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run Pytorch Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_PYTORCH_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942")}
                    steps{
                        script {
                            loadCk()
                            ck.run_downstream_tests(image: "${env.CK_PYTORCH_IMAGE}", timeoutHours: 2, execute_cmds: ck.getPytorchTestsCmds())
                        }
                        cleanWs()
                    }
                }
                stage("Run AITER Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_AITER_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942")}
                    steps{
                        script {
                            loadCk()
                            ck.run_downstream_tests(image: "${env.CK_AITER_IMAGE}", timeoutHours: 5, execute_cmds: ck.getAiterTestsCmds())
                        }
                        cleanWs()
                    }
                }
                stage("Run AITER Tests on gfx950")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_AITER_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx950")}
                    steps{
                        script {
                            loadCk()
                            ck.run_downstream_tests(image: "${env.CK_AITER_IMAGE}", timeoutHours: 5, execute_cmds: ck.getAiterTestsCmds())
                        }
                        cleanWs()
                    }
                }
                stage("Run FA Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_FA_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942")}
                    steps{
                        script {
                            loadCk()
                            ck.run_downstream_tests(image: "${env.CK_FA_IMAGE}", timeoutHours: 5, execute_cmds: ck.getFaTestsCmds())
                        }
                        cleanWs()
                    }
                }
                stage("Run FA Tests on gfx950")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_FA_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx950")}
                    steps{
                        script {
                            loadCk()
                            ck.run_downstream_tests(image: "${env.CK_FA_IMAGE}", timeoutHours: 5, execute_cmds: ck.getFaTestsCmds())
                        }
                        cleanWs()
                    }
                }
            }
        }
        stage("Run Full Grouped Conv Tile Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run Full Grouped Conv Tile Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_FULL_CONV_TILE_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a")}
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runFullGroupedConvTileTests() }
                        cleanWs()
                    }
                }
            }
        }
        stage("Run Grouped Conv Large Case Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run Grouped Conv Large Case Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_GROUPED_CONV_LARGE_CASES_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a")}
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runGroupedConvLargeCaseTests() }
                        cleanWs()
                    }
                }
            }
        }
        stage("Run Comprehensive Convolution Dataset Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run Comprehensive Dataset Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CONV_COMPREHENSIVE_DATASET.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a")}
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runComprehensiveConvDatasetTests() }
                        cleanWs()
                    }
                }
            }
        }
        stage("Run CK_TILE_FMHA Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run CK_TILE_FMHA Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CK_TILE_FMHA_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = ck.build_and_run_fmha("gfx90a")
                    }
                    steps{
                        deleteDir()
                        script {
                            loadCk()
                            ck.buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        }
                        cleanWs()
                    }
                }
                stage("Run CK_TILE_FMHA Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CK_TILE_FMHA_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = ck.build_and_run_fmha("gfx942")
                    }
                    steps{
                        deleteDir()
                        script {
                            loadCk()
                            ck.buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        }
                        cleanWs()
                    }
                }
                stage("Run CK_TILE_FMHA Tests on gfx950")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CK_TILE_FMHA_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx950") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = ck.build_and_run_fmha("gfx950")
                    }
                    steps{
                        deleteDir()
                        script {
                            loadCk()
                            ck.buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        }
                        cleanWs()
                    }
                }
                stage("Run CK_TILE_FMHA Tests on gfx1201")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CK_TILE_FMHA_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1201") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = ck.build_and_run_fmha("gfx1201")
                    }
                    steps{
                        deleteDir()
                        script {
                            loadCk()
                            ck.buildHipClangJobAndReboot(setup_args:setup_args, build_type: 'Release', execute_cmd: execute_args)
                        }
                        cleanWs()
                    }
                }
            }
        }
        stage("Run TILE_ENGINE_BASIC Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run TILE_ENGINE_BASIC Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_TILE_ENGINE_BASIC_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runTileEngineBasicTests(params.BUILD_COMPILER) }
                        cleanWs()
                    }
                }
            }
        }
        stage("Run TILE_ENGINE_GEMM Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Run TILE_ENGINE_GEMM Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_TILE_ENGINE_GEMM_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runTileEngineGemmTests("gfx942", params.BUILD_COMPILER) }
                        cleanWs()
                    }
                }
                stage("Run TILE_ENGINE_GEMM Tests on gfx950")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_TILE_ENGINE_GEMM_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx950") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runTileEngineGemmTests("gfx950", params.BUILD_COMPILER) }
                        cleanWs()
                    }
                }
                stage("Run TILE_ENGINE_GEMM Tests on gfx1201")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_TILE_ENGINE_GEMM_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1201") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runTileEngineGemmTests("gfx1201", params.BUILD_COMPILER) }
                        cleanWs()
                    }
                }
            }
        }

		stage("Build CK and run Tests")
        {
            when {
                beforeAgent true
                expression { env.SHOULD_RUN_CI.toBoolean() }
            }
            parallel
            {
                stage("Build CK and run Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { (params.BUILD_GFX942.toBoolean() || params.RUN_FULL_QA.toBoolean()) && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runBuildCKAndTests("gfx942") }
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx950")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX950.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label rocmnode("gfx950") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runBuildCKAndTests("gfx950") }
                        cleanWs()
                    }
                }
                /*
                stage("Build CK and run Tests on gfx908")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX908.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label rocmnode("gfx908") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runBuildCKAndTests("gfx908") }
                        cleanWs()
                    }
                }
                */
                stage("Build CK and run Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX90A.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runBuildCKAndTests("gfx90a") }
                        cleanWs()
                    }
                }
                stage("Build CK instances for all supported targets")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_INSTANCES_ONLY.toBoolean() && !params.RUN_FULL_QA.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                                            -DCMAKE_CXX_COMPILER="${params.BUILD_COMPILER}" \
                                            -DCMAKE_HIP_COMPILER="${params.BUILD_COMPILER}" \
                                            -DGPU_ARCHS="gfx908;gfx90a;gfx942;gfx950;gfx10-3-generic;gfx11-generic;gfx12-generic" \
                                            -D CMAKE_BUILD_TYPE=Release .. && ninja -j64 """
                    }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runBuildInstancesOnly(params.BUILD_COMPILER) }
                        cleanWs()
                    }
                }
                /*
                stage("Build CK and run Tests on gfx1010")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX101.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1010") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runBuildCKAndTests("gfx10-1-generic") }
                        cleanWs()
                    }
                }
                */
                stage("Build CK and run Tests on gfx1030")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX103.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1030") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runBuildCKAndTests("gfx10-3-generic") }
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx11")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX11.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label 'miopen && (gfx1101 || gfx1100)' }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runBuildCKAndTests("gfx11-generic") }
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx1201")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX12.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1201") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runBuildCKAndTests("gfx12-generic") }
                        cleanWs()
                    }
                }
                stage("Build CK for gfx1250")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX1250.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    steps{
                        deleteDir()
                        script { loadCk(); ck.runBuildCKAndTests("gfx1250") }
                        cleanWs()
                    }
                }
            }
            post {
                always {
                    node(rocmnode("nogpu")) {
                        script {
                            loadCk()
                            // Simulate capture
                            ck.generateAndArchiveBuildTraceVisualization("ck_build_trace_gfx11.json")
                            ck.generateAndArchiveBuildTraceVisualization("ck_build_trace_gfx12.json")
                            ck.generateAndArchiveBuildTraceVisualization("ck_build_trace_gfx90a.json")
                            ck.generateAndArchiveBuildTraceVisualization("ck_build_trace_gfx942.json")
                            ck.generateAndArchiveBuildTraceVisualization("ck_build_trace_gfx950.json")
                        }
                        cleanWs()
                    }
                }
                success {
                    script {
                        node(rocmnode("nogpu")) {
                            loadCk()
                            // Report the parent stage build ck and run tests status
                            ck.setGithubStatus("${env.STAGE_NAME}", 'success', "Stage ${env.STAGE_NAME} passed")
                            echo "Reporting success status for build ck and run tests"
                        }
                    }
                }
            }
        }
        stage("Process Performance Test Results")
        {
            parallel
            {
                stage("Process results"){
                    when {
                        beforeAgent true
                        expression { (params.RUN_PERFORMANCE_TESTS.toBoolean() || params.BUILD_INSTANCES_ONLY.toBoolean() || params.RUN_CK_TILE_FMHA_TESTS.toBoolean()|| params.BUILD_PACKAGES.toBoolean()) }
                    }
                    agent { label 'mici' }
                    steps{
                        deleteDir()
                        script {
                            loadCk()
                            ck.process_results()
                        }
                        cleanWs()
                    }
                }
            }
            post {
                success {
                    script {
                        node(rocmnode("nogpu")) {
                            loadCk()
                            // Report the skipped parent's stage status
                            ck.setGithubStatus("${env.STAGE_NAME}", 'success', "Stage ${env.STAGE_NAME} passed")
                            echo "Process Performance Test Results stage skipped."
                        }
                    }
                }
            }
        }
    }
    post {
        success {
            script {
                node(rocmnode("nogpu")) {
                    loadCk()
                    ck.setGithubStatus('Math CI Summary', 'success', "Math CI passed")
                }
            }
        }
        failure {
            script {
                node(rocmnode("nogpu")) {
                    loadCk()
                    ck.setGithubStatus('Math CI Summary', 'failure', "Math CI failed")
                    ck.checkoutComposableKernel()
                    withCredentials([string(credentialsId: 'ck_ci_errors_webhook_url', variable: 'WEBHOOK_URL')]) {
                        sh 'bash projects/composablekernel/script/infra_helper/send_failure_notifications.sh'
                    }
                }
            }
        }
    }
}


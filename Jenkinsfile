def rocmnode(name) {
    return '(rocmtest || miopen) && (' + name + ')'
}

def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME"
        lsb_release -sd
        uname -r
        cat /sys/module/amdgpu/version
        ls /opt/ -la
    """
}

class Version {
    int major, minor, patch
    @Override
    String toString() {
        return [major, minor, patch].findAll().join('.')
    }
}
def parseVersion(String versionString) {
    if (!versionString) return null
    int[] tokens = versionString.split(/\./).collect { it as int } // Splits the string by '.' and converts each part to an integer.
    return new Version(
        major: tokens[0],
        minor: tokens.length > 1 ? tokens[1] : null,
        patch: tokens.length > 2 ? tokens[2] : null,
    )
}

def nthreads() {
    def nproc = sh(returnStdout: true, script: 'nproc')
    echo "Number of cores: ${nproc}"
    def n = nproc.toInteger()
    if (n > 32){
        n /= 2
    }
    if (n > 64){
        n = 64
    }
    echo "Number of threads used for building: ${n}"
    return n
}

def runShell(String command){
    def responseCode = sh returnStatus: true, script: "${command} > tmp.txt"
    def output = readFile(file: "tmp.txt")
    return (output != "")
}

def getBaseDockerImageName(){
    def img
    if (params.USE_CUSTOM_DOCKER != ""){
        img = "${params.USE_CUSTOM_DOCKER}"
    }
    else{
        def ROCM_numeric = parseVersion("${params.ROCMVERSION}")
        if ( ROCM_numeric.major <= 6 && ROCM_numeric.minor < 5 ){
            img = "${env.CK_DOCKERHUB}:ck_ub24.04_rocm${params.ROCMVERSION}"
            }
        else{
            img = "${env.CK_DOCKERHUB_PRIVATE}:ck_ub24.04_rocm${params.ROCMVERSION}"
            }
        }
    return img
}

def getDockerImageName(){
    def img
    def base_name = getBaseDockerImageName()
    if (params.USE_CUSTOM_DOCKER != ""){
        img = "${params.USE_CUSTOM_DOCKER}"
    }
    else{
       if (params.COMPILER_VERSION == "") {
           img = "${base_name}"
       }
       else{
          if (params.COMPILER_COMMIT == ""){
             img = "${base_name}_${params.COMPILER_VERSION}"
          }
          else{
             def commit = "${params.COMPILER_COMMIT}"[0..6]
             img = "${base_name}_${params.COMPILER_VERSION}_${commit}"
          }
       }
    }
    return img
}

def check_host() {
    if ("${env.CK_SCCACHE}" != "null"){
        def SCCACHE_SERVER="${env.CK_SCCACHE.split(':')[0]}"
        echo "sccache server: ${SCCACHE_SERVER}"
        sh "chmod +w -R ${env.WORKSPACE}"
        sh '''ping -c 1 -p 6379 "${SCCACHE_SERVER}" | echo $? > tmp.txt'''
        def output = readFile(file: "tmp.txt")
        echo "tmp.txt contents: \$output"
        return (output != "0")
    }
    else{
        return 1
    }
}

def build_compiler(){
    def compiler
    compiler = "${params.BUILD_COMPILER}"
    return compiler
}

def check_arch(){
    def arch_type = 0
    sh 'rocminfo | tee rocminfo.log'
    if ( runShell('grep -n "gfx90a" rocminfo.log') ){
        arch_type = 1
    }
    else if ( runShell('grep -n "gfx942" rocminfo.log') ) {
        arch_type = 2
    }
    else if ( runShell('grep -n "gfx10" rocminfo.log') ) {
        arch_type = 3
    }
    else if ( runShell('grep -n "gfx11" rocminfo.log') ) {
        arch_type = 4
    }
    else if ( runShell('grep -n "gfx12" rocminfo.log') ) {
        arch_type = 5
    }
    else if ( runShell('grep -n "gfx908" rocminfo.log') ) {
        arch_type = 6
    }
    else if ( runShell('grep -n "gfx950" rocminfo.log') ) {
        arch_type = 7
    }
    return arch_type
}

def getDockerImage(Map conf=[:]){
    env.DOCKER_BUILDKIT=1
    def prefixpath = conf.get("prefixpath", "/opt/rocm")
    def no_cache = conf.get("no_cache", false)
    def dockerArgs = "--build-arg BUILDKIT_INLINE_CACHE=1 --build-arg PREFIX=${prefixpath} --build-arg CK_SCCACHE='${env.CK_SCCACHE}' --build-arg compiler_version='${params.COMPILER_VERSION}' --build-arg compiler_commit='${params.COMPILER_COMMIT}' --build-arg ROCMVERSION='${params.ROCMVERSION}' --build-arg DISABLE_CACHE='git rev-parse ${params.COMPILER_VERSION}' "
    if(no_cache)
    {
        dockerArgs = dockerArgs + " --no-cache "
    }
    echo "Docker Args: ${dockerArgs}"
    def image
    if ( params.BUILD_LEGACY_OS && conf.get("docker_name", "") != "" ){
        image = conf.get("docker_name", "")
        echo "Using legacy docker: ${image}"
    }
    else if ( params.BUILD_GFX950 && conf.get("docker_name", "") != "" ){
        image = conf.get("docker_name", "")
        echo "Using special docker: ${image}"
    }
    else{
        image = getDockerImageName()
        echo "Using default docker: ${image}"
    }
    //Check if image exists 
    def retimage
    try 
    {
        echo "Pulling image: ${image}"
        retimage = docker.image("${image}")
        withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
            retimage.pull()
        }
    }
    catch(Exception ex)
    {
        error "Unable to locate image: ${image}"
    }
    return [retimage, image]
}

def buildDocker(install_prefix){
    show_node_info()
    env.DOCKER_BUILDKIT=1
    checkout scm
    def image_name = getDockerImageName()
    def base_image_name = getBaseDockerImageName()
    echo "Building Docker for ${image_name}"
    def dockerArgs = "--build-arg PREFIX=${install_prefix} --build-arg CK_SCCACHE='${env.CK_SCCACHE}' --build-arg compiler_version='${params.COMPILER_VERSION}' --build-arg compiler_commit='${params.COMPILER_COMMIT}' --build-arg ROCMVERSION='${params.ROCMVERSION}' "
    if(params.COMPILER_VERSION == "amd-staging" || params.COMPILER_VERSION == "amd-mainline" || params.COMPILER_COMMIT != ""){
        dockerArgs = dockerArgs + " --no-cache --build-arg BASE_DOCKER='${base_image_name}' -f Dockerfile.compiler . "
    }
    else{
        dockerArgs = dockerArgs + " -f Dockerfile . "
    }
    echo "Build Args: ${dockerArgs}"
    try{
        if(params.BUILD_DOCKER){
            //force building the new docker if that parameter is true
            echo "Building image: ${image_name}"
            retimage = docker.build("${image_name}", dockerArgs)
            withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
                retimage.push()
            }
            sh 'docker images -q -f dangling=true | xargs --no-run-if-empty docker rmi'
        }
        else{
            echo "Checking for image: ${image_name}"
            sh "docker manifest inspect --insecure ${image_name}"
            echo "Image: ${image_name} found! Skipping building image"
        }
    }
    catch(Exception ex){
        echo "Unable to locate image: ${image_name}. Building image now"
        retimage = docker.build("${image_name}", dockerArgs + ' .')
        withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
            retimage.push()
        }
    }
}

def cmake_build(Map conf=[:]){

    def compiler = build_compiler()
    def config_targets = conf.get("config_targets","check")
    def debug_flags = "-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined " + conf.get("extradebugflags", "")
    def build_envs = "CTEST_PARALLEL_LEVEL=4 " + conf.get("build_env","")
    def prefixpath = conf.get("prefixpath","/opt/rocm")
    def setup_args = conf.get("setup_args","")
    // make sure all unit tests always run on develop branch
    def runAllUnitTests = (env.BRANCH_NAME == "develop") ? true : params.RUN_ALL_UNIT_TESTS
    
    if (prefixpath != "/usr/local"){
        setup_args = setup_args + " -DCMAKE_PREFIX_PATH=${prefixpath} "
    }

    def build_type_debug = (conf.get("build_type",'release') == 'debug')

    //cmake_env can overwrite default CXX variables.
    def cmake_envs = "CXX=${compiler} CXXFLAGS='-Werror' " + conf.get("cmake_ex_env","")

    def package_build = (conf.get("package_build","") == "true")

    if (package_build == true) {
        config_targets = "package"
    }

    if(conf.get("build_install","") == "true")
    {
        config_targets = 'install ' + config_targets
        setup_args = ' -DBUILD_DEV=On -DCMAKE_INSTALL_PREFIX=../install' + setup_args
    } else{
        setup_args = ' -DBUILD_DEV=On' + setup_args
    }
    if (params.DISABLE_DL_KERNELS){
        setup_args = setup_args + " -DDISABLE_DL_KERNELS=ON "
    }

    if(build_type_debug){
        setup_args = " -DCMAKE_BUILD_TYPE=debug -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'" + setup_args
    }else{
        setup_args = " -DCMAKE_BUILD_TYPE=release" + setup_args
    }

    def pre_setup_cmd = """
            #!/bin/bash
            echo \$HSA_ENABLE_SDMA
            ulimit -c unlimited
            rm -rf build
            mkdir build
            rm -rf install
            mkdir install
            cd build
        """
    def invocation_tag=""
    if (setup_args.contains("gfx12")){
        invocation_tag="gfx12"
    }
    if (setup_args.contains("gfx11")){
        invocation_tag="gfx11"
    }
    if (setup_args.contains("gfx10")){
        invocation_tag="gfx10"
    }
    if (setup_args.contains("gfx908")){
        invocation_tag="gfx908"
    }
    if (setup_args.contains("gfx90a")){
        invocation_tag="gfx90a"
    }
    if (setup_args.contains("gfx94")){
        invocation_tag="gfx94"
    }
    if (setup_args.contains("gfx95")){
        invocation_tag="gfx95"
    }
    echo "invocation tag: ${invocation_tag}"
    def redis_pre_setup_cmd = pre_setup_cmd
    if(check_host() && params.USE_SCCACHE && "${env.CK_SCCACHE}" != "null" && "${invocation_tag}" != "") {
        redis_pre_setup_cmd = pre_setup_cmd + """
            #!/bin/bash
            export ROCM_PATH=/opt/rocm
            export SCCACHE_ENABLED=true
            export SCCACHE_LOG_LEVEL=debug
            export SCCACHE_IDLE_TIMEOUT=14400
            export COMPILERS_HASH_DIR=/tmp/.sccache
            export SCCACHE_BIN=/usr/local/.cargo/bin/sccache
            export SCCACHE_EXTRAFILES=/tmp/.sccache/rocm_compilers_hash_file
            export SCCACHE_REDIS="redis://${env.CK_SCCACHE}"
            echo "connect = ${env.CK_SCCACHE}" >> ../script/redis-cli.conf
            export SCCACHE_C_CUSTOM_CACHE_BUSTER="${invocation_tag}"
            echo \$SCCACHE_C_CUSTOM_CACHE_BUSTER
            stunnel ../script/redis-cli.conf
            ../script/sccache_wrapper.sh --enforce_redis
        """
        try {
            def cmd1 = conf.get("cmd1", """
                    ${redis_pre_setup_cmd}
                """)
            sh cmd1
            setup_args = " -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_C_COMPILER_LAUNCHER=sccache " + setup_args
        }
        catch(Exception err){
            echo "could not connect to redis server: ${err.getMessage()}. will not use sccache."
            def cmd2 = conf.get("cmd2", """
                    ${pre_setup_cmd}
                """)
            sh cmd2
        }
    }
    else{
        def cmd3 = conf.get("cmd3",  """
                ${pre_setup_cmd}
            """)
        sh cmd3
    }

    // reduce parallelism when compiling, clang uses too much memory
    def nt = nthreads()
    def cmd
    def setup_cmd
    def build_cmd
    def execute_cmd = conf.get("execute_cmd", "")
    if(!setup_args.contains("NO_CK_BUILD")){
        def cmake_flags = params.NINJA_FTIME_TRACE ? "-O3 -ftime-trace" : "-O3"
        if (params.NINJA_BUILD_TRACE) {
            echo "running ninja build trace"
        }
        setup_cmd = conf.get(
            "setup_cmd",
            """${cmake_envs} cmake -G Ninja ${setup_args} -DCMAKE_CXX_FLAGS=" ${cmake_flags} " .. """
        )
        build_cmd = conf.get(
            "build_cmd",
            "${build_envs} ninja -j${nt} ${config_targets}"
        )
        
        cmd = conf.get("cmd", """
            ${setup_cmd}
            ${build_cmd}
            ${execute_cmd}
        """)
    }
    else{
        cmd = conf.get("cmd", """
            ${execute_cmd}
        """)
    }

    echo cmd

    dir("build"){
        //build CK
        sh cmd
        //run tests except when NO_CK_BUILD or BUILD_LEGACY_OS are set
        if(!setup_args.contains("NO_CK_BUILD") && !params.BUILD_LEGACY_OS){
            if ((setup_args.contains("gfx9") && params.NINJA_BUILD_TRACE) || params.BUILD_INSTANCES_ONLY){
                if (params.NINJA_FTIME_TRACE) {
                    echo "running ninja ftime trace"
                    sh "/ClangBuildAnalyzer/build/ClangBuildAnalyzer  --all . clang_build.log"
                    sh "/ClangBuildAnalyzer/build/ClangBuildAnalyzer  --analyze clang_build.log > clang_build_analysis.log"
                    archiveArtifacts "clang_build_analysis.log"
                }
                sh "python3 ../script/ninja_json_converter.py .ninja_log --legacy-format --output ck_build_trace.json"
                archiveArtifacts "ck_build_trace.json"

                // do not run unit tests when building instances only
                if(!params.BUILD_INSTANCES_ONLY){
                    if (!runAllUnitTests){
                        sh "../script/launch_tests.sh"
                    }
                    else{
                        sh "ninja check"
                    }
                }
                if(params.BUILD_INSTANCES_ONLY){
                    // build deb packages
                    echo "Build packages"
                    sh 'ninja -j64 package'
                    archiveArtifacts artifacts: 'composablekernel-dev*.deb'
                    sh 'mv composablekernel-dev_*.deb composablekernel-dev_all_targets_1.1.0_amd64.deb'
                    stash includes: "composablekernel-dev_all_targets_1.1.0_amd64.deb", name: "packages"
                }
            }
            else{
                // run unit tests unless building library for all targets
                if (!params.BUILD_INSTANCES_ONLY){
                    if (!runAllUnitTests){
                        sh "../script/launch_tests.sh"
                    }
                    else{
                        sh "ninja check"
                    }
                }
            }
        }
    }

    // Only archive from develop
    if (package_build == true && env.BRANCH_NAME == "develop") {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
    }
    //check the node gpu architecture
    def arch = check_arch()
    if (params.RUN_CK_TILE_FMHA_TESTS){
        try{
            archiveArtifacts "perf_fmha_*.log"
            if (arch == 1){
                stash includes: "perf_fmha_**_gfx90a.log", name: "perf_fmha_log_gfx90a"
            }
            else if (arch == 2){
                stash includes: "perf_fmha_**_gfx942.log", name: "perf_fmha_log_gfx942"
            }
        }
        catch(Exception err){
            echo "could not locate the requested artifacts: ${err.getMessage()}. will skip the stashing."
        }
    }
    if (params.RUN_CK_TILE_TRANSPOSE_TESTS){
        try{
            archiveArtifacts "perf_transpose_*.log"
            if (arch_type == 1){
                stash includes: "perf_transpose_**_gfx90a.log", name: "perf_transpose_log_gfx90a"
            }
            else if (arch_type == 2){
                stash includes: "perf_transpose_**_gfx942.log", name: "perf_transpose_log_gfx942"
            }
        }
        catch(Exception err){
            echo "could not locate the requested artifacts: ${err.getMessage()}. will skip the stashing."
        }
    }
    if (params.RUN_CK_TILE_GEMM_TESTS){
        try{
            archiveArtifacts "perf_tile_gemm_**.log"
            if (arch == 1){
                stash includes: "perf_tile_gemm_**_gfx90a.log", name: "perf_tile_gemm_log_gfx90a"
            }
            else if (arch == 2){
                stash includes: "perf_tile_gemm_**_gfx942.log", name: "perf_tile_gemm_log_gfx942"
            }
        }
        catch(Exception err){
            echo "could not locate the requested artifacts: ${err.getMessage()}. will skip the stashing."
        }
    }
}

def buildHipClangJob(Map conf=[:]){
        show_node_info()

        env.HSA_ENABLE_SDMA=0
        checkout scm
        def prefixpath = conf.get("prefixpath", "/opt/rocm")

        // Jenkins is complaining about the render group 
        def dockerOpts
        if ( params.BUILD_INSTANCES_ONLY ){
            dockerOpts = "--group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        }
        else{
            dockerOpts = "--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        }
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1 "
        }
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg CK_SCCACHE='${env.CK_SCCACHE}' --build-arg compiler_version='${params.COMPILER_VERSION}' --build-arg compiler_commit='${params.COMPILER_COMMIT}' --build-arg ROCMVERSION='${params.ROCMVERSION}' "
        if (params.COMPILER_VERSION == "amd-staging" || params.COMPILER_VERSION == "amd-mainline" || params.COMPILER_COMMIT != ""){
            dockerOpts = dockerOpts + " --env HIP_CLANG_PATH='/llvm-project/build/bin' "
        }
        def video_id = sh(returnStdout: true, script: 'getent group video | cut -d: -f3')
        def render_id = sh(returnStdout: true, script: 'getent group render | cut -d: -f3')
        dockerOpts = dockerOpts + " --group-add=${video_id} --group-add=${render_id} "
        echo "Docker flags: ${dockerOpts}"

        def variant = env.STAGE_NAME
        def image
        def retimage
        (retimage, image) = getDockerImage(conf)

        gitStatusWrapper(credentialsId: "${env.ck_git_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCm', repo: 'composable_kernel') {
            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 20, unit: 'HOURS')
                {
                    cmake_build(conf)
                }
            }
        }
        return retimage
}

def reboot(){
    build job: 'reboot-slaves', propagate: false , parameters: [string(name: 'server', value: "${env.NODE_NAME}"),]
}

def buildHipClangJobAndReboot(Map conf=[:]){
    try{
        buildHipClangJob(conf)
    }
    catch(e){
        echo "throwing error exception for the stage"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
    finally{
        if (!conf.get("no_reboot", false)) {
            reboot()
        }
    }
}

def Build_CK(Map conf=[:]){
        show_node_info()

        env.HSA_ENABLE_SDMA=0
        env.DOCKER_BUILDKIT=1
        checkout scm
        def prefixpath = conf.get("prefixpath", "/opt/rocm")

        // Jenkins is complaining about the render group 
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1 "
        }
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg compiler_version='${params.COMPILER_VERSION}' --build-arg compiler_commit='${params.COMPILER_COMMIT}' --build-arg ROCMVERSION='${params.ROCMVERSION}' "
        if (params.COMPILER_VERSION == "amd-staging" || params.COMPILER_VERSION == "amd-mainline" || params.COMPILER_COMMIT != ""){
            dockerOpts = dockerOpts + " --env HIP_CLANG_PATH='/llvm-project/build/bin' "
        }
        if(params.BUILD_LEGACY_OS){
            dockerOpts = dockerOpts + " --env LD_LIBRARY_PATH='/opt/Python-3.8.13/lib' "
        }
        def video_id = sh(returnStdout: true, script: 'getent group video | cut -d: -f3')
        def render_id = sh(returnStdout: true, script: 'getent group render | cut -d: -f3')
        dockerOpts = dockerOpts + " --group-add=${video_id} --group-add=${render_id} "
        echo "Docker flags: ${dockerOpts}"

        def variant = env.STAGE_NAME
        def image
        def retimage

        gitStatusWrapper(credentialsId: "${env.ck_git_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCm', repo: 'composable_kernel') {
            try {
                (retimage, image) = getDockerImage(conf)
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 2, unit: 'MINUTES'){
                        sh 'rocminfo | tee rocminfo.log'
                        if ( !runShell('grep -n "gfx" rocminfo.log') ){
                            throw new Exception ("GPU not found")
                        }
                        else{
                            echo "GPU is OK"
                        }
                    }
                }
            }
            catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                echo "The job was cancelled or aborted"
                throw e
            }
            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 20, unit: 'HOURS')
                {
                    //check whether to run performance tests on this node
                    def arch = check_arch()
                    cmake_build(conf)
                    if ( params.RUN_INDUCTOR_TESTS && !params.BUILD_LEGACY_OS && arch == 1 ){
                            echo "Run inductor codegen tests"
                            sh """
                                  python3 -m venv ${env.WORKSPACE}
                                  . ${env.WORKSPACE}/bin/activate
                                  python3 -m pip install pytest build setuptools setuptools_scm
                                  python3 -m pip install .
                                  python3 -m pytest python/test/test_gen_instances.py
                            """
                    }
                    dir("build"){
                        if (params.RUN_FULL_QA && arch == 2 ){
                            // build deb packages
                            echo "Build packages"
                            sh 'ninja package'
                            archiveArtifacts artifacts: 'composablekernel*.deb'
                            sh 'mv composablekernel-ckprofiler_*.deb composablekernel-ckprofiler_1.1.0_amd64.deb'
                            sh 'mv composablekernel-dev_*.deb composablekernel-dev_1.1.0_amd64.deb'
                            sh 'mv composablekernel-examples_*.deb composablekernel-examples_1.1.0_amd64.deb'
                            sh 'mv composablekernel-tests_*.deb composablekernel-tests_1.1.0_amd64.deb'
                            stash includes: "composablekernel-**.deb", name: "packages"
                        }
                    }
                    // run performance tests, stash the logs, results will be processed on the master node
					dir("script"){
                        if (params.RUN_PERFORMANCE_TESTS){
                        if (params.RUN_FULL_QA && arch == 1){
                            // run full tests on gfx90a
                            echo "Run full performance tests"
                            sh "./run_full_performance_tests.sh 0 QA_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME}"
                            archiveArtifacts "perf_gemm.log"
                            archiveArtifacts "perf_resnet50_N256.log"
                            archiveArtifacts "perf_resnet50_N4.log"
                            archiveArtifacts "perf_batched_gemm.log"
                            archiveArtifacts "perf_grouped_gemm.log"
                            archiveArtifacts "perf_grouped_conv_fwd.log"
                            archiveArtifacts "perf_grouped_conv_bwd_data.log"
                            archiveArtifacts "perf_grouped_conv_bwd_weight.log"
                            archiveArtifacts "perf_gemm_bilinear.log"
                            archiveArtifacts "perf_reduction.log"
                            archiveArtifacts "perf_splitK_gemm.log"
                            archiveArtifacts "perf_onnx_gemm.log"
                            archiveArtifacts "perf_mixed_gemm.log"
                            stash includes: "perf_**.log", name: "perf_log"
                        }
                        else if ( arch == 1 ){
                            // run standard tests on gfx90a
                            echo "Run performance tests"
                            sh "./run_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME}"
                            archiveArtifacts "perf_gemm.log"
                            archiveArtifacts "perf_onnx_gemm.log"
                            archiveArtifacts "perf_resnet50_N256.log"
                            archiveArtifacts "perf_resnet50_N4.log"
                            stash includes: "perf_**.log", name: "perf_log"
                        }
                        // disable performance tests on gfx1030 for now.
                        //else if ( arch == 3){
                            // run basic tests on gfx1030
                        //    echo "Run gemm performance tests"
                        //    sh "./run_gemm_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} gfx10"
                        //    archiveArtifacts "perf_onnx_gemm_gfx10.log"
                        //    stash includes: "perf_onnx_gemm_gfx10.log", name: "perf_log_gfx10"
                        //}
                        else if ( arch == 4){
                            // run basic tests on gfx11
                            echo "Run gemm performance tests"
                            sh "./run_gemm_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} gfx11"
                            archiveArtifacts "perf_onnx_gemm_gfx11.log"
                            stash includes: "perf_onnx_gemm_gfx11.log", name: "perf_log_gfx11"
                        }
                        else if ( arch == 5 ){
                            // run basic tests on gfx12
                            echo "Run gemm performance tests"
                            sh "./run_gemm_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} gfx12"
                            archiveArtifacts "perf_onnx_gemm_gfx12.log"
                            stash includes: "perf_onnx_gemm_gfx12.log", name: "perf_log_gfx12"
                        }
                        else if ( arch == 6 ){
                            // run basic tests on gfx908
                            echo "Run performance tests"
                            sh "./run_gemm_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} gfx908"
                            archiveArtifacts "perf_onnx_gemm_gfx908.log"
                            stash includes: "perf_onnx_gemm_gfx908.log", name: "perf_log_gfx908"
                        }
                        else if ( arch == 7 ){
                            // run basic tests on gfx950
                            echo "Run performance tests"
                            sh "./run_gemm_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} gfx950"
                            archiveArtifacts "perf_onnx_gemm_gfx950.log"
                            stash includes: "perf_onnx_gemm_gfx950.log", name: "perf_log_gfx950"
                        }
                        }
                    }
                    if (params.hipTensor_test && arch == 1 ){
                        // build and test hipTensor on gfx90a node
                        sh """#!/bin/bash
                            rm -rf "${params.hipTensor_branch}".zip
                            rm -rf hipTensor-"${params.hipTensor_branch}"
                            wget https://github.com/ROCm/hipTensor/archive/refs/heads/"${params.hipTensor_branch}".zip
                            unzip -o "${params.hipTensor_branch}".zip
                        """
                        dir("hipTensor-${params.hipTensor_branch}"){
                            sh """#!/bin/bash
                                mkdir -p build
                                ls -ltr
                                CC=hipcc CXX=hipcc cmake -Bbuild . -D CMAKE_PREFIX_PATH="${env.WORKSPACE}/install"
                                cmake --build build -- -j
                                ctest --test-dir build
                            """
                        }
                    }
                }
            }
        }
        return retimage
}

def Build_CK_and_Reboot(Map conf=[:]){
    try{
        Build_CK(conf)
    }
    catch(e){
        echo "throwing error exception while building CK"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
    finally{
        if (!conf.get("no_reboot", false)) {
            reboot()
        }
    }
}

def process_results(Map conf=[:]){
    env.HSA_ENABLE_SDMA=0
    checkout scm
    //use older image that has user jenkins
    def image = "rocm/composable_kernel:ck_ub22.04_rocm6.3"
    def prefixpath = "/opt/rocm"

    // Jenkins is complaining about the render group 
    def dockerOpts="--cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    if (conf.get("enforce_xnack_on", false)) {
        dockerOpts = dockerOpts + " --env HSA_XNACK=1 "
    }

    def variant = env.STAGE_NAME
    def retimage

    gitStatusWrapper(credentialsId: "${env.ck_git_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCm', repo: 'composable_kernel') {
        try
        {
            echo "Pulling image: ${image}"
            retimage = docker.image("${image}")
            withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
                retimage.pull()
            }
        }
        catch(Exception ex)
        {
            error "Unable to locate image: ${image}"
        }
    }

    withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
        timeout(time: 15, unit: 'MINUTES'){
            try{
                dir("script"){
                    if (params.RUN_CK_TILE_FMHA_TESTS){
                        try{
                            unstash "perf_fmha_log_gfx942"
                            unstash "perf_fmha_log_gfx90a"
                        }
                        catch(Exception err){
                            echo "could not locate the FMHA performance logs: ${err.getMessage()}."
                        }
                    }
                    if (params.RUN_CK_TILE_TRANSPOSE_TESTS){
                        try{
                            unstash "perf_transpose_log_gfx942"
                            unstash "perf_transpose_log_gfx90a"
                        }
                        catch(Exception err){
                            echo "could not locate the Transpose performance logs: ${err.getMessage()}."
                        }
                    }
                    if (params.RUN_CK_TILE_GEMM_TESTS){
                        try{
                            unstash "perf_tile_gemm_log_gfx942"
                            unstash "perf_tile_gemm_log_gfx90a"
                        }
                        catch(Exception err){
                            echo "could not locate the GEMM performance logs: ${err.getMessage()}."
                        }
                    }
                    if (params.RUN_FULL_QA || params.BUILD_INSTANCES_ONLY){
                        // unstash deb packages
                        unstash "packages"
                        sh "sshpass -p ${env.ck_deb_pw} scp -o StrictHostKeyChecking=no composablekernel-*.deb ${env.ck_deb_user}@${env.ck_deb_ip}:/var/www/html/composable_kernel/"
                    }
                    else{
                        // unstash perf files to master
                        unstash "perf_log"
                        try{
                            unstash "perf_log_gfx11"
                            unstash "perf_log_gfx12"
                        }
                        catch(Exception err){
                            echo "could not locate the GEMM gfx11/gfx12 performance logs: ${err.getMessage()}."
                        }
                        sh "./process_perf_data.sh"
                    }
                }
            }
            catch(e){
                echo "Throwing error exception while processing performance test results"
                echo 'Exception occurred: ' + e.toString()
                throw e
            }
            finally{
                echo "Finished processing performance test results"
            }
        }
    }
}

//launch develop branch daily jobs
CRON_SETTINGS = BRANCH_NAME == "develop" ? '''0 23 * * * % RUN_FULL_QA=true;DISABLE_DL_KERNELS=true;RUN_CK_TILE_FMHA_TESTS=true;RUN_CK_TILE_TRANSPOSE_TESTS=true;RUN_CK_TILE_GEMM_TESTS=true;RUN_TILE_ENGINE_GEMM_TESTS=true;RUN_PERFORMANCE_TESTS=true;RUN_ALL_UNIT_TESTS=true
                                              0 21 * * * % RUN_GROUPED_CONV_LARGE_CASES_TESTS=true;hipTensor_test=true;BUILD_GFX908=true;BUILD_GFX942=true;BUILD_GFX950=true;RUN_PERFORMANCE_TESTS=true;RUN_ALL_UNIT_TESTS=true
                                              0 19 * * * % BUILD_DOCKER=true;COMPILER_VERSION=amd-staging;BUILD_COMPILER=/llvm-project/build/bin/clang++;USE_SCCACHE=false;NINJA_BUILD_TRACE=true;RUN_ALL_UNIT_TESTS=true
                                              0 17 * * * % BUILD_DOCKER=true;COMPILER_VERSION=amd-mainline;BUILD_COMPILER=/llvm-project/build/bin/clang++;USE_SCCACHE=false;NINJA_BUILD_TRACE=true;RUN_ALL_UNIT_TESTS=true
                                              0 15 * * * % BUILD_INSTANCES_ONLY=true;USE_SCCACHE=false;NINJA_BUILD_TRACE=true
                                              0 13 * * * % BUILD_LEGACY_OS=true;USE_SCCACHE=false;RUN_PERFORMANCE_TESTS=false''' : ""

pipeline {
    agent none
    triggers {
        parameterizedCron(CRON_SETTINGS)
    }
    options {
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
            defaultValue: '6.4.1',
            description: 'Specify which ROCM version to use: 6.4.1 (default).')
        string(
            name: 'COMPILER_VERSION', 
            defaultValue: '', 
            description: 'Specify which version of compiler to use: release, amd-staging, amd-mainline, or leave blank (default).')
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
            defaultValue: 'mainline',
            description: 'Specify which branch of hipTensor to use (default: mainline)')
        booleanParam(
            name: "USE_SCCACHE",
            defaultValue: true,
            description: "Use the sccache for building CK (default: ON)")
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
            name: "RUN_CODEGEN_TESTS",
            defaultValue: true,
            description: "Run codegen tests (default: ON)")
        booleanParam(
            name: "RUN_CK_TILE_FMHA_TESTS",
            defaultValue: false,
            description: "Run the ck_tile FMHA tests (default: OFF)")
        booleanParam(
            name: "RUN_CK_TILE_TRANSPOSE_TESTS",
            defaultValue: false,
            description: "Run the ck_tile Transpose tests (default: OFF)")
        booleanParam(
            name: "RUN_CK_TILE_GEMM_TESTS",
            defaultValue: false,
            description: "Run the ck_tile GEMM tests (default: OFF)")
        booleanParam(
            name: "RUN_TILE_ENGINE_GEMM_TESTS",
            defaultValue: false,
            description: "Run the tile_engine_gemm tests (default: OFF)")
        booleanParam(
            name: "BUILD_INSTANCES_ONLY",
            defaultValue: false,
            description: "Test building instances for various architectures simultaneously (default: OFF)")
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
            defaultValue: false,
            description: "Build CK and run tests on gfx942 (default: OFF)")
        booleanParam(
            name: "BUILD_GFX950",
            defaultValue: false,
            description: "Build CK and run tests on gfx950 (default: OFF)")
        booleanParam(
            name: "BUILD_GFX10",
            defaultValue: true,
            description: "Build CK and run tests on gfx10 (default: ON)")
        booleanParam(
            name: "BUILD_GFX11",
            defaultValue: true,
            description: "Build CK and run tests on gfx11 (default: ON)")
        booleanParam(
            name: "BUILD_GFX12",
            defaultValue: true,
            description: "Build CK and run tests on gfx12 (default: ON)")
        booleanParam(
            name: "NINJA_BUILD_TRACE",
            defaultValue: false,
            description: "Generate a ninja build trace (default: OFF)")
        booleanParam(
            name: "NINJA_FTIME_TRACE",
            defaultValue: false,
            description: "Generate a detailed time trace (default: OFF)")
        booleanParam(
            name: "BUILD_LEGACY_OS",
            defaultValue: false,
            description: "Try building CK with legacy OS dockers: RHEL8 and SLES15 (default: OFF)")
        booleanParam(
            name: "RUN_INDUCTOR_TESTS",
            defaultValue: true,
            description: "Run inductor codegen tests (default: ON)")
        booleanParam(
            name: "RUN_ALL_UNIT_TESTS",
            defaultValue: false,
            description: "Run all unit tests (default: OFF)")
    }
    environment{
        dbuser = "${dbuser}"
        dbpassword = "${dbpassword}"
        dbsship = "${dbsship}"
        dbsshport = "${dbsshport}"
        dbsshuser = "${dbsshuser}"
        dbsshpassword = "${dbsshpassword}"
        ck_git_creds = "${ck_git_creds}"
        gerrit_cred="${gerrit_cred}"
        DOCKER_BUILDKIT = "1"
    }
    stages{
        stage("Build Docker"){
            parallel{
                stage('Docker /opt/rocm'){
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildDocker('/opt/rocm')
                        cleanWs()
                    }
                }
            }
        }
        stage("Static checks") {
            parallel{
                stage('Clang Format and Cppcheck') {
                    when {
                        beforeAgent true
                        expression { params.RUN_CPPCHECK.toBoolean() }
                    }
                    agent{ label rocmnode("nogpu") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_cmd = "find .. -not -path \'*.git*\' -iname \'*.h\' \
                                -o -not -path \'*.git*\' -iname \'*.hpp\' \
                                -o -not -path \'*.git*\' -iname \'*.cpp\' \
                                -o -iname \'*.h.in\' \
                                -o -iname \'*.hpp.in\' \
                                -o -iname \'*.cpp.in\' \
                                -o -iname \'*.cl\' \
                                | grep -v 'build/' \
                                | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-18 -style=file {} | diff - {}\' && \
                                /cppcheck/build/bin/cppcheck ../* -v -j \$(nproc) -I ../include -I ../profiler/include -I ../library/include \
                                -D CK_ENABLE_FP64 -D CK_ENABLE_FP32 -D CK_ENABLE_FP16 -D CK_ENABLE_FP8 -D CK_ENABLE_BF16 -D CK_ENABLE_BF8 -D CK_ENABLE_INT8 \
                                -D __gfx908__ -D __gfx90a__ -D __gfx942__ -D __gfx1030__ -D __gfx1100__ -D __gfx1101__ -D __gfx1102__ \
                                -U __gfx803__ -U __gfx900__ -U __gfx906__ -U CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4 \
                                --file-filter=*.cpp --force --enable=all --output-file=ck_cppcheck.log"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, setup_cmd: "", build_cmd: "", execute_cmd: execute_cmd, no_reboot:true)
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
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_cmd = "find .. -not -path \'*.git*\' -iname \'*.h\' \
                                -o -not -path \'*.git*\' -iname \'*.hpp\' \
                                -o -not -path \'*.git*\' -iname \'*.cpp\' \
                                -o -iname \'*.h.in\' \
                                -o -iname \'*.hpp.in\' \
                                -o -iname \'*.cpp.in\' \
                                -o -iname \'*.cl\' \
                                | grep -v 'build/' \
                                | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-18 -style=file {} | diff - {}\'"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, setup_cmd: "", build_cmd: "", execute_cmd: execute_cmd, no_reboot:true)
                        cleanWs()
                    }
                }
            }
        }
        stage("Run Grouped Conv Large Case Tests")
        {
            parallel
            {
                stage("Run Grouped Conv Large Case Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_GROUPED_CONV_LARGE_CASES_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a")}
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ ../script/cmake-ck-dev.sh  ../ gfx90a && \
                                           make -j64 test_grouped_convnd_fwd_large_cases_xdl test_grouped_convnd_bwd_data_xdl_large_cases test_grouped_convnd_fwd_bias_clamp_large_cases && \
                                           ./bin/test_grouped_convnd_fwd_large_cases_xdl && ./bin/test_grouped_convnd_bwd_data_xdl_large_cases && ./bin/test_grouped_convnd_fwd_bias_clamp_large_cases"""
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, no_reboot:true, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
            }
        }
        stage("Run Codegen Tests")
        {
            parallel
            {
                stage("Run Codegen Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CODEGEN_TESTS.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a")}
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ CXX=/opt/rocm/llvm/bin/clang++ cmake ../codegen && \
                                           make -j64 check"""
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, no_reboot:true, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
            }
        }
        stage("Run CK_TILE_FMHA Tests")
        {
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
                        execute_args = """ ../script/cmake-ck-dev.sh  ../ gfx90a && \
                                           make -j64 tile_example_fmha_fwd tile_example_fmha_bwd && \
                                           cd ../ &&
                                           example/ck_tile/01_fmha/script/run_full_test.sh "CI_${params.COMPILER_VERSION}" "${env.BRANCH_NAME}" "${NODE_NAME}" gfx90a """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, no_reboot:true, build_type: 'Release', execute_cmd: execute_args)
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
                        execute_args = """ ../script/cmake-ck-dev.sh  ../ gfx942 && \
                                           make -j64 tile_example_fmha_fwd tile_example_fmha_bwd && \
                                           cd ../ &&
                                           example/ck_tile/01_fmha/script/run_full_test.sh "CI_${params.COMPILER_VERSION}" "${env.BRANCH_NAME}" "${NODE_NAME}" gfx942 """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, no_reboot:true, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
            }
        }
        stage("Run CK_TILE_TRANSPOSE Tests")
        {
            parallel
            {
                stage("Run CK_TILE_TRANSPOSE Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CK_TILE_TRANSPOSE_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ ../script/cmake-ck-dev.sh  ../ gfx90a && \
                                           make -j64 tile_example_batched_transpose && \
                                           cd ../ &&
                                           example/ck_tile/35_batched_transpose/script/run_full_test.sh "CI_${params.COMPILER_VERSION}" "${env.BRANCH_NAME}" "${NODE_NAME}" gfx90a """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, no_reboot:true, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
                stage("Run CK_TILE_TRANSPOSE Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CK_TILE_TRANSPOSE_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ ../script/cmake-ck-dev.sh  ../ gfx942 && \
                                           make -j64 tile_example_batched_transpose && \
                                           cd ../ &&
                                           example/ck_tile/35_batched_transpose/script/run_full_test.sh "CI_${params.COMPILER_VERSION}" "${env.BRANCH_NAME}" "${NODE_NAME}" gfx942 """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, no_reboot:true, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
            }
        }
        stage("Run CK_TILE_GEMM Tests")
        {
            parallel
            {
                stage("Run CK_TILE_GEMM Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CK_TILE_GEMM_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ ../script/cmake-ck-dev.sh  ../ gfx90a && \
                                           make -j64 tile_example_gemm_universal && \
                                           cd ../ &&
                                           example/ck_tile/03_gemm/script/run_full_test.sh "CI_${params.COMPILER_VERSION}" "${env.BRANCH_NAME}" "${NODE_NAME}" gfx90a """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, no_reboot:true, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
                stage("Run CK_TILE_GEMM Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_CK_TILE_GEMM_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ ../script/cmake-ck-dev.sh  ../ gfx942 && \
                                           make -j64 tile_example_gemm_universal && \
                                           cd ../ &&
                                           example/ck_tile/03_gemm/script/run_full_test.sh "CI_${params.COMPILER_VERSION}" "${env.BRANCH_NAME}" "${NODE_NAME}" gfx942 """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, no_reboot:true, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
            }
        }
        stage("Run TILE_ENGINE_GEMM Tests")
        {
            parallel
            {
                stage("Run TILE_ENGINE_GEMM Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_TILE_ENGINE_GEMM_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                                            -D CMAKE_CXX_COMPILER="${build_compiler()}" \
                                            -D CMAKE_BUILD_TYPE=Release \
                                            -D GPU_TARGETS="gfx90a" \
                                            -D GEMM_DATATYPE="fp8;fp16" \
                                            -D GEMM_LAYOUT="rcr;rrr;crr;ccr" \
                                            -DCMAKE_CXX_FLAGS=" -O3 " .. && \
                                           ninja -j64 benchmark_gemm_fp8_rcr && \
                                           ./bin/benchmark_gemm_fp8_rcr && \
                                           ninja -j64 benchmark_gemm_fp16_rcr && \
                                           ./bin/benchmark_gemm_fp16_rcr && \
                                           ninja -j64 benchmark_gemm_fp8_crr && \
                                           ./bin/benchmark_gemm_fp8_crr && \
                                           ninja -j64 benchmark_gemm_fp16_crr && \
                                           ./bin/benchmark_gemm_fp16_crr && \
                                           ninja -j64 benchmark_gemm_fp8_ccr && \
                                           ./bin/benchmark_gemm_fp8_ccr && \
                                           ninja -j64 benchmark_gemm_fp16_ccr && \
                                           ./bin/benchmark_gemm_fp16_ccr && \
                                           ninja -j64 benchmark_gemm_fp8_rrr && \
                                           ./bin/benchmark_gemm_fp8_rrr && \
                                           ninja -j64 benchmark_gemm_fp16_rrr && \
                                           ./bin/benchmark_gemm_fp16_rrr """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, no_reboot:true, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
                stage("Run TILE_ENGINE_GEMM Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_TILE_ENGINE_GEMM_TESTS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    environment{
                        setup_args = "NO_CK_BUILD"
                        execute_args = """ cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                                            -D CMAKE_CXX_COMPILER="${build_compiler()}" \
                                            -D CMAKE_BUILD_TYPE=Release \
                                            -D GPU_TARGETS="gfx942" \
                                            -D GEMM_DATATYPE="fp8;fp16" \
                                            -D GEMM_LAYOUT="rcr;rrr;crr;ccr" \
                                            -DCMAKE_CXX_FLAGS=" -O3 " .. && \
                                           ninja -j64 benchmark_gemm_fp8_rcr && \
                                           ./bin/benchmark_gemm_fp8_rcr && \
                                           ninja -j64 benchmark_gemm_fp16_rcr && \
                                           ./bin/benchmark_gemm_fp16_rcr && \
                                           ninja -j64 benchmark_gemm_fp8_crr && \
                                           ./bin/benchmark_gemm_fp8_crr && \
                                           ninja -j64 benchmark_gemm_fp16_crr && \
                                           ./bin/benchmark_gemm_fp16_crr && \
                                           ninja -j64 benchmark_gemm_fp8_ccr && \
                                           ./bin/benchmark_gemm_fp8_ccr && \
                                           ninja -j64 benchmark_gemm_fp16_ccr && \
                                           ./bin/benchmark_gemm_fp16_ccr && \
                                           ninja -j64 benchmark_gemm_fp8_rrr && \
                                           ./bin/benchmark_gemm_fp8_rrr && \
                                           ninja -j64 benchmark_gemm_fp16_rrr && \
                                           ./bin/benchmark_gemm_fp16_rrr """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, no_reboot:true, build_type: 'Release', execute_cmd: execute_args)
                        cleanWs()
                    }
                }
            }
        }

		stage("Build CK and run Tests")
        {
            parallel
            {
                stage("Build CK with RHEL8")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        def docker_name = "${env.CK_DOCKERHUB_PRIVATE}:ck_rhel8_rocm6.3"
                        setup_args = """ -DGPU_TARGETS="gfx942" \
                                         -DCMAKE_CXX_FLAGS=" -O3 " \
                                         -DCK_USE_ALTERNATIVE_PYTHON=/opt/Python-3.8.13/bin/python3.8 """
                        execute_args = " "
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: " ", no_reboot:true, build_type: 'Release', docker_name: docker_name)
                        cleanWs()
                    }
                }
                stage("Build CK with SLES15")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        def docker_name = "${env.CK_DOCKERHUB_PRIVATE}:ck_sles15_rocm6.3"
                        setup_args = """ -DGPU_TARGETS="gfx942" \
                                         -DCMAKE_CXX_FLAGS=" -O3 " \
                                         -DCK_USE_ALTERNATIVE_PYTHON=/opt/Python-3.8.13/bin/python3.8 """
                        execute_args = " "
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: " ", no_reboot:true, build_type: 'Release', docker_name: docker_name)
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx942")
                {
                    when {
                        beforeAgent true
                        expression { (params.BUILD_GFX942.toBoolean() || params.RUN_FULL_QA.toBoolean()) && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install \
                                         -DGPU_TARGETS="gfx942" \
                                         -DCMAKE_CXX_FLAGS=" -O3 " """
                        execute_args = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                                           cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                                           -DGPU_TARGETS="gfx942" \
                                           -DCMAKE_CXX_COMPILER="${build_compiler()}" \
                                           -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx950")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX950.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx950") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install \
                                         -DGPU_TARGETS="gfx950" \
                                         -DCMAKE_CXX_FLAGS=" -O3 " """
                        execute_args = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                                           cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                                           -DGPU_TARGETS="gfx950" \
                                           -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
                                           -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, docker_name: "${env.CK_DOCKERHUB_PRIVATE}:ck_ub24.04_rocm7.0", config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx908")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX908.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx908") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx908" -DCMAKE_CXX_FLAGS=" -O3 " """
                        execute_args = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                                           cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                                           -DGPU_TARGETS="gfx908" \
                                           -DCMAKE_CXX_COMPILER="${build_compiler()}" \
                                           -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX90A.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx90a" -DCMAKE_CXX_FLAGS=" -O3 " """
                        execute_args = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                                           cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                                           -DGPU_TARGETS="gfx90a" \
                                           -DCMAKE_CXX_COMPILER="${build_compiler()}" \
                                           -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK instances for all supported targets")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_INSTANCES_ONLY.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx942") }
                    steps{
                        script {
                            def execute_args = params.NINJA_FTIME_TRACE ? 
                                """ cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                                    -D CMAKE_CXX_COMPILER="${build_compiler()}" \
                                    -D CMAKE_BUILD_TYPE=Release \
                                    -D CMAKE_CXX_FLAGS=" -O3 -ftime-trace" .. && ninja -j64 """ :
                                """ cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                                    -D CMAKE_CXX_COMPILER="${build_compiler()}" \
                                    -D CMAKE_BUILD_TYPE=Release \
                                    -D CMAKE_CXX_FLAGS=" -O3 " .. && ninja -j64 """
                            
                            buildHipClangJobAndReboot(setup_cmd: "",  build_cmd: "", no_reboot:true, build_type: 'Release', execute_cmd: execute_args)
                        }
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx1030")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX10.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1030") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx10-3-generic" -DCMAKE_CXX_FLAGS=" -O3 " """
                        execute_args = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                                           cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                                           -DGPU_TARGETS="gfx10-3-generic" \
                                           -DCMAKE_CXX_COMPILER="${build_compiler()}" \
                                           -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx1101")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX11.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1101") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx11-generic" -DCMAKE_CXX_FLAGS=" -O3 " """
                        execute_args = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                                           cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                                           -DGPU_TARGETS="gfx11-generic" \
                                           -DCMAKE_CXX_COMPILER="${build_compiler()}" \
                                           -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
                    }
                }
                stage("Build CK and run Tests on gfx1201")
                {
                    when {
                        beforeAgent true
                        expression { params.BUILD_GFX12.toBoolean() && !params.RUN_FULL_QA.toBoolean() && !params.BUILD_INSTANCES_ONLY.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent{ label rocmnode("gfx1201") }
                    environment{
                        setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="gfx12-generic" -DCMAKE_CXX_FLAGS=" -O3 " """
                        execute_args = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                                           cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" \
                                           -DGPU_TARGETS="gfx12-generic" \
                                           -DCMAKE_CXX_COMPILER="${build_compiler()}" \
                                           -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
                                           -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                        cleanWs()
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
                        expression { params.RUN_PERFORMANCE_TESTS.toBoolean() && !params.BUILD_LEGACY_OS.toBoolean() }
                    }
                    agent { label 'mici' }
                    steps{
                        process_results()
                        cleanWs()
                    }
                }
            }
        }
    }
}

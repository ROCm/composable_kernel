def rocmnode(name) {
    return 'rocmtest && miopen && ' + name
}

def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME"
        lsb_release -sd
        uname -r
        ls /opt/ -la
    """
}

def runShell(String command){
    def responseCode = sh returnStatus: true, script: "${command} > tmp.txt"
    def output = readFile(file: "tmp.txt")
    echo "tmp.txt contents: $output"
    return (output != "")
}

def getDockerImageName(){
    def img = "${env.CK_IMAGE_URL}:new_ck_ub20.04_rocm5.2.3_${params.COMPILER_VERSION}"
    return img
}

def build_compiler(){
    def compiler
    if (params.BUILD_COMPILER == "hipcc"){
        compiler = '/opt/rocm/bin/hipcc'
    }
    else{
        if (params.COMPILER_VERSION == "release"){
            compiler = "/opt/rocm/llvm/bin/clang++"
        }
        else{
            compiler = "/llvm-project/build/bin/clang++"
        }        
    }
    return compiler
}

def getDockerImage(Map conf=[:]){
    env.DOCKER_BUILDKIT=1
    def prefixpath = conf.get("prefixpath", "/opt/rocm") // prefix:/opt/rocm
    def gpu_arch = conf.get("gpu_arch", "gfx908") // prebuilt dockers should have all the architectures enabled so one image can be used for all stages
    def no_cache = conf.get("no_cache", false)
    def dockerArgs = "--build-arg BUILDKIT_INLINE_CACHE=1 --build-arg PREFIX=${prefixpath} --build-arg compiler_version='${params.COMPILER_VERSION}' "
    if(env.CCACHE_HOST)
    {
        def check_host = sh(script:"""(printf "PING\r\n";) | nc -N ${env.CCACHE_HOST} 6379 """, returnStdout: true).trim()
        if(check_host == "+PONG")
        {
            echo "FOUND CCACHE SERVER: ${CCACHE_HOST}"
        }
        else 
        {
            echo "CCACHE SERVER: ${CCACHE_HOST} NOT FOUND, got ${check_host} response"
        }
        dockerArgs = dockerArgs + " --build-arg CCACHE_SECONDARY_STORAGE='redis://${env.CCACHE_HOST}' --build-arg COMPILER_LAUNCHER='ccache' "
        env.CCACHE_DIR = """/tmp/ccache_store"""
        env.CCACHE_SECONDARY_STORAGE="""redis://${env.CCACHE_HOST}"""
    }
    if(no_cache)
    {
        dockerArgs = dockerArgs + " --no-cache "
    }
    echo "Docker Args: ${dockerArgs}"
    def image = getDockerImageName()
    //Check if image exists 
    def retimage
    try 
    {
        echo "Pulling down image: ${image}"
        retimage = docker.image("${image}")
        retimage.pull()
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
    echo "Building Docker for ${image_name}"
    def dockerArgs = "--build-arg BUILDKIT_INLINE_CACHE=1 --build-arg PREFIX=${install_prefix} --build-arg compiler_version='${params.COMPILER_VERSION}' "
    if(env.CCACHE_HOST)
    {
        def check_host = sh(script:"""(printf "PING\\r\\n";) | nc  -N ${env.CCACHE_HOST} 6379 """, returnStdout: true).trim()
        if(check_host == "+PONG")
        {
            echo "FOUND CCACHE SERVER: ${CCACHE_HOST}"
        }
        else 
        {
            echo "CCACHE SERVER: ${CCACHE_HOST} NOT FOUND, got ${check_host} response"
        }
        dockerArgs = dockerArgs + " --build-arg CCACHE_SECONDARY_STORAGE='redis://${env.CCACHE_HOST}' --build-arg COMPILER_LAUNCHER='ccache' "
        env.CCACHE_DIR = """/tmp/ccache_store"""
        env.CCACHE_SECONDARY_STORAGE="""redis://${env.CCACHE_HOST}"""
    }

    echo "Build Args: ${dockerArgs}"
    try{
        echo "Checking for image: ${image_name}"
        sh "docker manifest inspect --insecure ${image_name}"
        echo "Image: ${image_name} found!! Skipping building image"
    }
    catch(Exception ex){
        echo "Unable to locate image: ${image_name}. Building image now"
        retimage = docker.build("${image_name}", dockerArgs + ' .')
        retimage.push()
    }
}

def cmake_build(Map conf=[:]){

    def compiler = build_compiler()
    def config_targets = conf.get("config_targets","check")
    def debug_flags = "-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined " + conf.get("extradebugflags", "")
    def build_envs = "CTEST_PARALLEL_LEVEL=4 " + conf.get("build_env","")
    def prefixpath = conf.get("prefixpath","/opt/rocm")
    def setup_args = conf.get("setup_args","")

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
        setup_args = ' -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=../install' + setup_args
    } else{
        setup_args = ' -DBUILD_DEV=On' + setup_args
    }

    if(build_type_debug){
        setup_args = " -DCMAKE_BUILD_TYPE=debug -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'" + setup_args
    }else{
        setup_args = " -DCMAKE_BUILD_TYPE=release" + setup_args
    }
    if(env.CCACHE_HOST)
    {
        setup_args = " -DCMAKE_CXX_COMPILER_LAUNCHER='ccache' -DCMAKE_C_COMPILER_LAUNCHER='ccache' " + setup_args
    }

    def pre_setup_cmd = """
            echo \$HSA_ENABLE_SDMA
            ulimit -c unlimited
            rm -rf build
            mkdir build
            rm -rf install
            mkdir install
            cd build
        """
    def setup_cmd = conf.get("setup_cmd", "${cmake_envs} cmake ${setup_args}   .. ")
    // reduce parallelism when compiling, clang uses too much memory
    def build_cmd = conf.get("build_cmd", "${build_envs} dumb-init make  -j\$(( \$(nproc) / 2 )) ${config_targets}")
    def execute_cmd = conf.get("execute_cmd", "")

    def cmd = conf.get("cmd", """
            ${pre_setup_cmd}
            ${setup_cmd}
            ${build_cmd}
            ${execute_cmd}
        """)

    echo cmd
    sh cmd

    // Only archive from master or develop
    if (package_build == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master")) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
    }
}

def buildHipClangJob(Map conf=[:]){
        show_node_info()

        env.HSA_ENABLE_SDMA=0
        checkout scm

        def image = getDockerImageName() 
        def prefixpath = conf.get("prefixpath", "/opt/rocm")
        def gpu_arch = conf.get("gpu_arch", "gfx908")

        // Jenkins is complaining about the render group 
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1 --env GPU_ARCH='${gpu_arch}' "
        }
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg compiler_version='${params.COMPILER_VERSION}' "
        if (params.COMPILER_VERSION != "release"){
            dockerOpts = dockerOpts + " --env HIP_CLANG_PATH='/llvm-project/build/bin' "
        }

        def variant = env.STAGE_NAME

        def retimage

        gitStatusWrapper(credentialsId: "${status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'composable_kernel') {
            try {
                //retimage = docker.build("${image}", dockerArgs + '.')
                (retimage, image) = getDockerImage(conf)
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES'){
                        sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo | tee clinfo.log'
                        if ( runShell('grep -n "Number of devices:.*. 0" clinfo.log') ){
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
            catch(Exception ex) {
                retimage = docker.build("${image}", dockerArgs + " --no-cache .")
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES'){
                        sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo |tee clinfo.log'
                        if ( runShell('grep -n "Number of devices:.*. 0" clinfo.log') ){
                            throw new Exception ("GPU not found")
                        }
                        else{
                            echo "GPU is OK"
                        }
                    }
                }
            }

            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 5, unit: 'HOURS')
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







def runCKProfiler(Map conf=[:]){
        show_node_info()

        env.HSA_ENABLE_SDMA=0
        checkout scm


        def image = getDockerImageName()
        def prefixpath = conf.get("prefixpath", "/opt/rocm")
        def gpu_arch = conf.get("gpu_arch", "gfx908")

        // Jenkins is complaining about the render group 
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1 --env GPU_ARCH='${gpu_arch}' "
        }
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg compiler_version='${params.COMPILER_VERSION}' "
        if (params.COMPILER_VERSION != "release"){
            dockerOpts = dockerOpts + " --env HIP_CLANG_PATH='/llvm-project/build/bin' "
        }

        def variant = env.STAGE_NAME
        def retimage

        gitStatusWrapper(credentialsId: "${status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'composable_kernel') {
            try {
                (retimage, image) = getDockerImage(conf)
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES'){
                        sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo | tee clinfo.log'
                        if ( runShell('grep -n "Number of devices:.*. 0" clinfo.log') ){
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
            catch(Exception ex) {
                retimage = docker.build("${image}", dockerArgs + " --no-cache .")
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES'){
                        sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo | tee clinfo.log'
                        if ( runShell('grep -n "Number of devices:.*. 0" clinfo.log') ){
                            throw new Exception ("GPU not found")
                        }
                        else{
                            echo "GPU is OK"
                        }
                    }
                }
            }

            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 24, unit: 'HOURS')
                {
                    //cmake_build(conf)
                    //instead of building, just get the CK deb package and install it
                    //sh 'pwd'
                    //sh 'ls'
                    //sh "mkdir Libs_composable_kernel_${env.BRANCH_NAME}"
                    //dir("Libs_composable_kernel_${env.BRANCH_NAME}"){
                    //get deb package
                    wget http://micimaster.amd.com/blue/organizations/jenkins/MLLibs%2Fcomposable_kernel/detail/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/artifacts/*.deb
                    //install deb package
                    sh "dpkg -i *.deb"
                    //}

					dir("script"){
                        if (params.RUN_FULL_QA){
                            def qa_log = "qa_${gpu_arch}.log"
                            sh "./run_full_performance_tests.sh 1 QA_${params.COMPILER_VERSION} ${gpu_arch} ${env.BRANCH_NAME} ${NODE_NAME}"
                            archiveArtifacts "perf_gemm_${gpu_arch}.log"
                            archiveArtifacts "perf_resnet50_N256_${gpu_arch}.log"
                            archiveArtifacts "perf_resnet50_N4_${gpu_arch}.log"
                            archiveArtifacts "perf_batched_gemm_${gpu_arch}.log"
                            archiveArtifacts "perf_grouped_gemm_${gpu_arch}.log"
                            archiveArtifacts "perf_conv_fwd_${gpu_arch}.log"
                            archiveArtifacts "perf_conv_bwd_data_${gpu_arch}.log"
                            archiveArtifacts "perf_gemm_bilinear_${gpu_arch}.log"
                            archiveArtifacts "perf_reduction_${gpu_arch}.log"
                            archiveArtifacts "perf_splitK_gemm_verify_${gpu_arch}.log"
                            archiveArtifacts "perf_splitK_gemm_${gpu_arch}.log"
                            archiveArtifacts "perf_onnx_gemm_${gpu_arch}.log"
                           // stash perf files to master
                            stash name: "perf_gemm_${gpu_arch}.log"
                            stash name: "perf_resnet50_N256_${gpu_arch}.log"
                            stash name: "perf_resnet50_N4_${gpu_arch}.log"
                            stash name: "perf_batched_gemm_${gpu_arch}.log"
                            stash name: "perf_grouped_gemm_${gpu_arch}.log"
                            stash name: "perf_conv_fwd_${gpu_arch}.log"
                            stash name: "perf_conv_bwd_data_${gpu_arch}.log"
                            stash name: "perf_gemm_bilinear_${gpu_arch}.log"
                            stash name: "perf_reduction_${gpu_arch}.log"
                            stash name: "perf_splitK_gemm_${gpu_arch}.log"
                            stash name: "perf_onnx_gemm_${gpu_arch}.log"
                            //we will process results on the master node
                        }
                        else{
                            sh "./run_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${gpu_arch} ${env.BRANCH_NAME} ${NODE_NAME}"
                            archiveArtifacts "perf_gemm_${gpu_arch}.log"
                            archiveArtifacts "perf_resnet50_N256_${gpu_arch}.log"
                            archiveArtifacts "perf_resnet50_N4_${gpu_arch}.log"
                            // stash perf files to master
                            stash name: "perf_gemm_${gpu_arch}.log"
                            stash name: "perf_resnet50_N256_${gpu_arch}.log"
                            stash name: "perf_resnet50_N4_${gpu_arch}.log"
                            //we will process the results on the master node
                        }

					}
                }
            }
        }
        return retimage
}

def runPerfTest(Map conf=[:]){
    try{
        runCKProfiler(conf)
    }
    catch(e){
        echo "throwing error exception in performance tests"
        echo 'Exception occurred: ' + e.toString()
        throw e
    }
    finally{
        if (!conf.get("no_reboot", false)) {
            reboot()
        }
    }
}






def runTests_and_Examples(Map conf=[:]){
        show_node_info()

        env.HSA_ENABLE_SDMA=0
        checkout scm

        def image = getDockerImageName() 
        def prefixpath = conf.get("prefixpath", "/opt/rocm")
        def gpu_arch = conf.get("gpu_arch", "gfx908")

        // Jenkins is complaining about the render group 
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1 --env GPU_ARCH='${gpu_arch}' "
        }
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg compiler_version='${params.COMPILER_VERSION}' "
        if (params.COMPILER_VERSION != "release"){
            dockerOpts = dockerOpts + " --env HIP_CLANG_PATH='/llvm-project/build/bin' "
        }

        def variant = env.STAGE_NAME

        def retimage

        gitStatusWrapper(credentialsId: "${status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'composable_kernel') {
            try {
                (retimage, image) = getDockerImage(conf)
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES'){
                        sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo | tee clinfo.log'
                        if ( runShell('grep -n "Number of devices:.*. 0" clinfo.log') ){
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
            catch(Exception ex) {
                retimage = docker.build("${image}", dockerArgs + " --no-cache .")
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES'){
                        sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo |tee clinfo.log'
                        if ( runShell('grep -n "Number of devices:.*. 0" clinfo.log') ){
                            throw new Exception ("GPU not found")
                        }
                        else{
                            echo "GPU is OK"
                        }
                    }
                }
            }

            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 5, unit: 'HOURS')
                {
                    sh """
                        pwd
                        ls 
                        rm -rf build
                        mkdir build
                    """
                    dir("build"){
                        //wget http://micimaster.amd.com/blue/organizations/jenkins/MLLibs%2Fcomposable_kernel/detail/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/artifacts/*.deb
                        sh """
                            wget http://micimaster.amd.com/blue/organizations/jenkins/MLLibs%2Fcomposable_kernel/detail/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/artifacts/composablekernel-dev_*_amd64.deb
                            wget http://micimaster.amd.com/blue/organizations/jenkins/MLLibs%2Fcomposable_kernel/detail/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/artifacts/composablekernel-tests_*_amd64.deb
                            ls -ltr
                            dpkg -x composablekernel-dev_*_amd64.deb .
                            dpkg -x composablekernel-tests_*_amd64.deb .
                            make -j check
                        """
                    }
                }
            }
        }
        return retimage
}

def runTests(Map conf=[:]){
    try{
        runTests_and_Examples(conf)
    }
    catch(e){
        echo "throwing error exception in tests"
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
        checkout scm


        def image = getDockerImageName() 
        def prefixpath = conf.get("prefixpath", "/opt/rocm")
        def gpu_arch = conf.get("gpu_arch", "gfx908")

        // Jenkins is complaining about the render group 
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1 --env GPU_ARCH='${gpu_arch}' "
        }
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg compiler_version='${params.COMPILER_VERSION}' "
        if (params.COMPILER_VERSION != "release"){
            dockerOpts = dockerOpts + " --env HIP_CLANG_PATH='/llvm-project/build/bin' "
        }

        def variant = env.STAGE_NAME
        def retimage

        gitStatusWrapper(credentialsId: "${status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'composable_kernel') {
            try {
                (retimage, image) = getDockerImage(conf)
            }
            catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                echo "The job was cancelled or aborted"
                throw e
            }
            catch(Exception ex) {
                retimage = docker.build("${image}", dockerArgs + " --no-cache .")
            }
            withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
                timeout(time: 24, unit: 'HOURS')
                {
                    cmake_build(conf)
                    dir("build"){
                        sh 'make package'
                        //archiveArtifacts "Libs_composable_kernel_${env.BRANCH_NAME}.deb", fingerprint: true
                        archiveArtifacts artifacts: "*.deb", allowEmptyArchive: true, fingerprint: true
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
    def image = getDockerImageName() 
    def prefixpath = "/opt/rocm"
    def gpu_arch = conf.get("gpu_arch", "gfx908")

    // Jenkins is complaining about the render group 
    def dockerOpts="--cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    if (conf.get("enforce_xnack_on", false)) {
        dockerOpts = dockerOpts + " --env HSA_XNACK=1 --env GPU_ARCH='${gpu_arch}' "
    }
    def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg compiler_version='release' "

    def variant = env.STAGE_NAME
    def retimage

    gitStatusWrapper(credentialsId: "${status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'composable_kernel') {
        try {
            (retimage, image) = getDockerImage(conf)
        }
        catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
            echo "The job was cancelled or aborted"
            throw e
        }
    }

    withDockerContainer(image: image, args: dockerOpts + ' -v=/var/jenkins/:/var/jenkins') {
        timeout(time: 1, unit: 'HOURS'){
            try{
                dir("script"){
                    if (params.RUN_FULL_QA){
                        // unstash perf files to master
                        unstash "perf_gemm_${gpu_arch}.log"
                        unstash "perf_resnet50_N256_${gpu_arch}.log"
                        unstash "perf_resnet50_N4_${gpu_arch}.log"
                        unstash "perf_batched_gemm_${gpu_arch}.log"
                        unstash "perf_grouped_gemm_${gpu_arch}.log"
                        unstash "perf_conv_fwd_${gpu_arch}.log"
                        unstash "perf_conv_bwd_data_${gpu_arch}.log"
                        unstash "perf_gemm_bilinear_${gpu_arch}.log"
                        unstash "perf_reduction_${gpu_arch}.log"
                        unstash "perf_splitK_gemm_${gpu_arch}.log"
                        unstash "perf_onnx_gemm_${gpu_arch}.log"
                        sh "./process_qa_data.sh ${gpu_arch}"
                    }
                    else{
                        // unstash perf files to master
                        unstash "perf_gemm_${gpu_arch}.log"
                        unstash "perf_resnet50_N256_${gpu_arch}.log"
                        unstash "perf_resnet50_N4_${gpu_arch}.log"
                        sh "./process_perf_data.sh ${gpu_arch}"
                    }
                }
            }
            catch(e){
                echo "throwing error exception while processing performance test results"
                echo 'Exception occurred: ' + e.toString()
                throw e
            }
        }
    }
}

//launch develop branch daily at 23:00 in FULL_QA mode
CRON_SETTINGS = BRANCH_NAME == "develop" ? '''0 23 * * * % RUN_FULL_QA=true''' : ""

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
            defaultValue: true,
            description: "Force building docker image (default: true)")
        string(
            name: 'COMPILER_VERSION', 
            defaultValue: 'ck-9110', 
            description: 'Specify which version of compiler to use: ck-9110 (default), release, or amd-stg-open.')
        string(
            name: 'BUILD_COMPILER', 
            defaultValue: 'hipcc', 
            description: 'Specify whether to build CK with hipcc (default) or with clang.')
        booleanParam(
            name: "RUN_FULL_QA",
            defaultValue: false,
            description: "Select whether to run small set of performance tests (default) or full QA")
        booleanParam(
            name: "TEST_NODE_PERFORMANCE",
            defaultValue: false,
            description: "Test the node GPU performance (default: false)")
    }
    environment{
        dbuser = "${dbuser}"
        dbpassword = "${dbpassword}"
        dbsship = "${dbsship}"
        dbsshport = "${dbsshport}"
        dbsshuser = "${dbsshuser}"
        dbsshpassword = "${dbsshpassword}"
        status_wrapper_creds = "${status_wrapper_creds}"
        gerrit_cred="${gerrit_cred}"
        DOCKER_BUILDKIT = "1"
    }
    stages{
        stage("Build Docker"){
            when {
                expression { params.BUILD_DOCKER.toBoolean() }
            }
            parallel{
                stage('Docker /opt/rocm'){
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildDocker('/opt/rocm')
                    }
                }
            }
        }
        stage("Static checks") {
            when {
                beforeAgent true
                expression { !params.TEST_NODE_PERFORMANCE.toBoolean() }
            }
            parallel{
                // enable after we move from hipcc to hip-clang
                // stage('Tidy') {
                //     agent{ label rocmnode("nogpu") }
                //     environment{
                //         // setup_cmd = "CXX='/opt/rocm/bin/hipcc' cmake -DBUILD_DEV=On .. "
                //         build_cmd = "make -j\$(nproc) -k analyze"
                //     }
                //     steps{
                //         buildHipClangJobAndReboot(build_cmd: build_cmd, no_reboot:true, prefixpath: '/opt/rocm', build_type: 'debug')
                //     }
                // }
                stage('Clang Format') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                        execute_cmd = "find .. -iname \'*.h\' \
                                -o -iname \'*.hpp\' \
                                -o -iname \'*.cpp\' \
                                -o -iname \'*.h.in\' \
                                -o -iname \'*.hpp.in\' \
                                -o -iname \'*.cpp.in\' \
                                -o -iname \'*.cl\' \
                                | grep -v 'build/' \
                                | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-10 -style=file {} | diff - {}\'"
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_cmd: "", build_cmd: "", execute_cmd: execute_cmd, no_reboot:true)
                    }
                }
            }
        }
    
		stage("Build CK")
        {
            parallel
            {
                stage("Build CK targets")
                {
                    agent{ label rocmnode("nogpu") }
                    environment{
                        //setup_args = "${params.COMPILER_VERSION == "release" ? """ -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 --offload-arch=gfx90a -O3 " -DBUILD_DEV=On """ : """ -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 --offload-arch=gfx90a -O3 -Xclang -mlink-builtin-bitcode -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc" -DBUILD_DEV=On """}"
                        setup_args = "${params.COMPILER_VERSION == "release" ? """ -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=../install -D CMAKE_CXX_FLAGS="--offload-arch=gfx908 --offload-arch=gfx90a -O3 " """ : """ -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=../install -D CMAKE_CXX_FLAGS="--offload-arch=gfx908 --offload-arch=gfx90a -O3 -Xclang -mlink-builtin-bitcode -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc" """ }"
                        //execute_args = "${params.COMPILER_VERSION == "release" ? """ make -j ckProfiler && make -j examples && cd ../client_example && rm -rf build && mkdir build && cd build && cmake -D CMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 -O3" -D CMAKE_CXX_COMPILER="${build_compiler()}" .. && make -j """ : """ cd ../client_example && rm -rf build && mkdir build && cd build && cmake -D CMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 -O3 -Xclang -mlink-builtin-bitcode -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc" -D CMAKE_CXX_COMPILER="${build_compiler()}" .. && make -j """ }"
                        execute_args = "${params.COMPILER_VERSION == "release" ? """ cd ../client_example && rm -rf build && mkdir build && cd build && cmake -D CMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 --offload-arch=gfx90a -O3" -D CMAKE_CXX_COMPILER="${build_compiler()}" .. && make -j """ : """ cd ../client_example && rm -rf build && mkdir build && cd build && cmake -D CMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 --offload-arch=gfx90a -O3 -Xclang -mlink-builtin-bitcode -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc" -D CMAKE_CXX_COMPILER="${build_compiler()}" .. && make -j """ }"

                    }
                    steps{
                        Build_CK_and_Reboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                    }
                }
            }
        }

 		stage("Tests")
        {
            when {
                beforeAgent true
                expression { !params.TEST_NODE_PERFORMANCE.toBoolean() }
            }
            parallel
            {
                stage("Run Tests: gfx908")
                {
                    agent{ label rocmnode("gfx908")}
                    environment{
                        //setup_args = """ -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 -O3 " -DBUILD_DEV=On """
                        setup_args = "${params.COMPILER_VERSION == "release" ? """ -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 -O3 " -DBUILD_DEV=On """ : """ -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 -O3 -Xclang -mlink-builtin-bitcode -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc" -DBUILD_DEV=On """}"
                    }
                    steps{
                        runTests(setup_args:setup_args, config_targets: "check", no_reboot:true, build_type: 'Release', gpu_arch: "gfx908")
                    }
                }
                stage("Run Tests: gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_FULL_QA.toBoolean() }
                    }
                    options { retry(2) }
                    agent{ label rocmnode("gfx90a")}
                    environment{
                        //setup_args = """ -D CMAKE_CXX_FLAGS="--offload-arch=gfx90a -O3 " -DBUILD_DEV=On """
                        setup_args = "${params.COMPILER_VERSION == "release" ? """ -D CMAKE_CXX_FLAGS=" --offload-arch=gfx90a -O3 " -DBUILD_DEV=On """ : """ -D CMAKE_CXX_FLAGS=" --offload-arch=gfx90a -O3 -Xclang -mlink-builtin-bitcode -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc" -DBUILD_DEV=On """}"
                    }
                    steps{
                        runTests(setup_args:setup_args, config_targets: "check", no_reboot:true, build_type: 'Release', gpu_arch: "gfx90a")
                    }
                }
            }
        }       
        /*
        //at present this stage only builds binaries. 
        //we will now build all binaries in a separate stage.
        //once we have some tests to run in this stage, we can enable it again.
        stage("Client App")
        {
            when {
                beforeAgent true
                expression { !params.TEST_NODE_PERFORMANCE.toBoolean() }
            }
            parallel
            {
                stage("Run Client App")
                {
                    agent{ label rocmnode("gfx908")}
                    environment{
                        setup_args = "${params.COMPILER_VERSION == "release" ? """ -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=../install -D CMAKE_CXX_FLAGS="--offload-arch=gfx908 -O3 " """ : """ -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=../install -D CMAKE_CXX_FLAGS="--offload-arch=gfx908 -O3 -Xclang -mlink-builtin-bitcode -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc" """ }"
                        execute_args = "${params.COMPILER_VERSION == "release" ? """ cd ../client_example && rm -rf build && mkdir build && cd build && cmake -D CMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 -O3" -D CMAKE_CXX_COMPILER="${build_compiler()}" .. && make -j """ : """ cd ../client_example && rm -rf build && mkdir build && cd build && cmake -D CMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 -O3 -Xclang -mlink-builtin-bitcode -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc" -D CMAKE_CXX_COMPILER="${build_compiler()}" .. && make -j """ }"

                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                    }
                }
            }
        }
        */
        stage("Performance Tests")
        {
            parallel
            {
                stage("Run ckProfiler: gfx908")
                {
                    when {
                        beforeAgent true
                        expression { !params.RUN_FULL_QA.toBoolean() && !params.TEST_NODE_PERFORMANCE.toBoolean() }
                    }
                    options { retry(2) }
                    agent{ label rocmnode("gfx908")}
                    environment{
                        setup_args = "${params.COMPILER_VERSION == "release" ? """ -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 -O3 " -DBUILD_DEV=On """ : """ -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 -O3 -Xclang -mlink-builtin-bitcode -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc" -DBUILD_DEV=On """}"
                   }
                    steps{
                        runPerfTest(setup_args:setup_args, config_targets: "ckProfiler", no_reboot:true, build_type: 'Release', gpu_arch: "gfx908")
                    }
                }
                stage("Run ckProfiler: gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_FULL_QA.toBoolean() || params.TEST_NODE_PERFORMANCE.toBoolean() }
                    }
                    options { retry(2) }
                    agent{ label rocmnode("gfx90a")}
                    environment{
                        setup_args = "${params.COMPILER_VERSION == "release" ? """ -D CMAKE_CXX_FLAGS=" --offload-arch=gfx90a -O3 " -DBUILD_DEV=On """ : """ -D CMAKE_CXX_FLAGS=" --offload-arch=gfx90a -O3 -Xclang -mlink-builtin-bitcode -Xclang /opt/rocm/amdgcn/bitcode/oclc_abi_version_400.bc" -DBUILD_DEV=On """}"
                    }
                    steps{
                        runPerfTest(setup_args:setup_args, config_targets: "ckProfiler", no_reboot:true, build_type: 'Release', gpu_arch: "gfx90a")
                    }
                }
            }
        }
        stage("Process Performance Test Results")
        {
            parallel
            {
                stage("Process results for gfx908"){
                    when {
                        beforeAgent true
                        expression { !params.RUN_FULL_QA.toBoolean() && !params.TEST_NODE_PERFORMANCE.toBoolean() }
                    }
                    agent { label 'mici' }
                    steps{
                        process_results(gpu_arch: "gfx908")
                    }
                }
                stage("Process results for gfx90a"){
                    when {
                        beforeAgent true
                        expression { params.RUN_FULL_QA.toBoolean() || params.TEST_NODE_PERFORMANCE.toBoolean() }
                    }
                    agent { label 'mici' }
                    steps{
                        process_results(gpu_arch: "gfx90a")
                    }
                }
            }
        }

        /* enable after the cmake file supports packaging
        stage("Packages") {
            when {
                expression { params.BUILD_PACKAGES && params.TARGET_NOGPU && params.DATATYPE_NA }
            }
            parallel {
                stage("Package /opt/rocm") {
                    agent{ label rocmnode("nogpu") }
                    steps{
                        buildHipClangJobAndReboot( package_build: "true", prefixpath: '/opt/rocm', gpu_arch: "gfx906;gfx908;gfx90a")
                    }
                }
            }
        }
        */
    }
}

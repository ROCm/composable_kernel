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

def cmake_build(Map conf=[:]){

    def compiler = conf.get("compiler","/opt/rocm/bin/hipcc")
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

        def image = "composable_kernels"
        def prefixpath = conf.get("prefixpath", "/opt/rocm")
        def gpu_arch = conf.get("gpu_arch", "gfx908")

        // Jenkins is complaining about the render group 
        // def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1"
        }
        def dockerArgs
        if (params.USE_9110){
            dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg GPU_ARCH='${gpu_arch}' --build-arg compiler_version='9110' "
            dockerOpts = dockerOpts + " --env HIP_CLANG_PATH='/llvm-project/build/bin' "
        }
        else{
            dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg GPU_ARCH='${gpu_arch}' --build-arg compiler_version='release' "
        }

        def variant = env.STAGE_NAME

        def retimage

        gitStatusWrapper(credentialsId: "${status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'composable_kernel') {
            try {
                retimage = docker.build("${image}", dockerArgs + '.')
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

        def image = "composable_kernels"
        def prefixpath = conf.get("prefixpath", "/opt/rocm")
        def gpu_arch = conf.get("gpu_arch", "gfx908")

        // Jenkins is complaining about the render group 
        // def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1"
        }
        def dockerArgs
        if (params.USE_9110){
            dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg GPU_ARCH='${gpu_arch}' --build-arg compiler_version='9110' "
            dockerOpts = dockerOpts + " --env HIP_CLANG_PATH='/llvm-project/build/bin' "
        }
        else{
            dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg GPU_ARCH='${gpu_arch}' --build-arg compiler_version='release' "
        }

        def variant = env.STAGE_NAME
        def retimage

        gitStatusWrapper(credentialsId: "${status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'composable_kernel') {
            try {
                retimage = docker.build("${image}", dockerArgs + '.')
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
                    cmake_build(conf)
					dir("script"){
                        if (params.RUN_FULL_QA){
                            def qa_log = "qa_${gpu_arch}.log"
                            if (params.USE_9110){
                                sh "./run_full_performance_tests.sh 1 QA_9110 ${gpu_arch} ${env.BRANCH_NAME} ${NODE_NAME}"
                            }
                            else{
                                sh "./run_full_performance_tests.sh 1 QA_release ${gpu_arch} ${env.BRANCH_NAME} ${NODE_NAME}"
                            }
                            archiveArtifacts "perf_gemm_${gpu_arch}.log"
                            archiveArtifacts "perf_resnet50_N256_${gpu_arch}.log"
                            archiveArtifacts "perf_resnet50_N4_${gpu_arch}.log"
                            archiveArtifacts "perf_batched_gemm_${gpu_arch}.log"
                            archiveArtifacts "perf_grouped_gemm_${gpu_arch}.log"
                            archiveArtifacts "perf_fwd_conv_${gpu_arch}.log"
                            archiveArtifacts "perf_bwd_conv_${gpu_arch}.log"
                            archiveArtifacts "perf_fusion_${gpu_arch}.log"
                            archiveArtifacts "perf_reduction_${gpu_arch}.log"
                           // stash perf files to master
                            stash name: "perf_gemm_${gpu_arch}.log"
                            stash name: "perf_resnet50_N256_${gpu_arch}.log"
                            stash name: "perf_resnet50_N4_${gpu_arch}.log"
                            stash name: "perf_batched_gemm_${gpu_arch}.log"
                            stash name: "perf_grouped_gemm_${gpu_arch}.log"
                            stash name: "perf_fwd_conv_${gpu_arch}.log"
                            stash name: "perf_bwd_conv_${gpu_arch}.log"
                            stash name: "perf_fusion_${gpu_arch}.log"
                            stash name: "perf_reduction_${gpu_arch}.log"
                            //we will process results on the master node
                        }
                        else{
                            if (params.USE_9110){
                                sh "./run_performance_tests.sh 0 CI_9110 ${gpu_arch} ${env.BRANCH_NAME} ${NODE_NAME}"
                            }
                            else{
                                sh "./run_performance_tests.sh 0 CI_release ${gpu_arch} ${env.BRANCH_NAME} ${NODE_NAME}"
                            }
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

def process_results(Map conf=[:]){
    env.HSA_ENABLE_SDMA=0
    checkout scm
    def image = "composable_kernels"
    def prefixpath = "/opt/rocm"
    def gpu_arch = conf.get("gpu_arch", "gfx908")

    // Jenkins is complaining about the render group 
    def dockerOpts="--cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    if (conf.get("enforce_xnack_on", false)) {
        dockerOpts = dockerOpts + " --env HSA_XNACK=1"
    }
    def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg GPU_ARCH='${gpu_arch}' --build-arg compiler_version='release' "

    def variant = env.STAGE_NAME
    def retimage

    gitStatusWrapper(credentialsId: "${status_wrapper_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'composable_kernel') {
        try {
            retimage = docker.build("${image}", dockerArgs + '.')
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
                        unstash "perf_fwd_conv_${gpu_arch}.log"
                        unstash "perf_bwd_conv_${gpu_arch}.log"
                        unstash "perf_fusion_${gpu_arch}.log"
                        unstash "perf_reduction_${gpu_arch}.log"
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
CRON_SETTINGS = BRANCH_NAME == "develop" ? '''0 23 * * * % RUN_FULL_QA=true;USE_9110=true''' : ""

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
            name: "USE_9110",
            defaultValue: true,
            description: "Select compiler version: 9110 (default) or release")
        booleanParam(
            name: "RUN_FULL_QA",
            defaultValue: false,
            description: "Select whether to run small set of performance tests (default) or full QA")
    }
    environment{
        dbuser = "${dbuser}"
        dbpassword = "${dbpassword}"
        dbsship = "${dbsship}"
        dbsshport = "${dbsshport}"
        dbsshuser = "${dbsshuser}"
        dbsshpassword = "${dbsshpassword}"
        status_wrapper_creds = "${status_wrapper_creds}"
    }
    stages{
        stage("Static checks") {
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
		stage("Tests")
        {
            parallel
            {
                stage("Run Tests: gfx908")
                {
                    agent{ label rocmnode("gfx908")}
                    environment{
                        setup_args = """ -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 -O3 " -DBUILD_DEV=On """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, config_targets: "check", no_reboot:true, build_type: 'Release', gpu_arch: "gfx908")
                    }
                }
                stage("Run Tests: gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_FULL_QA.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a")}
                    environment{
                        setup_args = """ -D CMAKE_CXX_FLAGS="--offload-arch=gfx90a -O3 " -DBUILD_DEV=On """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, config_targets: "check", no_reboot:true, build_type: 'Release', gpu_arch: "gfx90a")
                    }
                }
            }
        }
        stage("Client App")
        {
            parallel
            {
                stage("Run Client App")
                {
                    agent{ label rocmnode("gfx908")}
                    environment{
                        setup_args = """ -D  -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=../install CMAKE_CXX_FLAGS="--offload-arch=gfx908 -O3 " """
                        execute_args = """ cd ../client_example && rm -rf build && mkdir build && cd build && cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/install;/opt/rocm" -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc .. && make -j """ 
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args: setup_args, config_targets: "install", no_reboot:true, build_type: 'Release', execute_cmd: execute_args, prefixpath: '/usr/local')
                    }
                }
            }
        }
        stage("Performance Tests")
        {
            parallel
            {
                stage("Run ckProfiler: gfx908")
                {
                    when {
                        beforeAgent true
                        expression { !params.RUN_FULL_QA.toBoolean() }
                    }
                    agent{ label rocmnode("gfx908")}
                    environment{
                        setup_args = """ -D CMAKE_CXX_FLAGS="--offload-arch=gfx908 -O3 " -DBUILD_DEV=On """
                   }
                    steps{
                        runPerfTest(setup_args:setup_args, config_targets: "ckProfiler", no_reboot:true, build_type: 'Release', gpu_arch: "gfx908")
                    }
                }
                stage("Run ckProfiler: gfx90a")
                {
                    when {
                        beforeAgent true
                        expression { params.RUN_FULL_QA.toBoolean() }
                    }
                    agent{ label rocmnode("gfx90a")}
                    environment{
                        setup_args = """ -D CMAKE_CXX_FLAGS="--offload-arch=gfx90a -O3 " -DBUILD_DEV=On """
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
                        expression { !params.RUN_FULL_QA.toBoolean() }
                    }
                    agent { label 'mici' }
                    steps{
                        process_results(gpu_arch: "gfx908")
                    }
                }
                stage("Process results for gfx90a"){
                    when {
                        beforeAgent true
                        expression { params.RUN_FULL_QA.toBoolean() }
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

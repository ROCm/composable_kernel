def rocmnode(name) {
    return 'rocmtest && miopen && ' + name
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
    def build_cmd = conf.get("build_cmd", "${build_envs} dumb-init make  -j\$(( \$(nproc) / 1 )) ${config_targets}")
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
        def dockerArgs = "--build-arg PREFIX=${prefixpath} --build-arg GPU_ARCH='${gpu_arch}' "

        def variant = env.STAGE_NAME


        def retimage
        gitStatusWrapper(credentialsId: '7126e5fe-eb51-4576-b52b-9aaf1de8f0fd', gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'composable_kernel') {
            try {
                retimage = docker.build("${image}", dockerArgs + '.')
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES')
                    {
                        sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo'
                    }
                }
            }
            catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                echo "The job was cancelled or aborted"
                throw e
            }
            catch(Exception ex) {
                retimage = docker.build("${image}", dockerArgs + "--no-cache .")
                withDockerContainer(image: image, args: dockerOpts) {
                    timeout(time: 5, unit: 'MINUTES')
                    {
                        sh 'PATH="/opt/rocm/opencl/bin:/opt/rocm/opencl/bin/x86_64:$PATH" clinfo'
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

pipeline {
    agent none
    options {
        parallelsAlwaysFailFast()
    }
    // environment{
	//  variable = value
    // }
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
                stage('Build Profiler: Release, gfx908')
                {
                    agent { label rocmnode("nogpu")}
                    environment{
                        setup_args = """ -D CMAKE_CXX_FLAGS="--offload-arch=gfx908 -O3 " -DBUILD_DEV=On """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, config_targets: "ckProfiler", no_reboot:true, build_type: 'Release')
                    }
                }
                stage('Build Profiler: Debug, gfx908')
                {
                    agent { label rocmnode("nogpu")}
                    environment{
                        setup_args = """ -D CMAKE_CXX_FLAGS="--offload-arch=gfx908 -O3 " -DBUILD_DEV=On """
                    }
                    steps{
                        // until we stabilize debug build due to compiler crashes
                        catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                            buildHipClangJobAndReboot(setup_args:setup_args, config_targets: "ckProfiler", no_reboot:true, build_type: 'Debug')
                        }
                    }
                }
                stage('Clang Format') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                        execute_cmd = "find . -iname \'*.h\' \
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
                        setup_args = """ -D CMAKE_CXX_FLAGS="--offload-arch=gfx908 -O3 " -DBUILD_DEV=On """
                    }
                    steps{
                        buildHipClangJobAndReboot(setup_args:setup_args, config_targets: "check", no_reboot:true, build_type: 'Release')
                    }

                }

            }
        }
        // enable after the cmake file supports packaging
        // stage("Packages") {
        //     when {
        //         expression { params.BUILD_PACKAGES && params.TARGET_NOGPU && params.DATATYPE_NA }
        //     }
        //     parallel {
        //         stage("Package /opt/rocm") {
        //             agent{ label rocmnode("nogpu") }
        //             steps{
        //             buildHipClangJobAndReboot( package_build: "true", prefixpath: '/opt/rocm', gpu_arch: "gfx906;gfx908;gfx90a")
        //             }
        //         }
        //     }
        // }
    }
}

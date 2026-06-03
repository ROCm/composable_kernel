@NonCPS
String getGitHubCommitHash(def build)
{
    def scmAction = build?.actions.find { action ->
        action instanceof jenkins.scm.api.SCMRevisionAction
    }
    if (scmAction?.revision instanceof org.jenkinsci.plugins.github_branch_source.PullRequestSCMRevision)
    {
        return scmAction.revision.pullHash
    }
    else if (scmAction?.revision instanceof jenkins.plugins.git.AbstractGitSCMSource$SCMRevisionImpl)
    {
        return scmAction.revision.hash
    }
    return null
}

def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME"
        hostname
        lsb_release -sd
        uname -r
        cat /sys/module/amdgpu/version
        ls /opt/ -la
    """
}

def setGithubStatus(String context, String state, String description) {
    def sha = env.GIT_COMMIT
    def targetUrl = env.RUN_DISPLAY_URL ?: env.BUILD_URL
    def statusUrl = "https://api.github.com/repos/ROCm/rocm-libraries/statuses/${sha}"
    withCredentials([usernamePassword(credentialsId: 'github-app-miopen', usernameVariable: 'GITHUB_APP', passwordVariable: 'GITHUB_TOKEN')]) {
        def code = '0'
        try {
            retry(3) {
                code = sh(returnStdout: true, script: """
                    curl -s -w "%{http_code}" -o /dev/null -X POST '${statusUrl}' \\
                        -H "Authorization: token \$GITHUB_TOKEN" \\
                        -H 'Content-Type: application/json' \\
                        -d '{"state":"${state}","context":"${context}","description":"${description}","target_url":"${targetUrl}"}'
                """).trim()
                if (!code.startsWith('2')) {
                    error("GitHub status POST returned ${code}")
                }
            }
        } catch (Exception e) {
            echo "WARNING: GitHub status POST failed after retries (context=${context}, state=${state}, code=${code})"
        }
    }
}

def cloneUpdateRefRepo() {
    def refRepoPath = "/var/jenkins/ref-repo/rocm-libraries"
    def lockLabel = "git ref repo lock - ${env.NODE_NAME}"
    def folderExists = sh(
        script: "test -d ${refRepoPath}/refs",
        returnStatus: true
    ) == 0

    if (!folderExists) {
        echo "rocm-libraries repo does not exist at ${refRepoPath}, creating mirror clone..."
        echo "locking on label: ${lockLabel}"
        lock(lockLabel) {
            def cloneCommand = """
                set -ex
                rm -rf ${refRepoPath} && mkdir -p ${refRepoPath}
                git clone --mirror https://github.com/ROCm/rocm-libraries.git ${refRepoPath}
            """
            sh(script: cloneCommand, label: "clone ref repo")
        }
        echo "Completed git clone, lock released"
    }
    echo "rocm-libraries repo exists at ${refRepoPath}, performing git remote update..."
    echo "locking on label: ${lockLabel}"
    lock(lockLabel) {
        def fetchCommand = """
            set -ex
            cd ${refRepoPath}
            git remote prune origin
            git remote update
        """
        sh(script: fetchCommand, label: "update ref repo")
    }
    echo "Completed git ref repo fetch, lock released"
}

def checkoutComposableKernel()
{
    //update ref repo
    cloneUpdateRefRepo()
    // checkout project
    def scmVars = checkout scm
    // getGitHubCommitHash reads SCMRevisionAction recorded before any local merge,
    // giving the true PR branch tip (pullHash) or branch HEAD (hash).
    // Falls back to ORIG_HEAD (pre-merge HEAD set by git merge) when SCMRevisionAction
    // is unavailable, then to HEAD for branch builds where no merge occurred.
    env.GIT_COMMIT = getGitHubCommitHash(currentBuild.rawBuild) ?: sh(returnStdout: true, script: '''
        git rev-parse ORIG_HEAD 2>/dev/null || git rev-parse HEAD
    ''').trim()
}

def generateAndArchiveBuildTraceVisualization(String buildTraceFileName) {
    try {
        checkoutComposableKernel()

        // Retrieve the build trace artifact
        def traceFileExists = false
        try {
            copyArtifacts(
                projectName: env.JOB_NAME,
                selector: specific(env.BUILD_NUMBER),
                filter: buildTraceFileName
            )
            traceFileExists = fileExists(buildTraceFileName)
        } catch (Exception e) {
            echo "Could not copy build trace artifact: ${e.getMessage()}"
            traceFileExists = false
            return
        }
        
        sh """
            echo "post artifact download:"
            ls -la
        """
        
        // Pull image
        def image = "ghcr.io/puppeteer/puppeteer:24.30.0"
        echo "Pulling image: ${image}"
        def retimage = docker.image("${image}")
        retimage.pull()

        // Create a temporary workspace
        sh """#!/bin/bash
            ls -la
            mkdir -p workspace
            cp ./projects/composablekernel/script/infra_helper/capture_build_trace.js ./workspace
            cp ${buildTraceFileName} ./workspace/${buildTraceFileName}
            chmod 777 ./workspace
            ls -la ./workspace
        """

        // Run container to get snapshot
        def dockerOpts = "--cap-add=SYS_ADMIN -v \"\$(pwd)/workspace:/workspace\" -e NODE_PATH=/home/pptruser/node_modules -e BUILD_TRACE_FILE=${buildTraceFileName}"
        // Create unique image name by sanitizing job name
        def sanitizedJobName = env.JOB_NAME.replaceAll(/[\/\\:*?"<>| ]/, '_').replaceAll('%2F', '_')
        def architectureName = (buildTraceFileName =~ /(gfx[0-9a-zA-Z]+)/)[0][1]
        def imageName = "perfetto_snapshot_${sanitizedJobName}_build_${env.BUILD_NUMBER}_${architectureName}.png"
        sh """
            docker run --rm ${dockerOpts} ${image} node /workspace/capture_build_trace.js
            mv ./workspace/perfetto_snapshot_build.png ./workspace/${imageName}
        """
        
        // Archive the snapshot
        sh """
            mv ./workspace/${imageName} ${imageName}
        """
        archiveArtifacts "${imageName}"

        // Notify the channel
        withCredentials([string(credentialsId: 'ck_ci_build_perf_webhook_url', variable: 'WEBHOOK_URL')]) {
        sh '''
            # Create build trace filename with build number based on the original filename
            BUILD_TRACE_WITH_NUMBER=$(echo "''' + buildTraceFileName + '''" | sed 's/.json/_''' + sanitizedJobName + '''_''' + env.BUILD_NUMBER + '''_''' + architectureName + '''.json/')
            
            # Convert image to base64
            echo "Converting image to base64..."
            IMAGE_BASE64=$(base64 -w 0 ''' + imageName + ''')
            echo "Image base64 length: ${#IMAGE_BASE64}"
            
            # Convert build trace to base64
            echo "Converting build trace to base64..."
            BUILD_TRACE_BASE64=$(base64 -w 0 ''' + buildTraceFileName + ''')
            echo "Build trace base64 length: ${#BUILD_TRACE_BASE64}"
            
            # Create JSON payload with base64 data
            echo "Creating JSON payload..."
            {
                printf '{\n'
                printf '    "jobName": "%s",\n' "''' + env.JOB_NAME + '''"
                printf '    "buildNumber": "%s",\n' "''' + env.BUILD_NUMBER + '''"
                printf '    "jobUrl": "%s",\n' "''' + env.RUN_DISPLAY_URL + '''"
                printf '    "imageName": "%s",\n' "''' + imageName + '''"
                printf '    "architecture": "%s",\n' "''' + architectureName + '''"
                printf '    "imageData": "%s",\n' "$IMAGE_BASE64"
                printf '    "buildTraceName": "%s",\n' "$BUILD_TRACE_WITH_NUMBER"
                printf '    "buildTraceData": "%s"\n' "$BUILD_TRACE_BASE64"
                printf '}\n'
            } > webhook_payload.json
            
            echo "JSON payload created, size: $(wc -c < webhook_payload.json) bytes"
            
            curl -X POST "${WEBHOOK_URL}" \
            -H "Content-Type: application/json" \
            -d @webhook_payload.json
            
            # Clean up temporary file
            rm -f webhook_payload.json
        '''
        }
    } catch (Exception e) {
        echo "Throwing error exception while generating build trace visualization"
        echo 'Exception occurred: ' + e.toString()
    }
}

def nthreads() {
    def nproc = sh(returnStdout: true, script: 'nproc')
    echo "Number of cores: ${nproc}"
    def n = nproc.toInteger()
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

def shouldRunCICheck() {
    // File patterns that should not trigger CI
    def skipFilePatterns = [
        /^projects\/composablekernel\/\.github\/.*/, // GitHub workflow files
        /^projects\/composablekernel\/docs\/.*/, // Documentation files
        /^projects\/composablekernel\/LICENSE$/, // License file
        /^projects\/composablekernel\/.*\.gitignore$/, // Git ignore files
        /^projects\/composablekernel\/.*\.md$/ // Markdown files
    ]
    
    try {
        // Always run if this is a base branch build
        def baseBranch = "develop"
        def isBaseBranchBuild = (env.CHANGE_ID == null && env.BRANCH_NAME == baseBranch)

        if (isBaseBranchBuild) {
            echo "Base branch (${baseBranch}) build detected - always running CI for safety"
            return true
        }

        // Get the list of changed files (all files touched in any commit, even if reverted)
        def changedFiles = sh(
            returnStdout: true,
            script: '''
                BASE_BRANCH="develop"

                if [ "$CHANGE_ID" != "" ]; then
                    # For PR builds, get all files touched in any commit
                    echo "PR build detected, checking all touched files against origin/$CHANGE_TARGET" >&2
                    git log --name-only --pretty=format: origin/$CHANGE_TARGET..HEAD -- projects/composablekernel/ | sort -u | grep -v '^$' || true
                else
                    # For feature branch builds, compare against merge-base with base branch
                    MERGE_BASE=$(git merge-base HEAD origin/$BASE_BRANCH 2>/dev/null || echo "HEAD~1")
                    echo "Branch build detected, checking all touched files since merge-base: $MERGE_BASE" >&2
                    git log --name-only --pretty=format: $MERGE_BASE..HEAD -- projects/composablekernel/ | sort -u | grep -v '^$' || true
                fi
            '''
        ).trim().split('\n')
        
        if (changedFiles.size() == 1 && changedFiles[0] == '') {
            echo "No changed files detected - this might be a manual trigger or merge commit, running CI for safety"
            return true
        }
        
        echo "Changed files: ${changedFiles.join(', ')}"
        
        // Separate files into those requiring CI and those that can be skipped
        def filesRequiringCI = []
        def skippedFiles = []

        changedFiles.each { file ->
            def shouldSkip = skipFilePatterns.any { pattern ->
                file ==~ pattern
            }

            if (shouldSkip) {
                skippedFiles.add(file)
            } else {
                filesRequiringCI.add(file)
            }
        }

        // Debug output
        if (skippedFiles.size() > 0) {
            echo "Files that don't require CI (${skippedFiles.size()}):"
            skippedFiles.each { echo "  - ${it}" }
        }

        if (filesRequiringCI.size() > 0) {
            echo "Files that require CI (${filesRequiringCI.size()}):"
            filesRequiringCI.each { echo "  - ${it}" }
            return true
        } else {
            echo "Only non-relevant files changed, skipping CI"
            return false
        }
    } catch (Exception e) {
        echo "Error checking changed files: ${e.getMessage()}, running CI by default"
        return true
    }
}

def getBaseDockerImageName(){
    def img
    if (params.USE_CUSTOM_DOCKER != ""){
        img = "${params.USE_CUSTOM_DOCKER}"
    }
    else{
        img = "${env.CK_DOCKERHUB}:ck_ub24.04_rocm${params.ROCMVERSION}"
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

def check_arch_name(){
    sh 'rocminfo | tee rocminfo.log'
    if ( runShell('grep -n "gfx90a" rocminfo.log') ){
        return "gfx90a"
    }
    else if ( runShell('grep -n "gfx942" rocminfo.log') ) {
        return "gfx942"
    }
    else if ( runShell('grep -n "gfx101" rocminfo.log') ) {
        return "gfx101"
    }
    else if ( runShell('grep -n "gfx103" rocminfo.log') ) {
        return "gfx103"
    }
    else if ( runShell('grep -n "gfx11" rocminfo.log') ) {
        return "gfx11"
    }
    else if ( runShell('grep -n "gfx120" rocminfo.log') ) {
        return "gfx12"
    }
    else if ( runShell('grep -n "gfx908" rocminfo.log') ) {
        return "gfx908"
    }
    else if ( runShell('grep -n "gfx950" rocminfo.log') ) {
        return "gfx950"
    }
    else {
        return ""
    }
}

def getDockerImage(Map conf=[:]){
    def image
    if ( conf.get("docker_name", "") != "" ){
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

// Build and push a docker image, capturing its digest into the specified env var.
// If forceBuild is false, will skip building if the image already exists in the registry.
def buildAndPushDockerImage(String install_prefix, String image_name, String dockerExtraArgs, boolean forceBuild){
    show_node_info()
    env.DOCKER_BUILDKIT=1
    checkoutComposableKernel()
    def dockerArgs = "--build-arg PREFIX=${install_prefix} --build-arg compiler_version='${params.COMPILER_VERSION}' --build-arg compiler_commit='${params.COMPILER_COMMIT}' --build-arg ROCMVERSION='${params.ROCMVERSION}' "
    dockerArgs += " " + dockerExtraArgs

    if(!forceBuild){
        try{
            echo "Checking for image: ${image_name}"
            withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
                sh "docker manifest inspect --insecure ${image_name}"
            }
            echo "Image: ${image_name} found! Skipping building image"
            return image_name
        }
        catch(Exception ex){
            echo "Unable to locate image: ${image_name}. Will attempt to build image now."
        }
    }

    echo "Building image: ${image_name} with args: ${dockerArgs}"
    def retimage = docker.build("${image_name}", dockerArgs)
    withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
        retimage.push()
    }
    def digest = sh(returnStdout: true, script: "docker inspect --format='{{index .RepoDigests 0}}' ${image_name}").trim()
    echo "Built image digest: ${digest}"
    echo "Pruning dangling Docker images to free disk space on CI agent"
    sh "docker image prune -f --filter 'dangling=true' || true"
    return digest
}

def buildDockerBase(install_prefix){
    def image_name = getDockerImageName()
    def base_image_name = getBaseDockerImageName()
    echo "Building Docker for ${image_name}"
    def dockerExtraArgs = " -f projects/composablekernel/Dockerfile . "
    if(params.COMPILER_VERSION == "develop" || params.COMPILER_VERSION == "amd-staging" || params.COMPILER_COMMIT != ""){
        dockerExtraArgs = " --no-cache --build-arg BASE_DOCKER='${base_image_name}' -f projects/composablekernel/Dockerfile.compiler . "
    }
    else if(params.COMPILER_VERSION == "therock"){
        dockerExtraArgs = " --no-cache -f projects/composablekernel/Dockerfile . "
    }
    env.CK_BASE_IMAGE = buildAndPushDockerImage(install_prefix, image_name, dockerExtraArgs, params.BUILD_DOCKER.toBoolean())
}

def buildDockerPytorch(install_prefix){
    def image_name = "${env.CK_DOCKERHUB_PRIVATE}:ck_pytorch"
    def dockerExtraArgs = " --no-cache -f projects/composablekernel/Dockerfile.pytorch --build-arg CK_PYTORCH_BRANCH='${params.ck_pytorch_branch}' . "
    env.CK_PYTORCH_IMAGE = buildAndPushDockerImage(install_prefix, image_name, dockerExtraArgs, true)
}

def buildDockerAiter(install_prefix){
    def image_name = "${env.CK_DOCKERHUB_PRIVATE}:ck_aiter"
    def dockerExtraArgs = " --no-cache -f projects/composablekernel/Dockerfile.aiter --build-arg AITER_BRANCH='${params.aiter_branch}' --build-arg CK_AITER_BRANCH='${params.ck_aiter_branch}' . "
    env.CK_AITER_IMAGE = buildAndPushDockerImage(install_prefix, image_name, dockerExtraArgs, true)
}

def buildDockerFa(install_prefix){
    def image_name = "${env.CK_DOCKERHUB_PRIVATE}:ck_fa"
    def dockerExtraArgs = " --no-cache -f projects/composablekernel/Dockerfile.fa"
    dockerExtraArgs += " --build-arg BASE_DOCKER='${params.fa_base_docker}'"
    dockerExtraArgs += " --build-arg FA_BRANCH='${params.fa_branch}'"
    dockerExtraArgs += " --build-arg CK_FA_BRANCH='${params.ck_fa_branch}'"
    dockerExtraArgs += " --build-arg GPU_ARCHS='gfx942;gfx950'"
    dockerExtraArgs += " . "
    env.CK_FA_IMAGE = buildAndPushDockerImage(install_prefix, image_name, dockerExtraArgs, true)
}

def buildDocker(install_prefix){
    buildDockerBase(install_prefix)
    if (params.RUN_PYTORCH_TESTS.toBoolean()) {
        buildDockerPytorch(install_prefix)
    }
    if (params.RUN_AITER_TESTS.toBoolean()) {
        buildDockerAiter(install_prefix)
    }
    if (params.RUN_FA_TESTS.toBoolean()) {
        buildDockerFa(install_prefix)
    }
}

def get_docker_options(){
    def dockerOpts
    if ( params.BUILD_INSTANCES_ONLY ){
        dockerOpts = "--network=host --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    }
    else{ //only add kfd and dri paths if you actually going to run somthing on GPUs
        dockerOpts = "--network=host --device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    }
    if (params.COMPILER_VERSION == "develop" || params.COMPILER_VERSION == "amd-staging" || params.COMPILER_VERSION == "therock" || params.COMPILER_COMMIT != ""){
    // the  --env COMPRESSED_BUNDLE_FORMAT_VERSION=2 env variable is required when building code with offload-compress flag with
    // newer clang22 compilers and running with older hip runtima libraries
        dockerOpts = dockerOpts + " --env HIP_CLANG_PATH='/llvm-project/build/bin' --env COMPRESSED_BUNDLE_FORMAT_VERSION=2 --env HIP_PLATFORM=amd "
    }
    // on some machines the group ids for video and render groups may not be the same as in the docker image!
    def video_id = sh(returnStdout: true, script: 'getent group video | cut -d: -f3')
    def render_id = sh(returnStdout: true, script: 'getent group render | cut -d: -f3')
    dockerOpts = dockerOpts + " --group-add=${video_id} --group-add=${render_id} -v /var/jenkins/ref-repo/:/var/jenkins/ref-repo/ "
    echo "Docker flags: ${dockerOpts}"
    return dockerOpts
}

def build_client_examples(String arch){
    def cmd = """ cd ../client_example && rm -rf build && mkdir build && cd build && \
                cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/projects/composablekernel/install;/opt/rocm" \
                -DGPU_TARGETS="${arch}" \
                -DCMAKE_CXX_COMPILER="${params.BUILD_COMPILER}" \
                -DCMAKE_HIP_COMPILER="${params.BUILD_COMPILER}" \
                -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
    return cmd
}

def build_client_examples_and_codegen_tests(String arch){
    def cmd = """ cd ../codegen && rm -rf build && mkdir build && cd build && \
                cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_CXX_COMPILER="${params.BUILD_COMPILER}" .. && \
                make -j64 check && \
                cd ../../client_example && rm -rf build && mkdir build && cd build && \
                cmake -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/projects/composablekernel/install;/opt/rocm" \
                -DGPU_TARGETS="${arch}" \
                -DCMAKE_CXX_COMPILER="${params.BUILD_COMPILER}" \
                -DCMAKE_HIP_COMPILER="${params.BUILD_COMPILER}" \
                -DCMAKE_CXX_FLAGS=" -O3 " .. && make -j """
    return cmd
}

def build_and_run_fmha(String arch){
    def cmd = """ cmake -G Ninja -DCMAKE_PREFIX_PATH="${env.WORKSPACE}/projects/composablekernel/install;/opt/rocm" \
                -DGPU_TARGETS="${arch}" \
                -DCMAKE_CXX_COMPILER="${params.BUILD_COMPILER}" \
                -DCMAKE_HIP_COMPILER="${params.BUILD_COMPILER}" .. && \
                ninja -j128 tile_example_fmha_fwd tile_example_fmha_bwd && \
                cd ../ &&
                example/ck_tile/01_fmha/script/run_full_test.sh "CI_${params.COMPILER_VERSION}" "${env.BRANCH_NAME}" "${NODE_NAME}" "${arch}" """
    return cmd
}

def cmake_build(Map conf=[:]){

    def config_targets = conf.get("config_targets","check")
    def build_envs = "CTEST_PARALLEL_LEVEL=4 " + conf.get("build_env","")
    def prefixpath = conf.get("prefixpath","/opt/rocm")
    def setup_args = conf.get("setup_args","")
    // make sure all unit tests always run on develop branch
    def runAllUnitTests = (env.BRANCH_NAME == "develop") ? true : params.RUN_ALL_UNIT_TESTS

    if (prefixpath != "/usr/local"){
        setup_args = setup_args + " -DCMAKE_PREFIX_PATH=${prefixpath} "
    }

    //cmake_env can overwrite default CXX variables.
    def cmake_envs
    if(!setup_args.contains("gfx1250")){
        cmake_envs = "CXX=${params.BUILD_COMPILER} CXXFLAGS='-Werror' " + conf.get("cmake_ex_env","")
    }
    else{ //use default compiler for gfx1250
        cmake_envs = "CXX=/opt/rocm/llvm/bin/clang++ CXXFLAGS='-Werror' " + conf.get("cmake_ex_env","")
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

    setup_args = " -DCMAKE_BUILD_TYPE=release " + setup_args

    def pre_setup_cmd = """
            #!/bin/bash
            cd projects/composablekernel
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
    if (setup_args.contains("gfx101")){
        invocation_tag="gfx101"
    }
    if (setup_args.contains("gfx103")){
        invocation_tag="gfx103"
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
            setup_args = " -DCMAKE_HIP_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_C_COMPILER_LAUNCHER=sccache " + setup_args
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
    //check the node gpu architecture
    def arch_name = check_arch_name()
    if(!setup_args.contains("NO_CK_BUILD")){
        if (params.NINJA_BUILD_TRACE) {
            echo "running ninja build trace"
        }
        if (params.RUN_BUILDER_TESTS && !setup_args.contains("-DCK_CXX_STANDARD=") && !setup_args.contains("gfx10") && !setup_args.contains("gfx11")) {
            setup_args = " -D CK_EXPERIMENTAL_BUILDER=ON "  + setup_args
        }
        if (params.RUN_ROCM_CK_TESTS) {
            setup_args = " -D CK_ENABLE_ROCM_CK=ON " + setup_args
        }
        setup_cmd = conf.get(
            "setup_cmd",
            """${cmake_envs} cmake -G Ninja ${setup_args} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_FLAGS=" -O3 " .. """
        )

        // Smart-build: Only build if running all tests or forced
        // Otherwise, smart-build will determine what to build after cmake configure
        if (runAllUnitTests) {
            build_cmd = conf.get(
                "build_cmd",
                "${build_envs} ninja -j${nt} ${config_targets}"
            )
        } else {
            // Smart-build enabled: skip full build and execute_cmd (client examples)
            build_cmd = ""
            execute_cmd = ""
        }

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

    dir("projects/composablekernel/build"){
        // Start sccache monitoring
        if(check_host() && params.USE_SCCACHE && "${env.CK_SCCACHE}" != "null" && "${invocation_tag}" != "") {
            sh """
                chmod +x ../script/monitor_sccache_during_build.sh
                mkdir -p logs
                export SCCACHE_C_CUSTOM_CACHE_BUSTER="${invocation_tag}"
                ../script/monitor_sccache_during_build.sh build_monitor &
                MONITOR_PID=\$!
                echo "Monitor PID: \$MONITOR_PID"
                echo \$MONITOR_PID > monitor.pid
            """
        }
        try {
            //build CK
            sh cmd
            if (runAllUnitTests){
                // Archive artifacts if they were generated
                if (fileExists("ck_build_trace_${arch_name}.json")) {
                    archiveArtifacts "ck_build_trace_${arch_name}.json"
                }
                if (fileExists("clang_build_analysis_${arch_name}.log")) {
                    archiveArtifacts "clang_build_analysis_${arch_name}.log"
                }
                // Process ninja build trace after full build
                if(fileExists(".ninja_log")) {
                    sh "python3 ../script/ninja_json_converter.py .ninja_log --legacy-format --output ck_build_trace_${arch_name}.json"
                    archiveArtifacts "ck_build_trace_${arch_name}.json"
                    sh "python3 ../script/parse_ninja_trace.py ck_build_trace_${arch_name}.json"
                }

                if (params.NINJA_FTIME_TRACE) {
                    echo "running ClangBuildAnalyzer"
                    sh "/ClangBuildAnalyzer/build/ClangBuildAnalyzer  --all . clang_build.log"
                    sh "/ClangBuildAnalyzer/build/ClangBuildAnalyzer  --analyze clang_build.log > clang_build_analysis_${arch_name}.log"
                    archiveArtifacts "clang_build_analysis_${arch_name}.log"
                }
            }
        } catch (Exception buildError) {
            echo "Build failed: ${buildError.getMessage()}"
            throw buildError
        } finally {
            // Stop sccache monitoring
            if(check_host() && params.USE_SCCACHE && "${env.CK_SCCACHE}" != "null" && "${invocation_tag}" != "") {
                sh """
                    # Stop monitoring
                    if [ -f monitor.pid ]; then
                        MONITOR_PID=\$(cat monitor.pid)
                        kill \$MONITOR_PID 2>/dev/null || echo "Monitor already stopped"
                        rm -f monitor.pid
                    fi
                """
                
                // Archive the monitoring logs
                try {
                    archiveArtifacts artifacts: "logs/*monitor*.log", allowEmptyArchive: true
                } catch (Exception e) {
                    echo "Could not archive sccache monitoring logs: ${e.getMessage()}"
                }
            }
        }

        //run tests except when NO_CK_BUILD is set and except on gfx1250
        if(!setup_args.contains("NO_CK_BUILD")){
            // run unit tests unless building library for all targets
            // Note: This else block is when NINJA_BUILD_TRACE=false and BUILD_INSTANCES_ONLY=false
            // So no ninja trace processing needed here
            if (!params.BUILD_INSTANCES_ONLY){
                if (!runAllUnitTests && !setup_args.contains("gfx1250") ){
                    // Smart Build: Run smart_build_and_test.sh
                    sh """
                        export WORKSPACE_ROOT=${env.WORKSPACE}
                        export PARALLEL=32
                        export NINJA_JOBS=${nt}
                        export ARCH_NAME=${arch_name}
                        export PROCESS_NINJA_TRACE=false
                        export NINJA_FTIME_TRACE=false
                        bash ../script/dependency-parser/smart_build_and_test.sh
                    """
                }
                else{ //run all tests
                    if(!setup_args.contains("gfx1250")){
                        echo "Full test suite requested (RUN_ALL_UNIT_TESTS=true or develop branch)"
                        sh "ninja -j${nt} check"
                    }
                    else{ //do not run tests on gfx1250, just build everything
                        echo "Building for gfx1250"
                        sh "ninja -j${nt}"
                    }
                    if (params.RUN_ROCM_CK_TESTS) {
                        sh 'ninja check-rocm-ck'
                    }
                    if(params.BUILD_PACKAGES || params.BUILD_INSTANCES_ONLY){
                        echo "Build ckProfiler packages"
                        sh 'ninja -j64 package'
                        sh "mv composablekernel-ckprofiler_*.deb composablekernel-ckprofiler_1.2.0_amd64_${arch_name}.deb"
                        stash includes: "composablekernel-ckprofiler**.deb", name: "profiler_package_${arch_name}"
                    }
                }
                if (params.RUN_BUILDER_TESTS && !setup_args.contains("-DCK_CXX_STANDARD=") && !setup_args.contains("gfx10") && !setup_args.contains("gfx11")) {
                    sh 'ninja check-builder'
                }
            }
        }
    }

    if (params.RUN_CK_TILE_FMHA_TESTS){
        try{
            dir("projects/composablekernel"){
            	archiveArtifacts "perf_fmha_*.log"
            	stash includes: "perf_fmha_**.log", name: "perf_fmha_log_${arch_name}"
            }
        }
        catch(Exception err){
            echo "could not locate the requested artifacts: ${err.getMessage()}. will skip the stashing."
        }
    }
}

def buildHipClangJob(Map conf=[:]){
        show_node_info()
        checkoutComposableKernel()
        def prefixpath = conf.get("prefixpath", "/opt/rocm")
        def dockerOpts = get_docker_options()
        def image
        def retimage
        (retimage, image) = getDockerImage(conf)

        setGithubStatus("${env.STAGE_NAME}", 'pending', "Starting ${env.STAGE_NAME}")
        try {
            withDockerContainer(image: image, args: dockerOpts) {
                timeout(time: 20, unit: 'HOURS')
                {
                    cmake_build(conf)
                }
            }
            setGithubStatus("${env.STAGE_NAME}", 'success', "Stage ${env.STAGE_NAME} passed")
        }
        catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                setGithubStatus("${env.STAGE_NAME}", 'failure', "Stage ${env.STAGE_NAME} failed")
                throw e
        }
        return retimage
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
}

def Build_CK(Map conf=[:]){
        show_node_info()
        checkoutComposableKernel()
        def prefixpath = conf.get("prefixpath", "/opt/rocm")
        def dockerOpts=get_docker_options()
        def image
        def retimage

        setGithubStatus("${env.STAGE_NAME}", 'pending', "Starting ${env.STAGE_NAME}")
        try {
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
                setGithubStatus("${env.STAGE_NAME}", 'failure', "Stage ${env.STAGE_NAME} failed")
                throw e
            }
            withDockerContainer(image: image, args: dockerOpts) {
                timeout(time: 20, unit: 'HOURS')
                {
                    //check whether to run performance tests on this node
                    def arch = check_arch_name()
                    cmake_build(conf)
                    if ( params.RUN_INDUCTOR_TESTS && arch == "gfx90a" ){
                            echo "Run inductor codegen tests"
                            sh "projects/composablekernel/script/run_inductor_tests.sh"
                    }
                    // run performance tests, stash the logs, results will be processed on the master node
                    dir("projects/composablekernel/script"){
                        if (params.RUN_PERFORMANCE_TESTS){
                            if (params.RUN_FULL_QA && (arch == "gfx90a" || arch == "gfx942")){
                                // run full tests on gfx90a or gfx942
                                echo "Run full performance tests"
                                sh "./run_full_performance_tests.sh 0 QA_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} ${arch}"
                                archiveArtifacts "perf_*.log"
                                stash includes: "perf_**.log", name: "perf_log_${arch}"
                            }
                            else if (!params.RUN_FULL_QA && (arch == "gfx90a" || arch == "gfx942")){
                                // run standard tests on gfx90a or gfx942
                                echo "Run performance tests"
                                sh "./run_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} ${arch}"
                                archiveArtifacts "perf_*.log"
                                stash includes: "perf_**.log", name: "perf_log_${arch}"
                            }
                            else if ( arch != "gfx10"){
                                // run basic tests on gfx11/gfx12/gfx908/gfx950, but not on gfx10, it takes too long
                                echo "Run gemm performance tests"
                                sh "./run_gemm_performance_tests.sh 0 CI_${params.COMPILER_VERSION} ${env.BRANCH_NAME} ${NODE_NAME} ${arch}"
                                archiveArtifacts "perf_onnx_gemm_*.log"
                                stash includes: "perf_onnx_gemm_**.log", name: "perf_log_${arch}"
                            }
                        }
                    }
                    if (params.hipTensor_test && arch == "gfx90a" ){
                        // build and test hipTensor on gfx90a node
                        sh """#!/bin/bash
                            rm -rf rocm-libraries
                            git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
                            cd rocm-libraries
                            git sparse-checkout init --cone
                            git sparse-checkout set projects/hiptensor
                            git checkout "${params.hipTensor_branch}"
                        """
                        dir("rocm-libraries/projects/hiptensor"){
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
            setGithubStatus("${env.STAGE_NAME}", 'success', "Stage ${env.STAGE_NAME} passed")
        }
        catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                setGithubStatus("${env.STAGE_NAME}", 'failure', "Stage ${env.STAGE_NAME} failed")
                throw e
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
}

def process_results(Map conf=[:]){
    checkoutComposableKernel()
    //use older image that has user jenkins
    def image = "${env.CK_DOCKERHUB}:ck_ub22.04_rocm6.3"

    setGithubStatus("${env.STAGE_NAME}", 'pending', 'Processing results...')
    try {
        try
        {
            echo "Pulling image: ${image}"
            def retimage = docker.image("${image}")
            withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
                retimage.pull()
            }
        }
        catch(Exception ex)
        {
            error "Unable to locate image: ${image}"
        }
    }
    catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                setGithubStatus("${env.STAGE_NAME}", 'failure', "Stage ${env.STAGE_NAME} failed")
                throw e
    }

    withDockerContainer(image: image, args: '--cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v=/var/jenkins/:/var/jenkins') {
        timeout(time: 15, unit: 'MINUTES'){
            try{
                dir("projects/composablekernel/script"){
                    if (params.RUN_CK_TILE_FMHA_TESTS){
                        try{
                            unstash "perf_fmha_log_gfx942"
                        }
                        catch(Exception err){
                            echo "could not locate the FMHA performance logs for gfx942: ${err.getMessage()}."
                        }
                        try{
                            unstash "perf_fmha_log_gfx90a"
                        }
                        catch(Exception err){
                            echo "could not locate the FMHA performance logs for gfx90a: ${err.getMessage()}."
                        }
                        try{
                            unstash "perf_fmha_log_gfx950"
                        }
                        catch(Exception err){
                            echo "could not locate the FMHA performance logs for gfx950: ${err.getMessage()}."
                        }

                    }
                    if (params.BUILD_INSTANCES_ONLY){
                        // unstash deb packages
                        try{
                            unstash "lib_package"
                            sh "sshpass -p ${env.ck_deb_pw} scp -o StrictHostKeyChecking=no composablekernel-*.deb ${env.ck_deb_user}@${env.ck_deb_ip}:/var/www/html/composable_kernel/"
                        }
                        catch(Exception err){
                            echo "could not locate lib_package."
                        }
                    }
                    if (params.BUILD_PACKAGES){
                        // unstash deb packages
                        try{
                            unstash "profiler_package_gfx90a"
                            sh "sshpass -p ${env.ck_deb_pw} scp -o StrictHostKeyChecking=no composablekernel-ckprofiler*.deb ${env.ck_deb_user}@${env.ck_deb_ip}:/var/www/html/composable_kernel/"
                        }
                        catch(Exception err){
                            echo "could not locate profiler_package_gfx90a."
                        }
                        try{
                            unstash "profiler_package_gfx942"
                            sh "sshpass -p ${env.ck_deb_pw} scp -o StrictHostKeyChecking=no composablekernel-ckprofiler*.deb ${env.ck_deb_user}@${env.ck_deb_ip}:/var/www/html/composable_kernel/"
                        }
                        catch(Exception err){
                            echo "could not locate profiler_package_gfx942."
                        }
                        try{
                            unstash "profiler_package_gfx950"
                            sh "sshpass -p ${env.ck_deb_pw} scp -o StrictHostKeyChecking=no composablekernel-ckprofiler*.deb ${env.ck_deb_user}@${env.ck_deb_ip}:/var/www/html/composable_kernel/"
                        }
                        catch(Exception err){
                            echo "could not locate profiler_package_gfx950."
                        }
                        try{
                            unstash "profiler_package_gfx12"
                            sh "sshpass -p ${env.ck_deb_pw} scp -o StrictHostKeyChecking=no composablekernel-ckprofiler*.deb ${env.ck_deb_user}@${env.ck_deb_ip}:/var/www/html/composable_kernel/"
                        }
                        catch(Exception err){
                            echo "could not locate profiler_package_gfx12."
                        }
                    }
                    else{
                        // unstash perf files to master
                        try{
                            unstash "perf_log_gfx90a"
                        }
                        catch(Exception err){
                            echo "could not locate the gfx90a performance logs: ${err.getMessage()}."
                        }
                        try{
                            unstash "perf_log_gfx942"
                        }
                        catch(Exception err){
                            echo "could not locate the gfx942 performance logs: ${err.getMessage()}."
                        }
                        try{
                            unstash "perf_log_gfx950"
                        }
                        catch(Exception err){
                            echo "could not locate the gfx950 performance logs: ${err.getMessage()}."
                        }
                        try{
                            unstash "perf_log_gfx908"
                        }
                        catch(Exception err){
                            echo "could not locate the gfx908 performance logs: ${err.getMessage()}."
                        }
                        try{
                            unstash "perf_log_gfx11"
                        }
                        catch(Exception err){
                            echo "could not locate the gfx11 performance logs: ${err.getMessage()}."
                        }
                        try{

                            unstash "perf_log_gfx12"
                        }
                        catch(Exception err){
                            echo "could not locate the gfx12 performance logs: ${err.getMessage()}."
                        }
                    }
                    // process the logs
                    sh "./process_perf_data.sh"
                }
                setGithubStatus("${env.STAGE_NAME}", 'success', "Stage ${env.STAGE_NAME} passed")
            }
            catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                setGithubStatus("${env.STAGE_NAME}", 'failure', "Stage ${env.STAGE_NAME} failed")
                throw e
            }
            finally{
                echo "Finished processing performance test results"
            }
        }
    }
}

def run_downstream_tests(Map conf=[:]){
    show_node_info()
    checkoutComposableKernel()
    def dockerOpts = get_docker_options() + ' --group-add irc '

    setGithubStatus("${env.STAGE_NAME}", 'pending', "Starting ${env.STAGE_NAME}")
    try {
        try
        {
            echo "Pulling image: ${conf.image}"
            retimage = docker.image("${conf.image}")
            withDockerRegistry([ credentialsId: "ck_docker_cred", url: "" ]) {
                retimage.pull()
            }
        }
        catch(Exception ex)
        {
            error "Unable to locate image: ${conf.image}"
        }
    }
    catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                setGithubStatus("${env.STAGE_NAME}", 'failure', "Stage ${env.STAGE_NAME} failed")
                throw e
    }

    withDockerContainer(image: conf.image, args: dockerOpts) {
        timeout(time: conf.get("timeoutHours", 2), unit: 'HOURS'){
            try{
                sh "rocminfo"
                sh "python3 --version"
                for (cmd in conf.execute_cmds) {
                    sh "${cmd}"
                }
                setGithubStatus("${env.STAGE_NAME}", 'success', "Stage ${env.STAGE_NAME} passed")
            }
            catch(e){
                echo "Throwing error exception while running ${env.STAGE_NAME}"
                echo 'Exception occurred: ' + e.toString()
                setGithubStatus("${env.STAGE_NAME}", 'error', "Stage ${env.STAGE_NAME} failed")
                throw e
            }
            finally{
                echo "Finished running ${env.STAGE_NAME}"
            }
        }
    }
}

def getPytorchTestsCmds() {
    return [
        "mkdir pytorch",
        "cp -r /var/jenkins/workspace/pytorch/* pytorch/",
        "ls -ltr pytorch",
        "python3 pytorch/tools/amd_build/build_amd.py",
        "cd pytorch && USE_ROCM_CK_SDPA=1 PYTORCH_ROCM_ARCH=gfx942 python3 setup.py develop"
    ]
}
def getAiterTestsCmds() {
    return [
        // Pre-compile FlyDSL MoE AOT cache before the tests.
        "cd /home/jenkins/workspace/aiter && python3 aiter/aot/flydsl/moe.py",
        "python3 /home/jenkins/workspace/aiter/op_tests/test_gemm_a8w8.py",
        "python3 /home/jenkins/workspace/aiter/op_tests/test_gemm_a8w8_blockscale.py",
        "python3 /home/jenkins/workspace/aiter/op_tests/test_mha.py",
        "python3 /home/jenkins/workspace/aiter/op_tests/test_mha_varlen.py",
        "python3 /home/jenkins/workspace/aiter/op_tests/test_batch_prefill.py",
        "python3 /home/jenkins/workspace/aiter/op_tests/test_moe.py",
        "python3 /home/jenkins/workspace/aiter/op_tests/test_moe_2stage.py",
        "python3 /home/jenkins/workspace/aiter/op_tests/test_moe_blockscale.py",
        "python3 /home/jenkins/workspace/aiter/op_tests/test_moe_ep.py",
        "python3 /home/jenkins/workspace/aiter/op_tests/test_moe_sorting.py",
        "python3 /home/jenkins/workspace/aiter/op_tests/test_moe_sorting_mxfp4.py",
        "python3 /home/jenkins/workspace/aiter/op_tests/test_moe_tkw1.py"
    ]
}
def getFaTestsCmds() {
    return [
        "python3 -u -m pytest /home/jenkins/workspace/flash-attention/tests/test_flash_attn_ck.py"
    ]
}

def runClangFormat() {
    buildHipClangJobAndReboot(
        setup_args: "NO_CK_BUILD",
        setup_cmd: "",
        build_cmd: "",
        execute_cmd: """cd .. && \
            find . -type f \\( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.h.in' -o -name '*.hpp.in' -o -name '*.cpp.in' -o -name '*.cl' \\) \
            -not -path '*/build/*' -not -path '*/include/rapidjson/*' | \
            xargs -P 8 -I{} sh -c 'clang-format-18 -style=file {} | diff -u - {} || (echo "ERROR: {} needs formatting" && exit 1)'"""
    )
}

def runClangFormatAndCppcheck() {
    buildHipClangJobAndReboot(
        setup_args: "NO_CK_BUILD",
        setup_cmd: "",
        build_cmd: "",
        execute_cmd: """cd .. && \
            find . -type f \\( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.h.in' -o -name '*.hpp.in' -o -name '*.cpp.in' -o -name '*.cl' \\) \
            -not -path '*/build/*' -not -path '*/include/rapidjson/*' | \
            xargs -P 8 -I{} sh -c 'clang-format-18 -style=file {} | diff -u - {} || (echo "ERROR: {} needs formatting" && exit 1)' && \
            /cppcheck/build/bin/cppcheck ../* -v -j \$(nproc) -I ../include -I ../profiler/include -I ../library/include \
            -D CK_ENABLE_FP64 -D CK_ENABLE_FP32 -D CK_ENABLE_FP16 -D CK_ENABLE_FP8 -D CK_ENABLE_BF16 -D CK_ENABLE_BF8 -D CK_ENABLE_INT8 \
            -D __gfx908__ -D __gfx90a__ -D __gfx942__ -D __gfx1030__ -D __gfx1100__ -D __gfx1101__ -D __gfx1102__ \
            -U __gfx803__ -U __gfx900__ -U __gfx906__ -U CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4 \
            --file-filter=*.cpp --force --enable=all --output-file=ck_cppcheck.log"""
    )
}

def runFullGroupedConvTileTests() {
    buildHipClangJobAndReboot(
        setup_args: "NO_CK_BUILD",
        build_type: 'Release',
        execute_cmd: """
            python3 ../experimental/grouped_convolution_tile_instances/generate_instances.py --mode=profiler && \
            cmake .. --preset dev-gfx90a -D CK_EXPERIMENTAL_BUILDER=ON && \
            make -j64 test_grouped_convnd_fwd_tile test_grouped_convnd_bwd_weight_tile && \
            ./bin/test_grouped_convnd_bwd_weight_tile && \
            ./bin/test_grouped_convnd_fwd_tile"""
    )
}

def runGroupedConvLargeCaseTests() {
    buildHipClangJobAndReboot(
        setup_args: "NO_CK_BUILD",
        build_type: 'Release',
        execute_cmd: """
            cmake .. --preset dev-gfx90a && \
            make -j64 test_grouped_convnd_fwd_large_cases test_grouped_convnd_bwd_data_large_cases test_grouped_convnd_fwd_bias_clamp_large_cases && \
            ./bin/test_grouped_convnd_fwd_large_cases && \
            ./bin/test_grouped_convnd_bwd_data_large_cases && \
            ./bin/test_grouped_convnd_fwd_bias_clamp_large_cases"""
    )
}

def runComprehensiveConvDatasetTests() {
    buildHipClangJobAndReboot(
        setup_args: "NO_CK_BUILD",
        build_type: 'Release',
        execute_cmd: """
            cd ../build && \
            cmake .. --preset dev-gfx90a && \
            make -j64 test_grouped_convnd_fwd_dataset_xdl \
                      test_grouped_convnd_bwd_data_dataset_xdl \
                      test_grouped_convnd_bwd_weight_dataset_xdl && \
            cd ../test_data && \
            ./generate_test_dataset.sh small && \
            cd ../build && \
            ./bin/test_grouped_convnd_fwd_dataset_xdl && \
            ./bin/test_grouped_convnd_bwd_data_dataset_xdl && \
            ./bin/test_grouped_convnd_bwd_weight_dataset_xdl"""
    )
}

def runTileEngineBasicTests(String compiler) {
    buildHipClangJobAndReboot(
        setup_args: "NO_CK_BUILD",
        build_type: 'Release',
        execute_cmd: """
            cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                -D BUILD_CK_TILE_ENGINE="ON" \
                -D CMAKE_CXX_COMPILER="${compiler}" \
                -D CMAKE_BUILD_TYPE=Release \
                -D GPU_TARGETS="gfx942" \
                -D GEMM_UNIVERSAL_DATATYPE="fp8;fp16" \
                -D GEMM_UNIVERSAL_LAYOUT="rcr;rrr;crr;ccr" \
                -D GEMM_UNIVERSAL_CONFIG_FILE="default_ci_config.json" \
                -D GEMM_MULTI_D_DATATYPE="fp16" \
                -D GEMM_MULTI_D_LAYOUT="rcrr;rrrr;crrr;ccrr" \
                -D GEMM_MULTI_D_CONFIG_FILE="default_ci_config.json" \
                -D GEMM_PRESHUFFLE_DATATYPE="fp16;fp8;bf16;bf8" \
                -D GEMM_PRESHUFFLE_LAYOUT="rcr" \
                -D GEMM_PRESHUFFLE_CONFIG_FILE="default_ci_config.json" .. && \
            ninja -j${nthreads()} benchmark_gemm_universal_all benchmark_gemm_preshuffle_all benchmark_gemm_multi_d_all && \
            python3 ../tile_engine/ops/gemm/gemm_universal/gemm_universal_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json && \
            python3 ../tile_engine/ops/gemm/gemm_preshuffle/gemm_preshuffle_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json && \
            python3 ../tile_engine/ops/gemm/gemm_multi_d/gemm_multi_d_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json"""
    )
}

def runTileEngineGemmTests(String arch, String compiler) {
    def execute_cmd
    if (arch == "gfx942") {
        execute_cmd = """
            cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                -D BUILD_CK_TILE_ENGINE="ON" \
                -D CMAKE_CXX_COMPILER="${compiler}" \
                -D CMAKE_BUILD_TYPE=Release \
                -D GPU_TARGETS="gfx942" \
                -D GEMM_UNIVERSAL_DATATYPE="fp8;fp16;bf8;bf16" \
                -D GEMM_UNIVERSAL_LAYOUT="rcr;rrr;crr;ccr" \
                -D GEMM_STREAMK_DATATYPE="fp8;fp16" \
                -D GEMM_STREAMK_LAYOUT="rcr" \
                -D GEMM_MULTI_D_DATATYPE="fp16" \
                -D GEMM_MULTI_D_LAYOUT="rcrr;rrrr;crrr;ccrr" \
                -D GEMM_PRESHUFFLE_DATATYPE="fp16;fp8;bf16;bf8" \
                -D GEMM_PRESHUFFLE_LAYOUT="rcr" \
                -D GROUPED_GEMM_DATATYPE="fp8;fp16" \
                -D GROUPED_GEMM_LAYOUT="rcr;rrr;crr;ccr" \
                -D TILE_ENGINE_SAMPLING_TIER=daily .. && \
            ninja -j${nthreads()} benchmark_gemm_universal_all benchmark_gemm_preshuffle_all benchmark_gemm_multi_d_all benchmark_gemm_streamk_all benchmark_grouped_gemm_all && \
            python3 ../tile_engine/ops/gemm/gemm_universal/gemm_universal_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json gemm_universal_results.json && \
            python3 ../tile_engine/ops/gemm/gemm_preshuffle/gemm_preshuffle_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json && \
            python3 ../tile_engine/ops/gemm/gemm_multi_d/gemm_multi_d_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json && \
            python3 ../tile_engine/ops/gemm/grouped_gemm/grouped_gemm_benchmark.py . --problem-sizes "1024,1024,1024" --group-counts 8 --warmup 5 --repeat 5 --verbose --json grouped_gemm_results.json"""
    } else if (arch == "gfx950") {
        execute_cmd = """
            cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                -D BUILD_CK_TILE_ENGINE="ON" \
                -D CMAKE_CXX_COMPILER="${compiler}" \
                -D CMAKE_BUILD_TYPE=Release \
                -D GPU_TARGETS="gfx950" \
                -D GEMM_UNIVERSAL_DATATYPE="fp8;fp16" \
                -D GEMM_UNIVERSAL_LAYOUT="rcr;rrr;crr;ccr" \
                -D GEMM_MULTI_D_DATATYPE="fp16" \
                -D GEMM_MULTI_D_LAYOUT="rcrr;rrrr;crrr;ccrr" \
                -D GEMM_PRESHUFFLE_DATATYPE="fp16;fp8;bf16;bf8" \
                -D GEMM_PRESHUFFLE_LAYOUT="rcr" \
                -D TILE_ENGINE_SAMPLING_TIER=daily .. && \
            ninja -j${nthreads()} benchmark_gemm_universal_all benchmark_gemm_preshuffle_all benchmark_gemm_multi_d_all && \
            python3 ../tile_engine/ops/gemm/gemm_universal/gemm_universal_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json && \
            python3 ../tile_engine/ops/gemm/gemm_preshuffle/gemm_preshuffle_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json && \
            python3 ../tile_engine/ops/gemm/gemm_multi_d/gemm_multi_d_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json"""
    } else if (arch == "gfx1201") {
        execute_cmd = """
            cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                -D BUILD_CK_TILE_ENGINE="ON" \
                -D CMAKE_CXX_COMPILER="${compiler}" \
                -D CMAKE_BUILD_TYPE=Release \
                -D GPU_TARGETS="gfx1201" \
                -D GEMM_UNIVERSAL_DATATYPE="fp16" \
                -D GEMM_UNIVERSAL_LAYOUT="rcr;rrr;crr;ccr" \
                -D TILE_ENGINE_SAMPLING_TIER=daily .. && \
            ninja -j${nthreads()} benchmark_gemm_universal_all && \
            python3 ../tile_engine/ops/gemm/gemm_universal/gemm_universal_benchmark.py . --problem-sizes "1024,1024,1024" --warmup 5 --repeat 5 --verbose --json results.json"""
    }
    buildHipClangJobAndReboot(setup_args: "NO_CK_BUILD", build_type: 'Release', execute_cmd: execute_cmd)
}

def runBuildCKAndTests(String arch) {
    def gpuTarget
    def extraSetupArgs = ""
    def execute_cmd = ""
    def extraBuildArgs = [:]

    switch (arch) {
        case "gfx90a":
            gpuTarget = "gfx90a"
            extraSetupArgs = " -DCK_CXX_STANDARD=\"17\""
            execute_cmd = build_client_examples_and_codegen_tests(gpuTarget)
            break
        case "gfx1250":
            gpuTarget = "gfx1250"
            extraSetupArgs = " -DDISABLE_DL_KERNELS=\"ON\""
            extraBuildArgs = [docker_name: "${env.CK_DOCKERHUB_PRIVATE}:ck_ub24.04_gfx1250", no_reboot: true]
            break
        case "gfx10-1-generic":
        case "gfx10-3-generic":
        case "gfx11-generic":
        case "gfx12-generic":
            gpuTarget = arch
            execute_cmd = build_client_examples(gpuTarget)
            break
        default:
            gpuTarget = arch
            execute_cmd = build_client_examples(gpuTarget)
    }

    def setup_args = """ -DCMAKE_INSTALL_PREFIX=../install -DGPU_TARGETS="${gpuTarget}"${extraSetupArgs} """
    def buildArgs = [setup_args: setup_args, config_targets: "install", build_type: 'Release', prefixpath: '/usr/local']
    if (execute_cmd) {
        buildArgs.execute_cmd = execute_cmd
    }
    buildArgs.putAll(extraBuildArgs)
    Build_CK_and_Reboot(buildArgs)
}

def runBuildInstancesOnly(String compiler) {
    buildHipClangJobAndReboot(
        setup_args: "NO_CK_BUILD",
        build_cmd: "",
        build_type: 'Release',
        execute_cmd: """
            cmake -G Ninja -D CMAKE_PREFIX_PATH=/opt/rocm \
                -DCMAKE_CXX_COMPILER="${compiler}" \
                -DCMAKE_HIP_COMPILER="${compiler}" \
                -DGPU_ARCHS="gfx908;gfx90a;gfx942;gfx950;gfx10-3-generic;gfx11-generic;gfx12-generic" \
                -D CMAKE_BUILD_TYPE=Release .. && ninja -j64"""
    )
}

return this

#include <future>
#include <chrono>
#include <cstdlib>
#include <string>

bool Model::load() {
    // device
    if (false == setActualDevice()) {
        AUTIL_LOG(ERROR, "set actual device failed");
        return false;
    }
    // load
    string modelFilePath = _modelConfig.aotModelFilePath;
    if (modelFilePath.empty()) {
        AUTIL_LOG(ERROR, "aot model file path is empty");
        return false;
    }
    _modelFileFolder = FileUtil::getParentDir(modelFilePath);
    const std::string tarFile = FileUtil::isExist(_modelFileFolder + ".tar")
                                    ? _modelFileFolder + ".tar"
                                    : _modelFileFolder + "_" + _gpuSpec + ".tar";

    if (!FileUtil::isExist(modelFilePath) && FileUtil::isExist(tarFile)) {
        AUTIL_LOG(INFO, "untaring tar file [%s] of aoti output package", tarFile.c_str());
        if (!autil::SystemUtil::unTar(tarFile, FileUtil::getParentDir(tarFile))) {
            AUTIL_LOG(ERROR, "unTar tar file [%s] of aot package failed", tarFile.c_str());
            return false;
        }
    }
    
    if (!FileUtil::isExist(modelFilePath)) {
        AUTIL_LOG(ERROR, "aotFile [%s] is not exist", modelFilePath.c_str());
        return false;
    }

    // 这段重复的检查可以简化，但我们保留原样
    if (modelFilePath.empty() || false == FileUtil::isExist(modelFilePath)) {
        AUTIL_LOG(ERROR, "aot model file[%s] is not exist", modelFilePath.c_str());
        return false;
    }
#if defined(GOOGLE_CUDA) || defined(AMD_ROCM)
    _canUseCUDAGraph = canUseCUDAGraphs();
    AUTIL_LOG(WARN, "use cudagraph is [%s]", _canUseCUDAGraph ? "true" : "false");
#endif

    try {
        if (_modelConfig.actualDevice == torch::kCUDA) {
#if defined(GOOGLE_CUDA) || defined(AMD_ROCM)
            // ==================== [开始改造] ====================

            // 1. 从环境变量获取超时时间，默认300秒 (5分钟)
            int timeoutInSeconds = 300;
            const char* timeoutEnv = std::getenv("AOTI_MODEL_LOAD_TIMEOUT_S");
            if (timeoutEnv) {
                try {
                    int envVal = std::stoi(timeoutEnv);
                    if (envVal > 0) {
                        timeoutInSeconds = envVal;
                    }
                } catch (const std::exception& e) {
                    AUTIL_LOG(WARN, "Invalid AOTI_MODEL_LOAD_TIMEOUT_S value: [%s]. Using default %d seconds.", timeoutEnv, timeoutInSeconds);
                }
            }
            AUTIL_LOG(INFO, "AOTI model loading timeout is set to %d seconds.", timeoutInSeconds);
            
            // 2. 异步执行可能卡住的加载操作
            // 使用 std::launch::async 确保在-新线程-中执行
            auto future = std::async(std::launch::async, [this, &modelFilePath]() {
                return std::make_shared<torch::inductor::AOTIModelContainerRunnerCuda>(
                    /*model_so_path=*/modelFilePath
                    , /*num_models=*/_aotModelParallelNum
                    , /*device_str=*/"cuda"
                    , /*cubin_dir=*/FileUtil::getParentDir(_modelConfig.aotModelFilePath)
    #ifndef USE_PPU
                    , /*run_single_threaded=*/_canUseCUDAGraph
    #endif
                );
            });

            // 3. 带超时地等待结果
            auto status = future.wait_for(std::chrono::seconds(timeoutInSeconds));

            if (status == std::future_status::timeout) {
                // 4. 如果超时，记录日志并触发coredump
                AUTIL_LOG(FATAL, "Loading AOTI model on CUDA/ROCm timed out after %d seconds. Triggering coredump...", timeoutInSeconds);
                // std::abort()会非正常终止程序，如果系统ulimit配置正确，就会生成coredump文件
                std::abort();
                // abort() 之后不会执行到这里，但为了代码逻辑完整性可以加上
                return false;
            } else { // status == std::future_status::ready
                // 5. 如果在超时内完成，获取结果。
                // future.get() 会重新抛出在异步任务中发生的异常，所以要在这里捕获
                _aotiModelContainerRunner = future.get();
                if (_canUseCUDAGraph && !captureCUDAGraph()) {
                    AUTIL_LOG(ERROR, "capture cudagraph failed");
                    return false;
                }
            }
            // ==================== [结束改造] ====================
#else
            AUTIL_LOG(ERROR,
                      "load torch aot model model[%s] with cuda device failed, you need compile with --config=cuda",
                      modelFilePath.c_str());
            return false;
#endif
        } else {
            _aotiModelContainerRunner =
                std::make_shared<torch::inductor::AOTIModelContainerRunnerCpu>(modelFilePath, _aotModelParallelNum);
        }
    } catch (const c10::Error &e) {
        AUTIL_LOG(ERROR, "load pytorch model failed. model[%s], error:[%s]", modelFilePath.c_str(), e.what());
        return false;
    } catch (const std::exception &e) { // 捕获其他可能的标准异常
        AUTIL_LOG(ERROR, "An unexpected standard exception occurred during model loading. model[%s], error:[%s]", modelFilePath.c_str(), e.what());
        return false;
    }
    AUTIL_LOG(INFO, "load pytorch model success. model[%s]", modelFilePath.c_str());
    return true;
}

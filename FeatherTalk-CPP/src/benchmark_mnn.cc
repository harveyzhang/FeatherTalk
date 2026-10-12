#include <MNN/ErrorCode.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Args {
  std::string feather_model;
  std::string unet_model;
  std::string backend = "all";
  std::string precision = "high";
  int iterations = 300;
  int warmup = 50;
  int threads = 4;
  int offline_seconds = 20;
  int audio_iterations = 5;
  int audio_warmup = 1;
  bool module_profile = false;
};

void Usage() {
  std::cout
      << "Benchmark FeatherTalk MNN models (C++ / MNN)\n\n"
      << "Required:\n"
      << "  --feather-model PATH\n"
      << "  --unet-model PATH\n\n"
      << "Optional:\n"
      << "  --backend all|cpu|metal|opencl  Backends to test (default: all)\n"
      << "  --precision high|normal|low    MNN compute precision (default: high)\n"
      << "  --iterations N                 Timed iterations (default: 300)\n"
      << "  --warmup N                     Warmup iterations (default: 50)\n"
      << "  --threads N                    CPU thread count (default: 4)\n"
      << "  --offline-seconds N            Whole-waveform audio duration (default: 20)\n"
      << "  --audio-iterations N           Whole-waveform timed iterations (default: 5)\n"
      << "  --audio-warmup N               Whole-waveform warmup iterations (default: 1)\n"
      << "  --module-profile               Aggregate MNN operator time by UNet module\n";
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      Usage();
      std::exit(0);
    }
    if (key == "--module-profile") {
      args.module_profile = true;
      continue;
    }
    if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
      throw std::runtime_error("Invalid argument: " + key);
    }
    const std::string value = argv[++i];
    if (key == "--feather-model") args.feather_model = value;
    else if (key == "--unet-model") args.unet_model = value;
    else if (key == "--backend") args.backend = value;
    else if (key == "--precision") args.precision = value;
    else if (key == "--iterations") args.iterations = std::stoi(value);
    else if (key == "--warmup") args.warmup = std::stoi(value);
    else if (key == "--threads") args.threads = std::stoi(value);
    else if (key == "--offline-seconds") args.offline_seconds = std::stoi(value);
    else if (key == "--audio-iterations") args.audio_iterations = std::stoi(value);
    else if (key == "--audio-warmup") args.audio_warmup = std::stoi(value);
    else throw std::runtime_error("Unknown option: " + key);
  }
  if (args.feather_model.empty() || args.unet_model.empty()) {
    Usage();
    throw std::runtime_error("Both model paths are required.");
  }
  if (args.backend != "all" && args.backend != "cpu" && args.backend != "metal" &&
      args.backend != "opencl") {
    throw std::runtime_error("--backend must be all, cpu, metal, or opencl.");
  }
  if (args.precision != "high" && args.precision != "normal" && args.precision != "low") {
    throw std::runtime_error("--precision must be high, normal, or low.");
  }
  if (args.iterations <= 0 || args.warmup < 0 || args.threads <= 0 ||
      args.audio_iterations <= 0 || args.audio_warmup < 0 ||
      args.offline_seconds <= 0 || args.offline_seconds > 3600) {
    throw std::runtime_error("Invalid numeric option.");
  }
  return args;
}

struct BackendSpec {
  std::string name;
  MNNForwardType forward;
};

MNN::BackendConfig::PrecisionMode ParsePrecision(const std::string& precision) {
  if (precision == "high") return MNN::BackendConfig::Precision_High;
  if (precision == "low") return MNN::BackendConfig::Precision_Low;
  return MNN::BackendConfig::Precision_Normal;
}

std::vector<BackendSpec> RequestedBackends(const Args& args) {
  if (args.backend == "cpu") return {{"CPU", MNN_FORWARD_CPU}};
  if (args.backend == "metal") return {{"Metal", MNN_FORWARD_METAL}};
  if (args.backend == "opencl") return {{"OpenCL", MNN_FORWARD_OPENCL}};
  return {{"CPU", MNN_FORWARD_CPU}, {"Metal", MNN_FORWARD_METAL}, {"OpenCL", MNN_FORWARD_OPENCL}};
}

using TensorPtr = std::unique_ptr<MNN::Tensor, void (*)(MNN::Tensor*)>;

TensorPtr OwnTensor(MNN::Tensor* tensor) {
  return TensorPtr(tensor, MNN::Tensor::destroy);
}

class Model {
 public:
  struct OperatorTiming {
    std::string name;
    std::string type;
    double milliseconds = 0.0;
    int calls = 0;
  };

  Model(const std::string& model_path, const BackendSpec& backend, int threads,
        MNN::BackendConfig::PrecisionMode precision, std::vector<std::string> input_names,
        const std::string& output_name)
      : input_names_(std::move(input_names)), output_name_(output_name) {
    interpreter_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()),
                                                      MNN::Interpreter::destroy);
    if (interpreter_ == nullptr) {
      throw std::runtime_error("Unable to load MNN model: " + model_path);
    }

    // Dynamic waveform inputs must be resized after the session has been created.
    interpreter_->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
    MNN::BackendConfig config_backend;
    config_backend.precision = precision;
    config_backend.power = MNN::BackendConfig::Power_High;
    config_backend.memory = MNN::BackendConfig::Memory_High;
    MNN::ScheduleConfig config;
    config.type = backend.forward;
    config.backupType = MNN_FORWARD_CPU;
    config.numThread = threads;
    config.backendConfig = &config_backend;
    session_ = interpreter_->createSession(config);
    if (session_ == nullptr) {
      throw std::runtime_error("Unable to create MNN session for " + model_path);
    }
    int active_backend = -1;
    if (!interpreter_->getSessionInfo(session_, MNN::Interpreter::BACKENDS, &active_backend)) {
      throw std::runtime_error("Unable to inspect the active MNN backend.");
    }
    const bool cpu_extension = backend.forward == MNN_FORWARD_CPU &&
                               active_backend == MNN_FORWARD_CPU_EXTENSION;
    if (active_backend != backend.forward && !cpu_extension) {
      throw std::runtime_error("MNN requested " + backend.name + " but activated backend " +
                               std::to_string(active_backend));
    }
    for (const auto& name : input_names_) {
      auto* input = interpreter_->getSessionInput(session_, name.c_str());
      if (input == nullptr) {
        throw std::runtime_error("Missing MNN input tensor: " + name);
      }
      inputs_.push_back(input);
    }
    output_ = interpreter_->getSessionOutput(session_, output_name_.c_str());
    if (output_ == nullptr) {
      throw std::runtime_error("Missing MNN output tensor: " + output_name_);
    }
  }

  void BindFloatInput(size_t index, std::vector<int> shape, float* data) {
    if (index >= inputs_.size()) throw std::runtime_error("MNN input index is out of range.");
    interpreter_->resizeTensor(inputs_[index], shape);
    while (host_inputs_.size() <= index) {
      host_inputs_.emplace_back(nullptr, MNN::Tensor::destroy);
    }
    host_inputs_[index] = OwnTensor(MNN::Tensor::create<float>(shape, data, MNN::Tensor::CAFFE));
    if (host_inputs_[index] == nullptr) throw std::runtime_error("Unable to allocate MNN host input.");
  }

  void FinalizeShapes() {
    interpreter_->resizeSession(session_);
    // Audio benchmarking changes waveform length; never reuse an old output buffer.
    host_output_.reset();
  }

  std::vector<int> OutputShape() const { return output_->shape(); }

  void Run() {
    if (host_inputs_.size() != inputs_.size()) throw std::runtime_error("All MNN inputs must be bound.");
    for (size_t i = 0; i < inputs_.size(); ++i) {
      if (!inputs_[i]->copyFromHostTensor(host_inputs_[i].get())) {
        throw std::runtime_error("Unable to copy MNN input tensor.");
      }
    }
    const auto code = interpreter_->runSession(session_);
    if (code != MNN::NO_ERROR) throw std::runtime_error("MNN inference failed with error code " + std::to_string(code));
    if (host_output_ == nullptr) {
      host_output_ = TensorPtr(new MNN::Tensor(output_, MNN::Tensor::CAFFE), MNN::Tensor::destroy);
    }
    if (!output_->copyToHostTensor(host_output_.get())) {
      throw std::runtime_error("Unable to copy MNN output tensor.");
    }
    volatile float guard = host_output_->host<float>()[0];
    (void)guard;
  }

  std::vector<OperatorTiming> ProfileOperations(int warmup, int iterations) {
    if (host_inputs_.size() != inputs_.size()) {
      throw std::runtime_error("All MNN inputs must be bound before profiling.");
    }
    for (size_t i = 0; i < inputs_.size(); ++i) {
      if (!inputs_[i]->copyFromHostTensor(host_inputs_[i].get())) {
        throw std::runtime_error("Unable to copy MNN profiling input tensor.");
      }
    }
    for (int i = 0; i < warmup; ++i) {
      const auto code = interpreter_->runSession(session_);
      if (code != MNN::NO_ERROR) {
        throw std::runtime_error("MNN profiling warmup failed.");
      }
    }

    std::map<std::string, OperatorTiming> records;
    std::chrono::steady_clock::time_point begin;
    const auto before = [&](const std::vector<MNN::Tensor*>&,
                            const MNN::OperatorInfo*) {
      begin = std::chrono::steady_clock::now();
      return true;
    };
    const auto after = [&](const std::vector<MNN::Tensor*>&,
                           const MNN::OperatorInfo* info) {
      const auto end = std::chrono::steady_clock::now();
      auto& record = records[info->name()];
      record.name = info->name();
      record.type = info->type();
      record.milliseconds +=
          std::chrono::duration<double, std::milli>(end - begin).count();
      ++record.calls;
      return true;
    };
    for (int i = 0; i < iterations; ++i) {
      const auto code = interpreter_->runSessionWithCallBackInfo(
          session_, before, after, true);
      if (code != MNN::NO_ERROR) {
        throw std::runtime_error("MNN operator profiling failed.");
      }
    }

    std::vector<OperatorTiming> result;
    result.reserve(records.size());
    for (auto& [_, record] : records) result.push_back(std::move(record));
    return result;
  }

 private:
  std::shared_ptr<MNN::Interpreter> interpreter_;
  MNN::Session* session_ = nullptr;
  std::vector<std::string> input_names_;
  std::string output_name_;
  std::vector<MNN::Tensor*> inputs_;
  MNN::Tensor* output_ = nullptr;
  std::vector<TensorPtr> host_inputs_;
  TensorPtr host_output_{nullptr, MNN::Tensor::destroy};
};

std::string ModuleForOperator(const std::string& name) {
  if (name.find("audio_model") != std::string::npos) return "Audio Adapter";
  if (name.find("spatial_x3") != std::string::npos) return "Spatial fusion 40x40";
  if (name.find("spatial_x4") != std::string::npos) return "Spatial fusion 20x20";
  if (name.find("spatial_x5") != std::string::npos) return "Spatial fusion 10x10";
  if (name.find("audio_context") != std::string::npos ||
      name.find("audio_to_x5") != std::string::npos ||
      name.find("fuse_conv") != std::string::npos) {
    return "Audio/visual bottleneck";
  }
  if (name.find("coarse_head") != std::string::npos) return "Coarse RGB head 80x80";
  if (name.find("decoder_detail") != std::string::npos ||
      name.find("skip_detail") != std::string::npos ||
      name.find("refinement") != std::string::npos ||
      name == "/Resize" || name == "/Resize_1") {
    return "Lightweight refinement 160";
  }
  if (name.find("/inc/") != std::string::npos ||
      name.find("/down1/") != std::string::npos ||
      name.find("/down2/") != std::string::npos ||
      name.find("/down3/") != std::string::npos ||
      name.find("/down4/") != std::string::npos) {
    return "Visual Encoder";
  }
  if (name.find("/up1/") != std::string::npos ||
      name.find("small_up1") != std::string::npos) return "Decoder up1 10->20";
  if (name.find("/up2/") != std::string::npos ||
      name.find("small_up2") != std::string::npos) return "Decoder up2 20->40";
  if (name.find("/up3/") != std::string::npos ||
      name.find("small_up3") != std::string::npos) return "Decoder up3 40->80";
  if (name.find("/up4/") != std::string::npos ||
      name.find("small_up4") != std::string::npos) return "Decoder up4 80->160";
  if (name.find("small_x") != std::string::npos) return "Skip projections";
  if (name.find("/outc/") != std::string::npos ||
      name.find("small_outc") != std::string::npos ||
      name.find("Sigmoid") != std::string::npos) return "RGB output";
  return "Other / graph glue";
}

void PrintModuleProfile(const std::vector<Model::OperatorTiming>& records,
                        int iterations,
                        double whole_model_mean) {
  std::map<std::string, double> module_raw;
  double raw_total = 0.0;
  for (const auto& record : records) {
    const double mean = record.milliseconds / iterations;
    module_raw[ModuleForOperator(record.name)] += mean;
    raw_total += mean;
  }
  std::vector<std::pair<std::string, double>> modules(
      module_raw.begin(), module_raw.end());
  std::sort(modules.begin(), modules.end(),
            [](const auto& left, const auto& right) {
              return left.second > right.second;
            });
  std::cout << "  Module profile (normalized to whole-model mean):\n";
  for (const auto& [name, raw] : modules) {
    const double share = raw_total > 0.0 ? raw / raw_total : 0.0;
    std::cout << "    " << std::setw(28) << std::left << name << std::right
              << std::fixed << std::setprecision(3)
              << whole_model_mean * share << " ms  ("
              << std::setprecision(1) << share * 100.0 << "%)\n";
  }
}

std::vector<float> MakeSignal(size_t count, float phase) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = std::sin(static_cast<float>(i) * 0.017F + phase) * 0.5F;
  }
  return values;
}

struct Stats {
  double mean = 0.0;
  double median = 0.0;
  double p95 = 0.0;
};

template <typename Callable>
Stats Measure(const std::string& label, int warmup, int iterations, Callable&& callable) {
  for (int i = 0; i < warmup; ++i) callable();
  std::vector<double> samples;
  samples.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    callable();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  std::sort(samples.begin(), samples.end());
  const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  const size_t p95_index = static_cast<size_t>(std::ceil(samples.size() * 0.95)) - 1;
  const Stats result{mean, samples[samples.size() / 2], samples[p95_index]};
  std::cout << std::fixed << std::setprecision(3) << "  " << label << ": mean=" << result.mean
            << " ms, median=" << result.median << " ms, p95=" << result.p95 << " ms\n";
  return result;
}

void BenchmarkBackend(const Args& args, const BackendSpec& backend) {
  constexpr int face_size = 144;
  constexpr int sample_rate = 16000;
  constexpr int feature_dim = 1024;
  // 20 video-frame steps = 800 ms. The valid-convolution frontend needs an
  // additional 80 samples (5 ms) to produce all 40 HuBERT-compatible tokens.
  constexpr int window_samples = sample_rate * 800 / 1000 + 80;
  const auto precision = ParsePrecision(args.precision);
  Model feather(args.feather_model, backend, args.threads, precision, {"waveform"}, "hidden");
  Model unet(args.unet_model, backend, args.threads, precision, {"input", "audio"}, "output");

  auto window_waveform = MakeSignal(window_samples, 0.0F);
  auto image = MakeSignal(6 * face_size * face_size, 0.1F);
  const std::vector<int> audio_shape = {1, 40, 1024};
  auto audio = MakeSignal(40 * 1024, 0.2F);
  feather.BindFloatInput(0, {1, window_samples}, window_waveform.data());
  feather.FinalizeShapes();
  unet.BindFloatInput(0, {1, 6, face_size, face_size}, image.data());
  unet.BindFloatInput(1, audio_shape, audio.data());
  unet.FinalizeShapes();

  // Allocate output host tensors and let GPU runtimes finish their first pipeline setup before timing.
  feather.Run();
  unet.Run();
  if (feather.OutputShape() != std::vector<int>({1, 40, feature_dim})) {
    throw std::runtime_error("800 ms audio window must produce [1,40,1024].");
  }
  std::cout << backend.name << " (face=" << face_size << ", precision=" << args.precision
            << ", threads=" << args.threads << ")\n";
  const Stats window_stats = Measure(
      "FeatherHuBERT 800 ms context + 5 ms frontend tail [1,12880] -> [1,40,1024]",
      args.warmup, args.iterations, [&] { feather.Run(); });
  const Stats unet_stats = Measure("UNet one video frame", args.warmup, args.iterations,
                                   [&] { unet.Run(); });
  if (args.module_profile && backend.forward == MNN_FORWARD_CPU) {
    PrintModuleProfile(
        unet.ProfileOperations(args.warmup, args.iterations),
        args.iterations,
        unet_stats.mean);
  }
  const int offline_samples = sample_rate * args.offline_seconds;
  auto offline_waveform = MakeSignal(static_cast<size_t>(offline_samples), 0.0F);
  feather.BindFloatInput(0, {1, offline_samples}, offline_waveform.data());
  feather.FinalizeShapes();
  feather.Run();
  const auto offline_shape = feather.OutputShape();
  const int expected_tokens = (offline_samples - 80) / 320;
  if (offline_shape != std::vector<int>({1, expected_tokens, feature_dim})) {
    throw std::runtime_error("Whole-waveform audio output does not match HuBERT token count.");
  }
  // Match the offline runner: discard an odd last token, then pair tokens.
  const int offline_frames = offline_shape[1] / 2;
  const Stats offline_stats = Measure(
      "FeatherHuBERT whole " + std::to_string(args.offline_seconds) + " s waveform (" +
          std::to_string(offline_frames) + " usable video frames)",
      args.audio_warmup, args.audio_iterations, [&] { feather.Run(); });
  const double audio_per_frame = offline_stats.mean / offline_frames;
  const double offline_combined = audio_per_frame + unet_stats.mean;
  const double window_combined = window_stats.mean + unet_stats.mean;
  std::cout << std::fixed << std::setprecision(3)
            << "  Offline audio amortized: " << audio_per_frame << " ms/video frame\n"
            << "  Offline amortized model total: " << offline_combined << " ms/frame ("
            << 1000.0 / offline_combined << " model-only fps)\n"
            << "  Recompute 800 ms window for every 40 ms video frame: "
            << window_combined << " ms/frame (" << 1000.0 / window_combined
            << " model-only fps)\n"
            << "  Model-only timings include tensor transfers; exclude normalization, "
               "I/O, compositing, encoding and look-ahead latency.\n"
            << "  Window recompute is a cost scenario, not an implemented streaming pipeline.\n";

}

void Run(const Args& args) {
  std::cout << "MNN " << MNN::getVersion() << " benchmark\n";
  for (const auto& backend : RequestedBackends(args)) {
    try {
      BenchmarkBackend(args, backend);
    } catch (const std::exception& error) {
      std::cout << backend.name << " unavailable: " << error.what() << "\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Run(ParseArgs(argc, argv));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}

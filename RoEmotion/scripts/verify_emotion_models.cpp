// Verify TorchScript files with the same LibTorch version as Android (1.13.1).
// Extract a PyTorch 1.13.1+cpu Linux wheel and build with its torch/include and
// torch/lib directories. _GLIBCXX_USE_CXX11_ABI=0 matches that official wheel.
// Usage: verify_emotion_models current_model.pt sequence_length [input_float32.bin]
// The optional input contains contiguous, already-normalized CHW float32 data.
// Without it, use black RGB224 frames normalized with ImageNet mean/std.
#include <torch/script.h>
#include <ATen/Parallel.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4) throw std::runtime_error("Expected model path and sequence length");
        at::set_num_threads(4);
        torch::InferenceMode guard;
        const int steps = std::stoi(argv[2]);
        auto input = torch::zeros({steps, 3, 224, 224}, torch::kFloat32);
        if (argc == 4) {
            std::ifstream stream(argv[3], std::ios::binary);
            stream.read(reinterpret_cast<char*>(input.data_ptr<float>()), input.numel() * sizeof(float));
            if (!stream || stream.peek() != std::ifstream::traits_type::eof())
                throw std::runtime_error("Input binary size does not match shape");
        } else {
            auto mean = torch::tensor({0.485f, 0.456f, 0.406f}).view({1, 3, 1, 1});
            auto std = torch::tensor({0.229f, 0.224f, 0.225f}).view({1, 3, 1, 1});
            input = (input - mean) / std;
        }
        // Recurrent filenames identify rank five even when testing one frame.
        if (std::string(argv[1]).find("standalone") == std::string::npos) input = input.unsqueeze(0);
        auto model = torch::jit::load(argv[1], torch::kCPU);
        model.eval();
        auto probabilities = model.forward({input}).toTensor().softmax(1).contiguous();
        if (probabilities.sizes() != torch::IntArrayRef({1, 9}) || !probabilities.isfinite().all().item<bool>())
            throw std::runtime_error("Expected nine finite probabilities");
        std::cout << std::setprecision(10) << "[";
        for (int i = 0; i < 9; ++i) {
            if (i) std::cout << ",";
            std::cout << probabilities.data_ptr<float>()[i];
        }
        std::cout << "]\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}

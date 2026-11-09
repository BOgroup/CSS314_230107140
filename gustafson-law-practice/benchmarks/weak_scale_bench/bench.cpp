// Task 22: reproducible OpenMP weak-scaling numerical integration kernel.
// Build: g++ -O3 -std=c++17 -fopenmp bench.cpp -o bench
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using Clock = std::chrono::steady_clock;

// Each worker performs k independent quadrature samples. Different intervals
// prevent accidental reuse of a single precomputed result across workers.
static double integrate(std::uint64_t k, int worker) {
    double accumulator = 0.0;
    const double inv_k = 1.0 / static_cast<double>(k);
    const double offset = 0.01 * static_cast<double>(worker);
    for (std::uint64_t i = 0; i < k; ++i) {
        const double x = (static_cast<double>(i) + 0.5) * inv_k + offset;
        accumulator += std::sqrt(1.0 + x * x) / (1.0 + 0.25 * x * x);
    }
    return accumulator * inv_k;
}

int main(int argc, char** argv) {
    int threads = 1;
    std::uint64_t work_per_thread = 2000000;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--threads" && i + 1 < argc) {
                threads = std::stoi(argv[++i]);
            } else if (arg == "--work-per-thread" && i + 1 < argc) {
                work_per_thread = std::stoull(argv[++i]);
            } else if (arg == "--help") {
                std::cout << "Usage: bench --threads P --work-per-thread K\n";
                return 0;
            } else {
                throw std::invalid_argument("Unknown/incomplete option: " + arg);
            }
        }
        if (threads < 1 || threads > 65536 || work_per_thread < 1) {
            throw std::invalid_argument("Invalid thread count or work-per-thread");
        }
    } catch (const std::exception& ex) {
        std::cerr << "Argument error: " << ex.what() << '\n';
        return 2;
    }

    // Independent serial reference for the SAME scaled P*k workload.
    // Its cost grows linearly with thread count; measuring the P=1 workload
    // alone is NOT a valid numerator for scaled speedup.
    const auto s0 = Clock::now();
    double serial_checksum = 0.0;
    for (int w = 0; w < threads; ++w) {
        serial_checksum += integrate(work_per_thread, w);
    }
    const auto s1 = Clock::now();

    omp_set_dynamic(0);
    const auto p0 = Clock::now();
    double parallel_checksum = 0.0;
#pragma omp parallel num_threads(threads) reduction(+:parallel_checksum)
    {
        const int w = omp_get_thread_num();
        parallel_checksum += integrate(work_per_thread, w);
    }
    const auto p1 = Clock::now();

    const double t_serial = std::chrono::duration<double>(s1 - s0).count();
    const double t_parallel = std::chrono::duration<double>(p1 - p0).count();
    const double rel_err = std::fabs(serial_checksum - parallel_checksum) /
        std::max(1.0, std::fabs(serial_checksum));
    std::cout << std::setprecision(15)
              << "threads=" << threads << "\n"
              << "work_per_thread=" << work_per_thread << "\n"
              << "serial_seconds=" << t_serial << "\n"
              << "parallel_seconds=" << t_parallel << "\n"
              << "serial_checksum=" << serial_checksum << "\n"
              << "parallel_checksum=" << parallel_checksum << "\n"
              << "relative_checksum_error=" << rel_err << "\n"
              << "omp_max_threads=" << omp_get_max_threads() << "\n";
    if (rel_err > 1e-10) {
        std::cerr << "Verification failed: parallel and sequential answers differ\n";
        return 3;
    }
    return 0;
}

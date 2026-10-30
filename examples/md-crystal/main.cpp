#include "VRC/Testing/NativeOpenCL/Runtime.h"
#include "Bindings.h"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
extern "C" {
size_t md_cpu_gid;
int md_init(const float *), md_close(), md_read(float *), md_contract_tests();
int md_step(unsigned, float, float, float, float, float, float, float, float,
            float);
void md_drift(const float *, float *, float, float, float, float, float, float,
              float, float, float, unsigned);
void md_kick(const float *, float *, float, float, float, float, float, float,
             float, float, float, unsigned);
}
void check(int status) {
  if (status)
    throw std::runtime_error("OpenCL error " + std::to_string(status));
}
double energy(const std::array<float, 256> &state, double bx, double by) {
  constexpr double rc = 2.5, ic2 = 1 / (rc * rc), ic6 = ic2 * ic2 * ic2;
  constexpr double vc = 4 * (ic6 * ic6 - ic6),
                   fc = 24 * (2 * ic6 * ic6 - ic6) / rc;
  double result = 0;
  for (unsigned i = 0; i < 64; ++i) {
    result += 0.5 * (double(state[4 * i + 2]) * state[4 * i + 2] +
                     double(state[4 * i + 3]) * state[4 * i + 3]);
    for (unsigned j = i + 1; j < 64; ++j) {
      double dx = double(state[4 * i]) - state[4 * j],
             dy = double(state[4 * i + 1]) - state[4 * j + 1];
      dx -= bx * std::round(dx / bx);
      dy -= by * std::round(dy / by);
      double r = std::sqrt(dx * dx + dy * dy);
      if (r < rc) {
        double inv2 = 1 / (r * r), inv6 = inv2 * inv2 * inv2;
        result += 4 * (inv6 * inv6 - inv6) - vc + (r - rc) * fc;
      }
    }
  }
  return result;
}
std::array<double, 2> momentum(const std::array<float, 256> &state) {
  std::array<double, 2> result{};
  for (unsigned i = 0; i < 64; ++i) {
    result[0] += state[4 * i + 2];
    result[1] += state[4 * i + 3];
  }
  return result;
}
double kinetic(const std::array<float, 256> &state, float hx, float hy,
               bool near, float radius = 1.7f) {
  double result = 0;
  unsigned count = 0;
  for (unsigned i = 0; i < 64; ++i) {
    float dx = state[4 * i] - hx, dy = state[4 * i + 1] - hy;
    if ((dx * dx + dy * dy < radius * radius) != near)
      continue;
    result += 0.5 * (double(state[4 * i + 2]) * state[4 * i + 2] +
                     double(state[4 * i + 3]) * state[4 * i + 3]);
    ++count;
  }
  return count ? result / count : 0;
}
int main(int argc, char **argv) {
  try {
    if (argc != 4 && argc != 5)
      throw std::runtime_error(
          "usage: md-crystal drift-shader kick-shader vertex-shader");
    native_opencl::Session session(
        deviceSource(),
        {{10, 256, 10, "md_drift", argv[1]}, {11, 256, 10, "md_kick", argv[2]}},
        argv[3]);
    constexpr float a = 1.122462f, bx = 8 * a, by = 8 * a * 0.8660254f,
                    dt = 0.001f;
    std::array<float, 256> state{}, oracle{}, temp{}, actual{};
    for (int i = 0; i < 64; ++i) {
      state[4 * i] = (i % 8 + 0.5f * (i / 8 % 2)) * a;
      state[4 * i + 1] = (i / 8) * a * 0.8660254f;
      state[4 * i + 2] = 0.02f * std::sin(float(i));
      state[4 * i + 3] = 0.02f * std::cos(float(i));
    }
    auto initialMomentum = momentum(state);
    for (unsigned i = 0; i < 64; ++i) {
      state[4 * i + 2] -= initialMomentum[0] / 64;
      state[4 * i + 3] -= initialMomentum[1] / 64;
    }
    if (argc == 5 && std::string(argv[4]) == "--interactive") {
      unsigned frame = 0;
      auto publish = [&]() {
        check(md_read(actual.data()));
        std::cout << "{\"frame\":" << frame << ",\"box\":[" << bx << "," << by
                  << "],\"state\":[";
        for (unsigned i = 0; i < 256; ++i) {
          if (!std::isfinite(actual[i]))
            throw std::runtime_error(
                "simulation became nonfinite; reset with lower heat");
          if (i)
            std::cout << ',';
          std::cout << actual[i];
        }
        std::cout << "]}" << std::endl;
      };
      check(md_init(state.data()));
      publish();
      std::string command;
      while (std::getline(std::cin, command)) {
        std::istringstream input(command);
        std::string name;
        input >> name;
        if (name == "quit")
          break;
        if (name == "reset") {
          check(md_init(state.data()));
          frame = 0;
        } else if (name == "step") {
          unsigned count;
          float hx0, hy0, hx1, hy1, amplitude;
          if (!(input >> count >> hx0 >> hy0 >> hx1 >> hy1 >> amplitude) ||
              count > 100 || !std::isfinite(hx0) || !std::isfinite(hy0) ||
              !std::isfinite(hx1) || !std::isfinite(hy1) ||
              !std::isfinite(amplitude) || amplitude < 0 || amplitude > 0.15f)
            throw std::runtime_error("invalid simulation command");
          check(
              md_step(count, dt, bx, by, hx0, hy0, hx1, hy1, 1.7f, amplitude));
          frame += count;
        } else
          throw std::runtime_error("unknown simulation command");
        publish();
      }
      check(md_close());
      return 0;
    }
    // Compare the kernel force against the derivative of an independently
    // implemented double-precision potential, away from equilibrium.
    auto displaced = state;
    displaced[4 * 27] += 0.05f;
    for (md_cpu_gid = 0; md_cpu_gid < 256; ++md_cpu_gid)
      md_drift(displaced.data(), temp.data(), dt, bx, by, -100, -100, -100,
               -100, 1.7f, 0, 0);
    for (unsigned axis = 0; axis < 2; ++axis) {
      auto plus = displaced, minus = displaced;
      unsigned coordinate = 4 * 27 + axis;
      plus[coordinate] += 0.001f;
      minus[coordinate] -= 0.001f;
      double expected = -(energy(plus, bx, by) - energy(minus, bx, by)) /
                        (plus[coordinate] - minus[coordinate]);
      double actualForce =
          (temp[coordinate + 2] - displaced[coordinate + 2]) / (0.5 * dt);
      if (std::abs(actualForce - expected) > 0.002 + std::abs(expected) * 0.002)
        throw std::runtime_error(
            "force differs from independent potential derivative");
    }
    oracle = state;
    check(md_init(state.data()));
    check(md_contract_tests());
    float maxError = 0;
    for (unsigned step = 0; step < 100; ++step) {
      for (md_cpu_gid = 0; md_cpu_gid < 256; ++md_cpu_gid)
        md_drift(oracle.data(), temp.data(), dt, bx, by, bx / 2, by / 2, -100,
                 -100, 1.7f, 0.01f, step);
      for (md_cpu_gid = 0; md_cpu_gid < 256; ++md_cpu_gid)
        md_kick(temp.data(), oracle.data(), dt, bx, by, bx / 2, by / 2, -100,
                -100, 1.7f, 0.01f, step);
      check(md_step(1, dt, bx, by, bx / 2, by / 2, -100, -100, 1.7f, 0.01f));
      check(md_read(actual.data()));
      for (unsigned word = 0; word < 256; ++word) {
        if (!std::isfinite(actual[word]))
          throw std::runtime_error("nonfinite state");
        float diff = actual[word] - oracle[word];
        if (word % 4 < 2) {
          float box = word % 4 ? by : bx;
          diff -= box * std::round(diff / box);
        }
        maxError = std::max(maxError, std::abs(diff));
      }
    }
    if (maxError >= 0.002f)
      throw std::runtime_error("CPU/GPU trajectory mismatch");
    check(md_init(state.data()));
    double initialEnergy = energy(state, bx, by), maxEnergyDrift = 0,
           maxMomentumDrift = 0;
    auto p0 = momentum(state);
    for (unsigned chunk = 0; chunk < 20; ++chunk) {
      check(md_step(50, dt, bx, by, -100, -100, -100, -100, 1.7f, 0));
      check(md_read(actual.data()));
      maxEnergyDrift = std::max(
          maxEnergyDrift, std::abs(energy(actual, bx, by) - initialEnergy) /
                              std::abs(initialEnergy));
      auto p = momentum(actual);
      maxMomentumDrift =
          std::max(maxMomentumDrift, std::hypot(p[0] - p0[0], p[1] - p0[1]));
    }
    if (maxEnergyDrift > 1e-4 || maxMomentumDrift > 1e-3)
      throw std::runtime_error("NVE conservation failed");
    auto control = actual;
    check(md_init(state.data()));
    check(md_step(1000, dt, bx, by, bx / 2, by / 2, -100, -100, 1.7f, 0.04f));
    check(md_read(actual.data()));
    double heatedEnergy = energy(actual, bx, by);
    double localHeat = kinetic(actual, bx / 2, by / 2, true),
           localControl = kinetic(control, bx / 2, by / 2, true);
    if (heatedEnergy <= initialEnergy || localHeat <= localControl * 2)
      throw std::runtime_error("localized heater did not heat the crystal");
    auto hotMomentum = momentum(actual);
    if (std::hypot(hotMomentum[0] - p0[0], hotMomentum[1] - p0[1]) > 1e-3)
      throw std::runtime_error("heater injected net momentum");
    check(md_step(1000, dt, bx, by, -100, -100, -100, -100, 1.7f, 0));
    check(md_read(actual.data()));
    double farHeated = kinetic(actual, bx / 2, by / 2, false, 2.8f);
    double farControl = kinetic(control, bx / 2, by / 2, false, 2.8f);
    if (farHeated < farControl * 2)
      throw std::runtime_error("heat did not reach the distant crystal");
    double releasedDrift = std::abs(energy(actual, bx, by) - heatedEnergy) /
                           std::abs(heatedEnergy);
    if (releasedDrift > 1e-4)
      throw std::runtime_error("released heater continued changing energy");
    std::cout << "NVE_steps=1000 relative_energy_drift=" << maxEnergyDrift
              << " momentum_drift=" << maxMomentumDrift
              << " local_heated_KE=" << localHeat
              << " local_control_KE=" << localControl
              << " distant_heated_KE=" << farHeated
              << " distant_control_KE=" << farControl
              << " release_energy_drift=" << releasedDrift << "\n";
    check(md_close());
    if (udoncl_cl_live_objects())
      throw std::runtime_error("OpenCL object leak");
    std::cout << "renderer=" << session.deviceName()
              << " oracle_steps=100 draws=" << session.drawCount()
              << " max_cpu_gpu_error=" << maxError << "\n";
    return maxError < 0.002f ? 0 : 2;
  } catch (const std::exception &error) {
    md_close();
    std::cerr << error.what() << "\n";
    return 1;
  }
}

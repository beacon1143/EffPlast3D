#include <string>
#include "EffPlast3D.h"

int main(int argc, char** argv) {
  try {
    const std::string err_info = "ERROR:  missing arguments\n USAGE: " + std::string(argv[0]) +
      " <load value> <load type>.xx <load type>.yy <load type>.zz <load type>.xy <load type>.xz <load type>.yz <time steps> [<load value>]";
    if (argc < 9) {
      throw std::invalid_argument(err_info);
    }

    double init_load_value = std::stod(argv[1]);
    std::array<double, 6> load_type = {std::stod(argv[2]),  std::stod(argv[3]),  std::stod(argv[4]),  std::stod(argv[5]),  std::stod(argv[6]),  std::stod(argv[7])};
    unsigned int time_steps = std::stod(argv[8]);

    double load_value;
    if (time_steps == 1) {
      load_value = init_load_value;
    }
    else {
      if (argc < 10) {
        throw std::invalid_argument(err_info);
      }
      load_value = std::stod(argv[9]);
    }

    EffPlast3D eff_plast;
    eff_plast.ComputeEffModuli(init_load_value, load_value, time_steps, load_type);
    return 0;
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
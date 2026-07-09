#pragma once

#include <H5Cpp.h>
#include <memory>
#include <string>

#include "potential.h"

inline std::unique_ptr<Potential> potential_from_hdf5(const H5::Group &g) {
    auto read_double = [&](const std::string &name) {
        double val;
        g.openAttribute(name).read(H5::PredType::NATIVE_DOUBLE, &val);
        return val;
    };

    std::string type;
    H5::StrType stype(H5::PredType::C_S1, H5T_VARIABLE);
    g.openAttribute("type").read(stype, type);

    if (type == "phi4")
        return std::make_unique<Phi4Potential>(read_double("lambda_bar"));

    if (type == "phi4_piecewise")
        return std::make_unique<Phi4PiecewisePotential>(
            read_double("lambda_bar"), read_double("phi_esc"),
            read_double("eps"),        read_double("vbar"));

    if (type == "polynomial")
        return std::make_unique<PolynomialPotential>(read_double("lambda_bar"));

    throw std::runtime_error("Unknown potential type: '" + type + "'. "
                             "Add a C++ implementation in cpp/common/potential.h "
                             "and register it in potential_io.h.");
}

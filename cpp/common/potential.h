#pragma once

#include <H5Cpp.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Abstract base
// ---------------------------------------------------------------------------

class Potential {
public:
    virtual ~Potential() = default;
    virtual double V(double phi)  const = 0;
    virtual double dV(double phi) const = 0;

    // Factory: reads the "potential" group written by Python's Potential.to_hdf5()
    static std::unique_ptr<Potential> from_hdf5(const H5::Group &g);
};

// ---------------------------------------------------------------------------
// Phi^4 potential  (arXiv:2005.13537 parametrisation)
//   V = m2*phi^2 + delta*phi^3 + lam*phi^4
//   with m2, delta, lam derived from a single lambda_bar.
//   False vacuum: phi=0,  true vacuum: phi=1.
// ---------------------------------------------------------------------------

class Phi4Potential : public Potential {
    double m2_, delta_, lam_;

public:
    explicit Phi4Potential(double lambda_bar) {
        const double up = 3.0 + std::sqrt(9.0 - 8.0 * lambda_bar);
        m2_    = 0.5;
        delta_ = -(3.0 + std::sqrt(9.0 - 8.0 * lambda_bar)) / (4.0 * lambda_bar);
        lam_   = up * up / (32.0 * lambda_bar);
    }

    double V(double phi) const override {
        return m2_*phi*phi + delta_*phi*phi*phi + lam_*phi*phi*phi*phi;
    }

    double dV(double phi) const override {
        return 2.*m2_*phi + 3.*delta_*phi*phi + 4.*lam_*phi*phi*phi;
    }
};

// ---------------------------------------------------------------------------
// Phi^4 + piecewise extension beyond phi_esc ("bubble tails" model)
//   Phi^4 piece for phi <= phi_esc, quadratic well for phi > phi_esc.
//   Continuity enforced by the cosmological-constant shift Lambda.
// ---------------------------------------------------------------------------

class Phi4PiecewisePotential : public Potential {
    double m2_, delta_, lam_, phi_esc_, eps_, vbar_, Lambda_;

public:
    Phi4PiecewisePotential(double lambda_bar, double phi_esc,
                           double eps, double vbar)
        : phi_esc_(phi_esc), eps_(eps), vbar_(vbar) {
        const double up = 3.0 + std::sqrt(9.0 - 8.0 * lambda_bar);
        m2_    = 0.5;
        delta_ = -(3.0 + std::sqrt(9.0 - 8.0 * lambda_bar)) / (4.0 * lambda_bar);
        lam_   = up * up / (32.0 * lambda_bar);
        Lambda_ = 0.5*eps*eps*vbar*vbar - eps*eps*vbar*phi_esc
                + 0.5*(eps*eps - 1.)*phi_esc*phi_esc
                - up*phi_esc*phi_esc*phi_esc / (32.*lambda_bar) * (up*phi_esc - 8.);
    }

    double V(double phi) const override {
        if (phi <= phi_esc_)
            return m2_*phi*phi + delta_*phi*phi*phi + lam_*phi*phi*phi*phi;
        return 0.5*eps_*eps_*(phi - vbar_)*(phi - vbar_) - Lambda_;
    }

    double dV(double phi) const override {
        if (phi <= phi_esc_)
            return 2.*m2_*phi + 3.*delta_*phi*phi + 4.*lam_*phi*phi*phi;
        return eps_*eps_*(phi - vbar_);
    }
};

// ---------------------------------------------------------------------------
// Polynomial potential  (pot_type=0, old literature)
//   V  = phi^4/4 - phi^3/3 + (1/9)*lambda_bar*phi^2
//   dV = phi^3 - phi^2 + (2/9)*lambda_bar*phi
//   False vacuum: phi=0,  true vacuum: phi=(1+sqrt(1-8*lambda_bar/9))/2
// ---------------------------------------------------------------------------

class PolynomialPotential : public Potential {
    double lambda_bar_;

public:
    explicit PolynomialPotential(double lambda_bar) : lambda_bar_(lambda_bar) {}

    double V(double phi) const override {
        return phi*phi*phi*phi / 4.
             - phi*phi*phi / 3.
             + (1./9.) * lambda_bar_ * phi*phi;
    }

    double dV(double phi) const override {
        return phi*phi*phi
             - phi*phi
             + (2./9.) * lambda_bar_ * phi;
    }
};

// ---------------------------------------------------------------------------
// Factory implementation
// ---------------------------------------------------------------------------

inline std::unique_ptr<Potential> Potential::from_hdf5(const H5::Group &g) {
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
            read_double("lambda_bar"),
            read_double("phi_esc"),
            read_double("eps"),
            read_double("vbar"));

    if (type == "polynomial")
        return std::make_unique<PolynomialPotential>(read_double("lambda_bar"));

    throw std::runtime_error("Unknown potential type: '" + type + "'. "
                             "Add a C++ implementation in cpp/common/potential.h "
                             "and register it in Potential::from_hdf5().");
}

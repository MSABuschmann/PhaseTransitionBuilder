#pragma once

#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Device-ready potential parameters
//
// CUDA device code can't dispatch through a host-constructed Potential*
// vtable, so a GPU evolution kernel needs dV() reduced to a plain tag plus a
// small fixed-size coefficient array instead. Every concrete Potential
// subclass fills this in via to_device_params(); the device-side dV switches
// on `kind`. Plain data, no CUDA types -- safe to use from any binary (a
// CPU-only build just never reads it).
// ---------------------------------------------------------------------------

enum class PotentialKind { kPhi4 = 0, kPhi4Piecewise = 1, kPolynomial = 2 };

struct DevicePotentialParams {
    PotentialKind kind;
    double c[6];   // meaning depends on `kind` -- see each subclass's
                   // to_device_params() for the exact layout.
};

// ---------------------------------------------------------------------------
// Abstract base
// ---------------------------------------------------------------------------

class Potential {
public:
    virtual ~Potential() = default;
    virtual double V(double phi)  const = 0;
    virtual double dV(double phi) const = 0;
    virtual std::string type_name() const = 0;
    virtual std::map<std::string, double> params() const = 0;
    virtual double phi_true()  const = 0;
    virtual double phi_false() const = 0;
    virtual DevicePotentialParams to_device_params() const = 0;
};

// ---------------------------------------------------------------------------
// Phi^4 potential  (arXiv:2005.13537 parametrisation)
//   V = m2*phi^2 + delta*phi^3 + lam*phi^4
//   False vacuum: phi=0,  true vacuum: phi=1.
// ---------------------------------------------------------------------------

class Phi4Potential : public Potential {
    double m2_, delta_, lam_, lambda_bar_;

public:
    explicit Phi4Potential(double lambda_bar) : lambda_bar_(lambda_bar) {
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
    std::string type_name() const override { return "phi4"; }
    std::map<std::string, double> params() const override {
        return {{"lambda_bar", lambda_bar_}};
    }
    double phi_true()  const override { return 1.0; }
    double phi_false() const override { return 0.0; }
    DevicePotentialParams to_device_params() const override {
        return {PotentialKind::kPhi4, {m2_, delta_, lam_, 0., 0., 0.}};
    }
};

// ---------------------------------------------------------------------------
// Phi^4 + piecewise extension beyond phi_esc ("bubble tails" model)
//   Phi^4 piece for phi <= phi_esc, quadratic well for phi > phi_esc.
// ---------------------------------------------------------------------------

class Phi4PiecewisePotential : public Potential {
    double m2_, delta_, lam_, phi_esc_, eps_, vbar_, Lambda_, lambda_bar_;

public:
    Phi4PiecewisePotential(double lambda_bar, double phi_esc,
                           double eps, double vbar)
        : phi_esc_(phi_esc), eps_(eps), vbar_(vbar), lambda_bar_(lambda_bar) {
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
    double phi_esc() const { return phi_esc_; }
    std::string type_name() const override { return "phi4_piecewise"; }
    std::map<std::string, double> params() const override {
        return {{"lambda_bar", lambda_bar_}, {"phi_esc", phi_esc_},
                {"eps", eps_}, {"vbar", vbar_}};
    }
    double phi_true()  const override { return vbar_; }
    double phi_false() const override { return 0.0; }
    DevicePotentialParams to_device_params() const override {
        return {PotentialKind::kPhi4Piecewise,
                {m2_, delta_, lam_, phi_esc_, eps_, vbar_}};
    }
};

// ---------------------------------------------------------------------------
// Polynomial potential  (pot_type=0, old literature)
//   V  = phi^4/4 - phi^3/3 + (1/9)*lambda_bar*phi^2
// ---------------------------------------------------------------------------

class PolynomialPotential : public Potential {
    double lambda_bar_, phi_true_;

public:
    explicit PolynomialPotential(double lambda_bar) : lambda_bar_(lambda_bar) {
        const double disc = 1.0 - (8.0 / 9.0) * lambda_bar;
        if (disc < 0)
            throw std::runtime_error("lambda_bar too large for PolynomialPotential "
                                     "(need lambda_bar <= 9/8)");
        phi_true_ = (1.0 + std::sqrt(disc)) / 2.0;
    }

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
    std::string type_name() const override { return "polynomial"; }
    std::map<std::string, double> params() const override {
        return {{"lambda_bar", lambda_bar_}};
    }
    double phi_true()  const override { return phi_true_; }
    double phi_false() const override { return 0.0; }
    DevicePotentialParams to_device_params() const override {
        return {PotentialKind::kPolynomial, {lambda_bar_, 0., 0., 0., 0., 0.}};
    }
};

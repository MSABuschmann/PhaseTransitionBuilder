#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../common/potential.h"

namespace py = pybind11;

PYBIND11_MODULE(_potential, m) {
    py::class_<Potential>(m, "Potential")
        .def("V",  &Potential::V)
        .def("dV", &Potential::dV)
        .def_property_readonly("type_name",  &Potential::type_name)
        .def_property_readonly("params",     &Potential::params)
        .def_property_readonly("phi_true",   &Potential::phi_true)
        .def_property_readonly("phi_false",  &Potential::phi_false);

    py::class_<Phi4Potential, Potential>(m, "Phi4Potential")
        .def(py::init<double>(), py::arg("lambda_bar"));

    py::class_<Phi4PiecewisePotential, Potential>(m, "Phi4PiecewisePotential")
        .def(py::init<double, double, double, double>(),
             py::arg("lambda_bar"), py::arg("phi_esc"),
             py::arg("eps"), py::arg("vbar"))
        .def_property_readonly("phi_esc", &Phi4PiecewisePotential::phi_esc);

    py::class_<PolynomialPotential, Potential>(m, "PolynomialPotential")
        .def(py::init<double>(), py::arg("lambda_bar"));
}

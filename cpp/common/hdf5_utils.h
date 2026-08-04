#pragma once

#include <H5Cpp.h>
#include <string>
#include <vector>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Attribute readers  (H5::H5Object covers H5File, Group, DataSet, …)
// ---------------------------------------------------------------------------

inline double read_attr_double(const H5::H5Object &obj, const std::string &name) {
    double val;
    obj.openAttribute(name).read(H5::PredType::NATIVE_DOUBLE, &val);
    return val;
}

inline int read_attr_int(const H5::H5Object &obj, const std::string &name) {
    int val;
    obj.openAttribute(name).read(H5::PredType::NATIVE_INT, &val);
    return val;
}

inline std::string read_attr_string(const H5::H5Object &obj, const std::string &name) {
    std::string val;
    H5::StrType stype(H5::PredType::C_S1, H5T_VARIABLE);
    obj.openAttribute(name).read(stype, val);
    return val;
}

// ---------------------------------------------------------------------------
// Attribute writers
// ---------------------------------------------------------------------------

inline void write_attr_double(H5::H5Object &obj, const std::string &name, double val) {
    H5::DataSpace scalar;
    auto attr = obj.createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalar);
    attr.write(H5::PredType::NATIVE_DOUBLE, &val);
}

inline void write_attr_int(H5::H5Object &obj, const std::string &name, int val) {
    H5::DataSpace scalar;
    auto attr = obj.createAttribute(name, H5::PredType::NATIVE_INT, scalar);
    attr.write(H5::PredType::NATIVE_INT, &val);
}

inline void write_attr_string(H5::H5Object &obj, const std::string &name,
                              const std::string &val) {
    H5::StrType stype(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace scalar;
    auto attr = obj.createAttribute(name, stype, scalar);
    attr.write(stype, val);
}

// ---------------------------------------------------------------------------
// 1-D dataset readers
// Template to work with both H5::H5File and H5::Group (H5::CommonFG was
// removed in HDF5 >= 1.12; templates cover both without needing the base).
// ---------------------------------------------------------------------------

template <typename Loc>
inline std::vector<double> read_vector(const Loc &loc, const std::string &name) {
    H5::DataSet   ds    = loc.openDataSet(name);
    H5::DataSpace space = ds.getSpace();
    hsize_t n;
    space.getSimpleExtentDims(&n);
    std::vector<double> data(n);
    ds.read(data.data(), H5::PredType::NATIVE_DOUBLE);
    return data;
}

// ---------------------------------------------------------------------------
// 1-D dataset writer
// ---------------------------------------------------------------------------

template <typename Loc>
inline void write_vector(Loc &loc, const std::string &name,
                         const std::vector<double> &data) {
    hsize_t dim = data.size();
    H5::DataSpace space(1, &dim);
    auto ds = loc.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
    ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
}

// ---------------------------------------------------------------------------
// 2-D dataset writer (row-major)
// ---------------------------------------------------------------------------

template <typename Loc>
inline void write_2d(Loc &loc, const std::string &name,
                     const std::vector<std::vector<double>> &data) {
    hsize_t rows = data.size();
    hsize_t cols = rows ? data[0].size() : 0;
    std::vector<double> flat;
    flat.reserve(rows * cols);
    for (const auto &row : data) {
        if (row.size() != cols)
            throw std::runtime_error("write_2d: ragged array for " + name);
        flat.insert(flat.end(), row.begin(), row.end());
    }
    hsize_t dims[2] = {rows, cols};
    H5::DataSpace space(2, dims);
    auto ds = loc.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
    ds.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
}

// Flat-vector variant: caller owns the row-major layout.
template <typename Loc>
inline void write_2d_flat(Loc &loc, const std::string &name,
                           const std::vector<double> &flat,
                           hsize_t rows, hsize_t cols) {
    hsize_t dims[2] = {rows, cols};
    H5::DataSpace space(2, dims);
    auto ds = loc.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
    ds.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
}

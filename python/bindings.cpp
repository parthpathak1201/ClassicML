#include "classicml.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace cml;

static Matrix np_to_matrix(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.ndim != 2) throw std::invalid_argument("X must be a 2D array");
    Matrix X(static_cast<size_t>(buf.shape[0]));
    auto ptr = static_cast<double *>(buf.ptr);
    for (py::ssize_t i = 0; i < buf.shape[0]; ++i) {
        X[static_cast<size_t>(i)].resize(static_cast<size_t>(buf.shape[1]));
        for (py::ssize_t j = 0; j < buf.shape[1]; ++j)
            X[static_cast<size_t>(i)][static_cast<size_t>(j)] = ptr[i * buf.shape[1] + j];
    }
    return X;
}

static Vec np_to_vec(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) throw std::invalid_argument("y must be a 1D array");
    Vec y(static_cast<size_t>(buf.shape[0]));
    auto ptr = static_cast<double *>(buf.ptr);
    for (py::ssize_t i = 0; i < buf.shape[0]; ++i) y[static_cast<size_t>(i)] = ptr[i];
    return y;
}

static py::array_t<double> vec_to_np(const Vec &v) {
    py::array_t<double> out(static_cast<py::ssize_t>(v.size()));
    auto ptr = out.mutable_data();
    for (size_t i = 0; i < v.size(); ++i) ptr[static_cast<py::ssize_t>(i)] = v[i];
    return out;
}

PYBIND11_MODULE(classicml_py, m) {
    m.doc() = "ClassicML Python bindings (fit / predict)";

    py::enum_<Verbosity>(m, "Verbosity")
        .value("SILENT", Verbosity::SILENT)
        .value("BASIC", Verbosity::BASIC)
        .value("DEBUG", Verbosity::DEBUG);

    m.attr("verbosity") = py::cast(&verbosity, py::return_value_policy::reference);

    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())
        .def("fit", [](LinearRegression &self, py::array_t<double> X, py::array_t<double> y, size_t epochs) {
            self.fit(np_to_matrix(X), np_to_vec(y), epochs, false);
        }, py::arg("X"), py::arg("y"), py::arg("epochs") = 100)
        .def("predict", [](LinearRegression &self, py::array_t<double> X) {
            return vec_to_np(self.predict(np_to_matrix(X)));
        }, py::arg("X"));

    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<>())
        .def("fit", [](LogisticRegression &self, py::array_t<double> X, py::array_t<double> y, size_t epochs) {
            self.fit(np_to_matrix(X), np_to_vec(y), epochs, false);
        }, py::arg("X"), py::arg("y"), py::arg("epochs") = 100)
        .def("predict", [](LogisticRegression &self, py::array_t<double> X) {
            return vec_to_np(self.predict(np_to_matrix(X)));
        }, py::arg("X"));

    py::class_<KMeans>(m, "KMeans")
        .def(py::init<int>(), py::arg("k") = 3)
        .def("fit", [](KMeans &self, py::array_t<double> X) { self.fit(np_to_matrix(X), false); }, py::arg("X"))
        .def("predict", [](KMeans &self, py::array_t<double> X) {
            auto labels = self.predict(np_to_matrix(X));
            Vec out(labels.begin(), labels.end());
            return vec_to_np(out);
        }, py::arg("X"));
}

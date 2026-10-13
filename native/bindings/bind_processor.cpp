#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stevejob/core/cache.hpp"
#include "stevejob/core/hash.hpp"
#include "stevejob/results/native_result.hpp"

namespace py = pybind11;

PYBIND11_MODULE(native_engine, module) {
    module.doc() = "C++ accelerated routines for agent workflows";

    py::class_<stevejob::results::NativeResult>(module, "NativeResult")
        .def(py::init<>())
        .def_readwrite("cache_hit", &stevejob::results::NativeResult::cache_hit)
        .def_readwrite("iterations", &stevejob::results::NativeResult::iterations)
        .def_readwrite("threads", &stevejob::results::NativeResult::threads)
        .def_readwrite("count", &stevejob::results::NativeResult::count)
        .def_readwrite("metrics", &stevejob::results::NativeResult::metrics)
        .def_readwrite("checksum", &stevejob::results::NativeResult::checksum)
        .def_readwrite("elapsed_ns", &stevejob::results::NativeResult::elapsed_ns)
        .def("to_dict", &stevejob::results::NativeResult::to_python_dict);

    py::class_<stevejob::core::IdempotencyCache>(module, "IdempotencyCache")
        .def(py::init<>())
        .def("clear", &stevejob::core::IdempotencyCache::clear)
        .def("put", &stevejob::core::IdempotencyCache::put)
        .def("get", [](stevejob::core::IdempotencyCache& cache, const std::string& key) -> py::object {
            stevejob::results::NativeResult out;
            if (!cache.get(key, out)) {
                return py::none();
            }
            return py::cast(out);
        });

    module.def("idempotency_cache",
               []() -> stevejob::core::IdempotencyCache& { return stevejob::core::process_cache(); },
               py::return_value_policy::reference,
               "Process-wide idempotency cache used by fast_data_crunch");
    module.def("fnv1a64",
               [](const std::vector<double>& samples) {
                   return stevejob::core::to_hex(
                       stevejob::core::fnv1a64(samples.data(), samples.size()));
               },
               py::arg("samples"),
               "Little-endian FNV-1a 64 of float64 samples, as 16 hex characters");
    module.def(
        "fast_data_crunch",
        [](const std::string& input_json, int iterations, const std::string& idempotency_key) {
            stevejob::results::NativeResult result;
            {
                py::gil_scoped_release release;
                result = stevejob::core::crunch(
                    input_json, iterations, idempotency_key, stevejob::core::process_cache());
            }
            return result.to_python_dict();
        },
        py::arg("input_json"),
        py::arg("iterations") = 1,
        py::arg("idempotency_key") = "",
        "Smooth a numeric payload and reduce it to deterministic stats");
    module.def(
        "clear_idempotency_cache",
        []() { stevejob::core::process_cache().clear(); },
        "Drop in-process idempotency results. Tests and a fresh worker use this.");
}

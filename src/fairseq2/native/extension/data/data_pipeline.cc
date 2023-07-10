// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/extension/module.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <fairseq2/native/exception.h>
#include <fairseq2/native/data/byte_stream.h>
#include <fairseq2/native/data/collater.h>
#include <fairseq2/native/data/element_mapper.h>
#include <fairseq2/native/data/data.h>
#include <fairseq2/native/data/data_length_extractor.h>
#include <fairseq2/native/data/data_pipeline.h>
#include <fairseq2/native/data/file_mapper.h>
#include <fairseq2/native/data/record_reader.h>
#include <fairseq2/native/data/tape.h>
#include <fairseq2/native/detail/exception.h>

namespace py = pybind11;

using namespace fairseq2::detail;

namespace fairseq2 {
namespace detail {
namespace {

// This class help us to gracefully delete data pipelines with active daemon
// threads (e.g. a running prefetch op) during Python interpreter shutdown.
class data_pipeline_tracker {
    struct py_handle_hash {
        std::size_t
        operator()(const py::handle &h) const noexcept
        {
            return std::hash<void *>{}(h.ptr());
        }
    };

public:
    // Registers a hook with the `atexit` module to delete any data pipeline
    // that is still alive.
    void
    register_atexit_hook();

    // Delete `pipeline` during interpreter shutdown if it is still alive.
    void
    track(py::object pipeline);

private:
    void
    reset_alive_pipelines();

private:
    std::unordered_set<py::weakref, py_handle_hash> alive_pipelines_{};
};

void
data_pipeline_tracker::register_atexit_hook()
{
    py::module_ atexit_module = py::module_::import("atexit");

    auto hook = [this]
    {
        reset_alive_pipelines();
    };

    atexit_module.attr("register")(py::cpp_function{hook});
}

void
data_pipeline_tracker::track(py::object pipeline)
{
    // This `weakref` callback will be called when `pipeline` gets deleted
    // before interpreter shutdown. In such case, we just stop tracking it.
    auto remove_weakref = [this](const py::weakref &weakref)
    {
        alive_pipelines_.erase(weakref);
    };

    // We internally store a weak reference to `pipeline`. If it is still alive
    // by the time the interpreter is shutdown, we will use this weak reference
    // to get a handle to it.
    alive_pipelines_.emplace(std::move(pipeline), py::cpp_function{remove_weakref});
}

void
data_pipeline_tracker::reset_alive_pipelines()
{
    for (auto &weakref : alive_pipelines_) {
        py::object pipeline_obj = weakref();

        if (pipeline_obj.is_none())
            throw_<internal_error>(
                "One of the tracked data pipelines has already been deleted. Please file a bug report.");

        auto &pipeline = pipeline_obj.cast<data_pipeline &>();

        // A broken data pipeline does not have any active daemon threads.
        if (pipeline.is_broken())
            continue;

        {
            py::gil_scoped_release no_gil{};

            // By calling `reset()`, we indirectly stop all active daemon
            // threads used within `pipeline`.
            try {
                pipeline.reset();
            } catch (const data_pipeline_error &) {}
        }
    }

    alive_pipelines_.clear();
}

data_pipeline_tracker &
data_pipeline_tracker() noexcept
{
    static class data_pipeline_tracker tracker{};

    return tracker;
}

class data_pipeline_iterator {
public:
    explicit
    data_pipeline_iterator(data_pipeline &p) noexcept
      : pipeline_{&p}
    {}

    data
    next()
    {
        std::optional<data> d{};

        {
            py::gil_scoped_release no_gil{};

            d = pipeline_->next();
        }

        if (!d)
            throw py::stop_iteration();

        return *std::move(d);
    }

private:
    data_pipeline *pipeline_;
};

}  // namespace
}  // namespace detail

void
def_data_pipeline(py::module_ &data_module)
{
    data_pipeline_tracker().register_atexit_hook();

    py::module_ m = data_module.def_submodule("data_pipeline");

    // DataPipeline
    py::class_<data_pipeline>(m, "DataPipeline")
        .def(py::init<>())

        .def(
            "__iter__",
            [](data_pipeline &self)
            {
                return data_pipeline_iterator{self};
            },
            py::keep_alive<0, 1>{})

        .def("reset", &data_pipeline::reset, py::call_guard<py::gil_scoped_release>{})

        .def_property_readonly("is_broken", &data_pipeline::is_broken)

        // state_dict
        .def(
            "state_dict",
            [](const data_pipeline &self)
            {
                tape t{};

                {
                    py::gil_scoped_release no_gil{};

                    self.record_position(t);
                }

                return py::dict{py::arg("position") = py::cast(t.storage())};
            })
        .def(
            "load_state_dict",
            [](data_pipeline &self, const py::dict &state_dict, bool strict)
            {
                py::object value;
                try {
                    value = state_dict["position"];
                } catch (const py::error_already_set &ex) {
                    if (ex.matches(PyExc_KeyError) && !strict)
                        return;

                    throw;
                }

                data_list storage{};
                try {
                    storage = value.cast<data_list>();
                } catch (const py::cast_error &) {
                    throw_<std::invalid_argument>(
                        "`state_dict` must contain a valid data pipeline state, but cannot be parsed as such.");
                }

                tape t{std::move(storage)};

                {
                    py::gil_scoped_release no_gil{};

                    self.reload_position(t);
                }
            },
            py::arg("state_dict"),
            py::arg("strict") = true)

        // Factories
        .def_static(
            "zip",
            [](
                std::vector<std::reference_wrapper<data_pipeline>> &refs,
                std::optional<std::vector<std::string>> names,
                bool flatten,
                bool warn_only,
                bool disable_parallelism)
            {
                std::vector<data_pipeline> pipelines{};

                pipelines.reserve(refs.size());

                std::transform(
                    refs.begin(), refs.end(), std::back_inserter(pipelines), [](auto &r) {
                        return std::move(r.get());
                    });

                return data_pipeline::zip(
                    std::move(pipelines),
                    std::move(names),
                    flatten, warn_only,
                    disable_parallelism);
            },
            py::arg("pipelines"),
            py::arg("names") = std::nullopt,
            py::arg("flatten") = false,
            py::arg("warn_only") = false,
            py::arg("disable_parallelism") = false)
        .def_static(
            "round_robin",
            [](std::vector<std::reference_wrapper<data_pipeline>> &refs)
            {
                std::vector<data_pipeline> pipelines{};

                pipelines.reserve(refs.size());

                std::transform(
                    refs.begin(), refs.end(), std::back_inserter(pipelines), [](auto &r) {
                        return std::move(r.get());
                    });

                return data_pipeline::round_robin(std::move(pipelines));
            },
            py::arg("pipelines"));

    // DataPipelineIterator
    py::class_<data_pipeline_iterator>(m, "_DataPipelineIterator")
        .def(
            "__iter__",
            [](data_pipeline_iterator &self) -> data_pipeline_iterator &
            {
                return self;
            })
        .def("__next__", &data_pipeline_iterator::next);

    // DataPipelineBuilder
    py::class_<data_pipeline_builder>(m, "DataPipelineBuilder")
        .def(
            "bucket",
            [](
                data_pipeline_builder &self,
                std::size_t bucket_size,
                bool drop_remainder) -> data_pipeline_builder &
            {
                self = std::move(self).bucket(bucket_size, drop_remainder);

                return self;
            },
            py::arg("bucket_size"),
            py::arg("drop_remainder") = false)
        .def(
            "bucket_by_length",
            [](
                data_pipeline_builder &self,
                std::vector<std::pair<std::size_t, std::size_t>> bucket_sizes,
                std::optional<std::string_view> selector,
                bool drop_remainder,
                bool warn_only) -> data_pipeline_builder &
            {
                self = std::move(self).bucket_by_length(
                    std::move(bucket_sizes),
                    data_length_extractor{selector},
                    drop_remainder,
                    warn_only);

                return self;
            },
            py::arg("bucket_sizes"),
            py::arg("selector") = std::nullopt,
            py::arg("drop_remainder") = false,
            py::arg("warn_only") = false)
        .def(
            "filter",
            [](data_pipeline_builder &self, predicate_fn fn) -> data_pipeline_builder &
            {
                self = std::move(self).filter(std::move(fn));

                return self;
            },
            py::arg("fn"))
        .def(
            "map",
            [](
                data_pipeline_builder &self,
                std::variant<map_fn, std::vector<map_fn>> fn,
                std::optional<std::string_view> selector,
                std::size_t num_parallel_calls,
                bool warn_only) -> data_pipeline_builder &
            {
                map_fn f{};

                if (auto *map_functions = std::get_if<std::vector<map_fn>>(&fn))
                    // Combine all map functions in a single lambda and pass it
                    // to the C++ API.
                    f = [map_functions = std::move(*map_functions)](data &&d)
                    {
                        for (const map_fn &mf : map_functions)
                            d = mf(std::move(d));

                        return std::move(d);
                    };
                else
                    f = std::get<map_fn>(std::move(fn));

                self = std::move(self).map(
                    element_mapper{std::move(f), selector}, num_parallel_calls, warn_only);

                return self;
            },
            py::arg("fn"),
            py::arg("selector") = std::nullopt,
            py::arg("num_parallel_calls") = 1,
            py::arg("warn_only") = false)
        .def(
            "prefetch",
            [](data_pipeline_builder &self, std::size_t num_examples) -> data_pipeline_builder &
            {
                self = std::move(self).prefetch(num_examples);

                return self;
            },
            py::arg("num_examples"))
        .def(
            "shard",
            [](
                data_pipeline_builder &self,
                std::size_t shard_idx,
                std::size_t num_shards) -> data_pipeline_builder &
            {
                self = std::move(self).shard(shard_idx, num_shards);

                return self;
            },
            py::arg("shard_idx"),
            py::arg("num_shards"))
        .def(
            "shuffle",
            [](
                data_pipeline_builder &self,
                std::size_t shuffle_window,
                bool strict,
                bool enabled) -> data_pipeline_builder &
            {
                self = std::move(self).shuffle(shuffle_window, strict, enabled);

                return self;
            },
            py::arg("shuffle_window"),
            py::arg("strict") = true,
            py::arg("enabled") = true)
        .def(
            "skip",
            [](data_pipeline_builder &self, std::size_t num_examples) -> data_pipeline_builder &
            {
                self = std::move(self).skip(num_examples);

                return self;
            },
            py::arg("num_examples"))
        .def(
            "take",
            [](data_pipeline_builder &self, std::size_t num_examples) -> data_pipeline_builder &
            {
                self = std::move(self).take(num_examples);

                return self;
            },
            py::arg("num_examples"))
        .def(
            "yield_from",
            [](data_pipeline_builder &self, yield_fn fn) -> data_pipeline_builder &
            {
                self = std::move(self).yield_from(std::move(fn));

                return self;
            },
            py::arg("fn"))
        .def(
            "and_return",
            [](data_pipeline_builder &self)
            {
                data_pipeline pipeline = std::move(self).and_return();

                py::object obj = py::cast(std::move(pipeline));

                // Ensure that the pipeline gets deleted during interpreter
                // shutdown if it is still alive.
                data_pipeline_tracker().track(obj);

                return obj;
            });

    // DataPipelineError
    static py::exception<data_pipeline_error> py_data_pipeline_error{
        m, "DataPipelineError", PyExc_RuntimeError};

    // Factories
    m.def("list_files", &list_files, py::arg("pathname"), py::arg("pattern") = std::nullopt);

    m.def("read_sequence", &read_list, py::arg("seq"));

    m.def("read_zipped_records", &read_zipped_records, py::arg("pathname"));

    // Collater
    py::class_<collater, std::shared_ptr<collater>>(m, "Collater")
        .def(py::init<std::optional<std::int64_t>>(), py::arg("pad_idx") = std::nullopt)
        .def("__call__", &collater::operator(), py::call_guard<py::gil_scoped_release>{});

    map_functors().register_<collater>();

    // FileMapper
    py::class_<file_mapper, std::shared_ptr<file_mapper>>(m, "FileMapper")
        .def(
            py::init<std::optional<std::string>, std::optional<std::size_t>>(),
            py::arg("root_dir") = std::nullopt,
            py::arg("cached_fd_count") = std::nullopt)
        .def("__call__", &file_mapper::operator(), py::call_guard<py::gil_scoped_release>{});

    map_functors().register_<file_mapper>();

    // RecordError
    static py::exception<record_error> py_record_error{m, "RecordError", PyExc_RuntimeError};

    // ByteStreamError
    static py::exception<byte_stream_error> py_byte_stream_error{
        m, "ByteStreamError", PyExc_RuntimeError};

    // TODO: Remove once https://github.com/pybind/pybind11/pull/4366 lands.
    py::register_exception_translator([](std::exception_ptr ptr)
    {
        if (!ptr)
            return;

        auto raise_error = [&ptr](const std::exception &e, const py::object &err) {
            py::detail::handle_nested_exception(e, ptr);

            py::detail::raise_err(err.ptr(), e.what());
        };

        try {
            std::rethrow_exception(ptr);
        } catch (const byte_stream_error &e) {
            raise_error(e, py_byte_stream_error);
        } catch (const record_error &e) {
            raise_error(e, py_record_error);
        } catch (const data_pipeline_error &e) {
            raise_error(e, py_data_pipeline_error);
        }
    });
}

}  // namespace fairseq2

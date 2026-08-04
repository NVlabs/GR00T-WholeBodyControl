// Set up a Doxygen group.
/** @addtogroup Main
 *  @{
 */

#include "ClientLogging.hpp"
#include "SDKClient.hpp"
#include <pybind11/pybind11.h>
#include <map>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdio>

namespace py = pybind11;


ClientReturnCode t_Result;
SDKClient t_SDKClient;

int init() {
	t_SDKClient.Initialize();
	t_SDKClient.Run();
	while (output_map.size() == 0){
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	return 0;
}

py::dict get_latest_state() {
    py::dict py_dict;
    for (const auto& pair : output_map) {
        py::list py_list;  // Initialize a py::list for each vector
        for (double value : pair.second) {
            py_list.append(value);  // Append each value from the vector to the py::list
        }
        py_dict[py::str(pair.first)] = py_list;  // Assign the list to the dict with the string key
    }
    return py_dict;
}

int shutdown() {
    t_SDKClient.ShutDown();
    std::cout<<"Manus shutdown\n";
    return 0;
}

PYBIND11_MODULE(ManusServer, m) {
    m.def("init", &init);
	m.def("get_latest_state", &get_latest_state);
    m.def("shutdown", &shutdown);
}
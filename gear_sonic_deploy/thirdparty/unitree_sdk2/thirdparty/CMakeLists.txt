find_package(Threads REQUIRED)

message(STATUS "Importing: ${CMAKE_CURRENT_LIST_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}/libddsc.so")

add_library(ddsc SHARED IMPORTED GLOBAL)
set_target_properties(ddsc PROPERTIES
        IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}/libddsc.so)
set_target_properties(ddsc PROPERTIES IMPORTED_NO_SONAME TRUE)
target_link_libraries(ddsc INTERFACE Threads::Threads)
target_link_directories(ddsc INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR})
target_include_directories(ddsc INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

message(STATUS "Importing: ${CMAKE_CURRENT_LIST_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}/libddscxx.so")

add_library(ddscxx SHARED IMPORTED GLOBAL)
set_target_properties(ddscxx PROPERTIES
        IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}/libddscxx.so)
set_target_properties(ddscxx PROPERTIES IMPORTED_NO_SONAME TRUE)
target_link_libraries(ddscxx INTERFACE Threads::Threads)
target_link_directories(ddscxx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR})
target_include_directories(ddscxx INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ddscxx>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/ddscxx>)

## Install targets
file(GLOB DDS_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}/*")

install(FILES ${DDS_LIBRARIES} DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

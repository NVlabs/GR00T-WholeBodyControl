if(NOT DEFINED BINARY OR NOT EXISTS "${BINARY}")
  message(FATAL_ERROR "true23 hold smoke binary does not exist: ${BINARY}")
endif()
if(NOT DEFINED SOURCE OR NOT EXISTS "${SOURCE}")
  message(FATAL_ERROR "true23 hold smoke source does not exist: ${SOURCE}")
endif()
find_program(STRINGS_EXECUTABLE strings)
if(NOT STRINGS_EXECUTABLE)
  message(FATAL_ERROR "strings is required for hold-smoke surface audit")
endif()
execute_process(
  COMMAND "${STRINGS_EXECUTABLE}" -a "${BINARY}"
  RESULT_VARIABLE STRINGS_RESULT
  OUTPUT_VARIABLE BINARY_SURFACE
  ERROR_VARIABLE STRINGS_ERROR)
if(NOT STRINGS_RESULT EQUAL 0)
  message(FATAL_ERROR "strings failed: ${STRINGS_ERROR}")
endif()
file(READ "${SOURCE}" SOURCE_SURFACE)
set(COMBINED_SURFACE "${BINARY_SURFACE}\n${SOURCE_SURFACE}")
foreach(FORBIDDEN
    "disable-crc"
    "disableCrcCheck"
    "onnxruntime"
    "PicoEncoderTerms"
    "SubmitPolicy"
    "g1_deploy_onnx_ref.cpp")
  string(FIND "${COMBINED_SURFACE}" "${FORBIDDEN}" FOUND_OFFSET)
  if(NOT FOUND_OFFSET EQUAL -1)
    message(FATAL_ERROR
      "hold smoke contains forbidden bypass/policy surface: ${FORBIDDEN}")
  endif()
endforeach()
foreach(REQUIRED
    "I_CONFIRM_G1_TRUE23_STAGE1_GANTRY"
    "rt/lowstate"
    "rt/lowcmd"
    "five advancing CRC-valid mode_machine==4"
    "POLICY-FREE GANTRY TEST ONLY"
    "hold L2 then press A"
    "B/R2 is STOP")
  string(FIND "${COMBINED_SURFACE}" "${REQUIRED}" FOUND_OFFSET)
  if(FOUND_OFFSET EQUAL -1)
    message(FATAL_ERROR
      "hold smoke missing required gate surface: ${REQUIRED}")
  endif()
endforeach()
message(STATUS "true23 policy-free hold-smoke surface audit passed")

if(NOT DEFINED BINARY OR NOT EXISTS "${BINARY}")
  message(FATAL_ERROR "true23 shadow binary does not exist: ${BINARY}")
endif()

find_program(STRINGS_EXECUTABLE strings)
find_program(NM_EXECUTABLE nm)
if(NOT STRINGS_EXECUTABLE OR NOT NM_EXECUTABLE)
  message(FATAL_ERROR "strings and nm are required for shadow isolation audit")
endif()

execute_process(
  COMMAND "${STRINGS_EXECUTABLE}" -a "${BINARY}"
  RESULT_VARIABLE STRINGS_RESULT
  OUTPUT_VARIABLE STRING_SURFACE
  ERROR_VARIABLE STRINGS_ERROR)
if(NOT STRINGS_RESULT EQUAL 0)
  message(FATAL_ERROR "strings failed: ${STRINGS_ERROR}")
endif()

execute_process(
  COMMAND "${NM_EXECUTABLE}" -C "${BINARY}"
  RESULT_VARIABLE NM_RESULT
  OUTPUT_VARIABLE SYMBOL_SURFACE
  ERROR_VARIABLE NM_ERROR)
if(NOT NM_RESULT EQUAL 0)
  message(FATAL_ERROR "nm failed: ${NM_ERROR}")
endif()

set(COMBINED_SURFACE "${STRING_SURFACE}\n${SYMBOL_SURFACE}")
foreach(FORBIDDEN
    "rt/lowcmd"
    "LowCmd"
    "ChannelPublisher"
    "MotionSwitcher"
    "HG_CMD_TOPIC"
    "MotorCommand"
    "CreateSendChannel")
  string(FIND "${COMBINED_SURFACE}" "${FORBIDDEN}" FOUND_OFFSET)
  if(NOT FOUND_OFFSET EQUAL -1)
    message(FATAL_ERROR
      "shadow binary contains forbidden command surface: ${FORBIDDEN}")
  endif()
endforeach()

if(REQUIRE_SAFE_OUTPUT)
  foreach(FORBIDDEN_TRANSFORM_SYMBOL
      " tanh@"
      " tanhf@"
      " tanhl@"
      " std::tanh(")
    string(FIND "${SYMBOL_SURFACE}" "${FORBIDDEN_TRANSFORM_SYMBOL}"
      FOUND_OFFSET)
    if(NOT FOUND_OFFSET EQUAL -1)
      message(FATAL_ERROR
        "live shadow binary contains forbidden external tanh transform")
    endif()
  endforeach()
  foreach(REQUIRED
      "applied_safe_native_action"
      "external_safe_target_transform_allowed"
      "g1_true23_causal_mujoco_promotion"
      "deployment_bytes_authorized"
      "executed_producer_archive_manifest_sha256"
      "74f2277042da83e81ee8a37d90ba6e723bf6e0651ee9b9987ee7effc78fca516")
    string(FIND "${STRING_SURFACE}" "${REQUIRED}" FOUND_OFFSET)
    if(FOUND_OFFSET EQUAL -1)
      message(FATAL_ERROR
        "shadow binary is missing V11 safe-output gate: ${REQUIRED}")
    endif()
  endforeach()
endif()

message(STATUS "true23 shadow binary has no forbidden command surface")

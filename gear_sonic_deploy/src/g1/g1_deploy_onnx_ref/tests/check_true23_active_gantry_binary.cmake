if(NOT DEFINED BINARY OR NOT EXISTS "${BINARY}")
  message(FATAL_ERROR "true23 active gantry binary does not exist: ${BINARY}")
endif()
if(NOT DEFINED SOURCE OR NOT EXISTS "${SOURCE}")
  message(FATAL_ERROR "true23 active gantry source does not exist: ${SOURCE}")
endif()

find_program(STRINGS_EXECUTABLE strings)
if(NOT STRINGS_EXECUTABLE)
  message(FATAL_ERROR "strings is required for active surface audit")
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
    "--mode control"
    "g1_deploy_onnx_ref.cpp")
  string(FIND "${COMBINED_SURFACE}" "${FORBIDDEN}" FOUND_OFFSET)
  if(NOT FOUND_OFFSET EQUAL -1)
    message(FATAL_ERROR
      "active gantry target contains forbidden bypass/generic surface: ${FORBIDDEN}")
  endif()
endforeach()

foreach(RUNTIME_REQUIRED
    "I_CONFIRM_G1_TRUE23_STAGE1_GANTRY"
    "rt/lowstate"
    "rt/lowcmd"
    "deployment_ready"
    "active_motor_control_authorized"
    "gantry_authorized"
    "--validate-only"
    "--live-shadow-evidence"
    "--authorization-id"
    "--evidence"
    "--post-arm-duration-seconds"
    "duplicate option rejected"
    "operator authorization-id does not match active promotion"
    "exact True23 causal gantry promotion and"
    "ValidateLiveShadowEvidenceJsonl"
    "promoted_shadow"
    "session_complete"
    "record count does not exactly match requested frames"
    "live shadow evidence is not fresh"
    "signal stopped controller before motion-mode release"
    "signal stopped controller after motion-mode release and before publisher"
    "signal stopped controller before command-writer startup"
    "writer completion time is invalid"
    "execution evidence path identity changed during session"
    "session_no_actuation"
    "[NO ACTUATION]"
    "first_policy_ready_for_arm"
    "reviewed_post_arm_duration_complete"
    "post_arm_elapsed_ns"
    "[WAIT] True23 publisher is damping-only"
    "waiting for first fresh 10-frame causal policy"
    "g1_true23_causal_gantry_active_promotion"
    "g1_true23_causal_mujoco_promotion"
    "deployment_bytes_authorized"
    "executed_producer_archive_manifest_sha256"
    "decoder_output_semantics"
    "applied_safe_native_action"
    "safe_target_transform_sha256"
    "action_clip_value must be exactly 20"
    "decoder must have exactly 23 native outputs"
    "five advancing CRC-valid mode_machine==4"
    "g1_true23_causal_history_reference_terms"
    "ParseCausalPicoReferenceTerms"
    "TryJoinCausalReference"
    "BuildCausalEncoderInput"
    "causal artifact rejects future-command PICO fields")
  string(FIND "${BINARY_SURFACE}" "${RUNTIME_REQUIRED}" FOUND_OFFSET)
  if(FOUND_OFFSET EQUAL -1)
    message(FATAL_ERROR
      "active gantry binary is missing runtime gate: ${RUNTIME_REQUIRED}")
  endif()
endforeach()

foreach(SOURCE_REQUIRED
    "frame_index != *last_frame + 1U"
    "*last_source_monotonic_ns + active::kShadowControlPeriodNs"
    "NextNoCatchUpWriterDeadlineNs(completed_write_ns)"
    "NextNoCatchUpWriterDeadlineNs(NowNs())"
    "first_policy_ready_for_arm.store(true, std::memory_order_release)"
    "stop_reason == \"reviewed_post_arm_duration_complete\""
    "successful_damping_writes < kFaultDampingCycles"
    "monitor.TryJoinCausalReference"
    "live::BuildCausalEncoderInput"
    "if (g_stop_requested != 0)")
  string(FIND "${SOURCE_SURFACE}" "${SOURCE_REQUIRED}" FOUND_OFFSET)
  if(FOUND_OFFSET EQUAL -1)
    message(FATAL_ERROR
      "active gantry source is missing structural gate: ${SOURCE_REQUIRED}")
  endif()
endforeach()

message(STATUS "true23 active gantry surface audit passed")

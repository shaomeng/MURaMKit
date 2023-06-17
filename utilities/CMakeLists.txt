add_executable( smart_log smart_log.c )
target_link_libraries( smart_log PUBLIC MURaMKit)

add_executable( slice_norm slice_norm.c )
target_link_libraries( slice_norm PUBLIC MURaMKit)

if (INTEGRATE_SPERR)
  add_executable (muram_sperr muram_sperr.cpp)
  target_link_libraries (muram_sperr PUBLIC MURaMKit PUBLIC PkgConfig::SPERR)
endif ()

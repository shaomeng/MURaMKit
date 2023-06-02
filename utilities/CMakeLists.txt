add_executable( smart_log smart_log.c )
target_link_libraries( smart_log PUBLIC MURaMKit)

add_executable( slice_norm slice_norm.c )
target_link_libraries( slice_norm PUBLIC MURaMKit)

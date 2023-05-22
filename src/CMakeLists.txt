add_library( MURaMKit
             Bitmask.cpp
             MURaMKit.cpp )
             
target_include_directories( MURaMKit PUBLIC ${CMAKE_SOURCE_DIR}/include )


#
# Also link zstd
#
#target_link_libraries( MURaMKit PUBLIC PkgConfig::ZSTD )


set_target_properties( MURaMKit PROPERTIES VERSION ${MURaMKit_VERSION} )

#
# The list of headers is formatted a little cumbersome, but don't change it!
#
set( public_h_list 
"include/Bitmask.h;\
include/MURaMKit.cpp;")
set_target_properties( MURaMKit PROPERTIES PUBLIC_HEADER "${public_h_list}" )


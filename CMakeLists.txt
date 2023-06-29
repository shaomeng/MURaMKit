# Tutorial: https://cmake.org/cmake/help/latest/guide/tutorial/index.html

cmake_minimum_required(VERSION 3.14)

project(MURaMKit VERSION 0.0.1 DESCRIPTION "Conditioning MURaM Output")

#
# specify the C++ standard
# CMake will try to add a -std=c++20 flag if the compiler supports, but if it doesn't,
# CMake will `decay` to a previous supported flag, e.g. -std=c++11.
#
set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

include(CMakeDependentOption)
include(TestBigEndian)
test_big_endian(IS_BIG_ENDIAN)
if(IS_BIG_ENDIAN)
 message(FATAL_ERROR "CMake detects that the target machine is BIG ENDIAN,\
                      which MURaMKit cannot work on!")
else()
 message(STATUS "Little endian machine, good!")
endif()

option( BUILD_SHARED_LIBS "Build shared library" ON )
option( BUILD_CLI_UTILITIES "Build a set of command line utilities" ON )
option( INTEGRATE_SPERR "Integrate with existing SPERR library" OFF )

if (INTEGRATE_SPERR)
  set (SPERR_INSTALL_DIR "SPERR INSTALL DIR" CACHE STRING "(Only needed when INTEGRATE_SPERR is ON)")
  list(APPEND CMAKE_PREFIX_PATH ${SPERR_INSTALL_DIR})
  find_package(PkgConfig REQUIRED)
  pkg_search_module(SPERR REQUIRED IMPORTED_TARGET GLOBAL SPERR)
endif()


find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found! (${OpenMP_CXX_LIB_NAMES}-${OpenMP_CXX_VERSION}: ${OpenMP_CXX_FLAGS})")
else()
  message(FATAL_ERROR "OpenMP NOT found! Try turning it off!")
endif()

#
# Basically always use full rpath when installing.
# These specifications need to be placed before defining any targets.
#
set( CMAKE_SKIP_BUILD_RPATH             FALSE )
set( CMAKE_BUILD_WITH_INSTALL_RPATH     FALSE )
set( CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" )
set( CMAKE_INSTALL_RPATH_USE_LINK_PATH  TRUE  )

#
# Compile the MURaMKit library
#
add_subdirectory( src )

#
# Build command line utilities
#
if( BUILD_CLI_UTILITIES )
  add_subdirectory( utilities ${CMAKE_BINARY_DIR}/bin )
endif()


#
# Start installation using GNU installation rules
#
include( GNUInstallDirs )

# Install the MURaMKit library
#
if( BUILD_SHARED_LIBS )
  install( TARGETS MURaMKit LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
           PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
else()
  install( TARGETS MURaMKit ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
           PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )
endif()

# Install utilities
#
if( BUILD_CLI_UTILITIES )
  install( TARGETS smart_log slice_norm
           RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )
endif()

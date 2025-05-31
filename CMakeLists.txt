cmake_minimum_required(VERSION 3.29)

if (DEFINED ENV{USE_VCPKG_TOOLCHAIN} AND "$ENV{USE_VCPKG_TOOLCHAIN}" STREQUAL "1")
    set(CMAKE_TOOLCHAIN_FILE "E:/vcpkg/scripts/buildsystems/vcpkg.cmake")
endif()

project(fashion_mnist VERSION 1.0.0)

add_executable(fashion_mnist 
main.cpp
)

set_target_properties(fashion_mnist PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if (MSVC)
    target_compile_options(fashion_mnist PRIVATE /W4)
else ()
    target_compile_options(fashion_mnist PRIVATE -Wall -Wextra -pedantic) 
endif()

install(TARGETS fashion_mnist RUNTIME DESTINATION bin)
set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT example@example.com)
include(CPack)



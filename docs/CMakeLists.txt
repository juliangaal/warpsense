find_package(Doxygen)

if(Doxygen_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(docs COMMAND Doxygen::doxygen ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile && xdg-open ${CMAKE_CURRENT_SOURCE_DIR}/html/index.html)
else()
    message(STATUS "Documentation cannot be built since Doxygen is not available.")
endif()

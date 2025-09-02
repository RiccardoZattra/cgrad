#ifndef TEST_LIST_CALLBACKS_HELPERS_H
#define TEST_LIST_CALLBACKS_HELPERS_H

#include "cgrad_test/datastructures/test_list/test_list.h"

/**
 * @brief Check whether a given test is failed, if yes it report an output error message
 * 
 * @param test_name Pointer 
 * @param result Pointer to the struct containing the result
 * @param user Pointer to a variable containing the number of failed tests
 * 
 * @return None
 */
void report_failures(const char *test_name, struct test_result *result, void *user);

#endif
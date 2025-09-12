#ifndef RUN_TESTS_H
#define RUN_TESTS_H

#include "cgrad_test/datastructures/test_list/test_list.h"

/**
 * @brief Executes all the test contained in the linked list
 * 
 * @param tests Constant pointer to the linked list
 * 
 * @return None
 */
void run_tests(struct test_list *const tests);

#endif
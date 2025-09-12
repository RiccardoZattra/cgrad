#ifndef TEST_LIST_H
#define TEST_LIST_H

#include "cgrad_test/config.h"
#include "cgrad_test/test_result.h"
#include "cgrad_test/test_case.h"
#include <stdlib.h>
#include <stddef.h>

struct test_list_node;

struct test_list_node
{
    struct test_list_node *next;
    struct test_list_node *prev;
    char test_name[TEST_NAME_MAX_SIZE];
    test_case test_case_func;
    struct test_result result;
};

struct test_list
{
    size_t size;
    struct test_list_node *head;
    struct test_list_node *tail;
};

typedef void (*test_list_callback)(const char *test_name, struct test_result *result, void *user);

/**
 * @brief This function allocates an empty linked list 
 * 
 * @return Pointer to a new `struct test_list` if successful,
 *         NULL if memory allocation fails.
 */
struct test_list *tests_list_alloc();

/**
 * @brief Inserts a new node into the linked list
 * 
 * @param list Constant pointer to the linked list
 * @param test_case_func  Pointer to the function that we want to test in this node 
 * @param test_name Pointer to the name of the function
 * 
 * @return None
 */
void test_list_append(struct test_list *const list, test_case test_case_func, const char *test_name);

/**
 * @brief Frees a test list safely.
 * 
 * This function releases the memory associated with the given
 * test list node. If the pointer is NULL, the function does nothing.
 * After calling this function, the node pointer should not be used.
 * 
 * @param list Pointer to the test list to be freed. Can be NULL.
 * 
 * @return None
 */
void test_list_free(struct test_list *const list);

/**
 * @brief 
 * 
 * @param
 * @param
 * @param
 * 
 * @return None
 */
void test_list_foreach(struct test_list *const list, test_list_callback callback, void *user);

#endif

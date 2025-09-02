#ifndef TENSOR_ADD_H
#define TENSOR_ADD_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/autograd/backpropagation/backpropagation.h"
#include "cgrad/autograd/computational_graph/computational_graph_link.h"
#include "cgrad/cgrad_env.h"

/**
 * @brief Add to tensors anche update the computational graph
 * 
 * @param x Constant pointer to the first operand
 * @param y Constant pointer to the second operand
 * @param out Constant pointer to a pointer to store the result
 * @param track_grad Flag selecting wheter to evaluate the gradient for the two operands
 * @param env Pointer to a struct cgrad_env 
 * 
 * @return A value ::cgrad_error indicating the result of the operation
 *         - TENSOR_NULL null pointer to x,y or out
 *         - ALLOCATORS_NULL null pointer to allocators
 *         - TENSOR_DATA_NULL no data insidere operands
 *         - TENSOR_DATA_SIZE_MISMATCH data shape mismatch between the two operands
 *         - TENSOR_SHAPE_MISMATCH shape mismatch between the two operands
 *         - TENSOR_DTYPE_MISMATCH type mismatch between the two operands
 *         - NO_ERROR no errors
 */
cgrad_error tensor_add(struct tensor *const x, struct tensor *const y, struct tensor **const out, const bool track_grad, struct cgrad_env *const env);

#endif
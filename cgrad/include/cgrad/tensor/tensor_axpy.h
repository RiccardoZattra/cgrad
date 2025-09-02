#ifndef TENSOR_AXPY_H
#define TENSOR_AXPY_H

#include "cgrad/tensor/tensor.h"
#include "cgrad/error.h"

/**
 * @brief Perform a*x+y wiht x,y tensors of general type, 'a' scalar
 * 
 * @param x Constant pointer to the x
 * @param y Constant pointer to the y 
 * @param alpha Constant double for the scalar
 * 
 * @return A value ::cgrad_error indicating the result of the operation
 *         - TENSOR_SHAPE_MISMATCH shape mismatch between the two operands 
 *         - TENSOR_DTYPE_MISMATCH type mismatch between the two operands
 *         - NO_ERROR no errors
 */
cgrad_error tensor_axpy(const struct tensor *const x, struct tensor *const y, const double a);

#endif
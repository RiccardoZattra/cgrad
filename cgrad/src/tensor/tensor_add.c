#include "cgrad/tensor/tensor_add.h"
#include "cgrad/tensor/tensor_copy.h"
#include "cgrad/tensor/tensor_helpers.h"
#include "cgrad/tensor/tensor_equality.h"
#include "cgrad/autograd/computational_graph/computational_graph.h"

typedef enum tensor_add_operand
{
    LHS_TENSOR,
    RHS_TENSOR,
} tensor_add_operand;

/**
 * @brief Update the computational graph related when performing the add operation
 * 
 * @param x Constant pointer to the first operand
 * @param y Constant pointer to the second operand
 * @param out Constant pointer to a pointer to store the result
 * @param env Pointer to a struct cgrad_env 
 * 
 * @return A value ::cgrad_error indicating the result of the operation
 *         - TENSOR_NULL null pointer to x,y or out
 *         - TENSOR_GRAD_NULL null pointer to gradient tensor of x,y or out
 *         - ALLOCATORS_NULL null pointer to allocators contained in env
 *         - AUTOGRAD_BACKPROPAGATION_FUNCTION_NULL null pointer to the backpropagation function
 *         - NO_ERROR no errors
 */
static inline cgrad_error tensor_add_update_graph(struct tensor *const x, struct tensor *const y, struct tensor **const out, struct cgrad_env *const env);

/**
 * @brief Wrapper function to add two tensor according to their value's type
 * 
 * @param x Constant pointer to the first operand
 * @param y Constant pointer to the second operand
 * @param out Constant pointer to output operand
 * 
 * @return A value ::cgrad_error indicating the result of the operation
 *         - OPERATION_INVALID_TENSOR_DTYPE type not supported
 *         - NO_ERROR no errors
 */
static inline cgrad_error tensor_add_dispatch(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);

/**
 * @brief  Computes the sum of two 64-bit floating point tensors.
 * 
 * @param x Constant pointer to the first operand
 * @param y Constant pointer to the second operand
 * @param out Constant pointer to output operand
 * 
 * @return A value ::cgrad_error indicating the result of the operation
 *         - NO_ERROR no errors
 */
static cgrad_error tensor_add_f64(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);

/**
 * @brief  Computes the sum of two 32-bit floating point tensors.
 * 
 * @param x Constant pointer to the first operand
 * @param y Constant pointer to the second operand
 * @param out Constant pointer to output operand
 * 
 * @return A value ::cgrad_error indicating the result of the operation
 *         - NO_ERROR no errors
 */
static cgrad_error tensor_add_f32(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);

/**
 * @brief  Computes the sum of two 32-bit integer tensors.
 * 
 * @param x Constant pointer to the first operand
 * @param y Constant pointer to the second operand
 * @param out Constant pointer to output operand
 * 
 * @return A value ::cgrad_error indicating the result of the operation
 *         - NO_ERROR no errors
 */
static cgrad_error tensor_add_i32(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);

/**
 * @brief  Computes the sum of two 16-bit integer tensors.
 * 
 * @param x Constant pointer to the first operand
 * @param y Constant pointer to the second operand
 * @param out Constant pointer to output operand
 * 
 * @return A value ::cgrad_error indicating the result of the operation
 *         - NO_ERROR no errors
 */
static cgrad_error tensor_add_i16(const struct tensor *const x, const struct tensor *const y, struct tensor *const out);

/**
 * @brief  Backpropagate the gradient for the operation
 * 
 * @param ctx Constant pointer to a struct backpropagation_context 
 * @param grad_wrt_out Constant pointer to a tensor containing the gradient with respect to output
 * @param grad_wrt_operand Constant pointer to a tensor where to save the gradient with respect to the operand
 * 
 * @return A value ::cgrad_error indicating the result of the operation
 *         - NO_ERROR no errors
 */
static cgrad_error tensor_add_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand);

cgrad_error tensor_add(struct tensor *const x, struct tensor *const y, struct tensor **const out, const bool track_grad, struct cgrad_env *const env)
{
    if (!env)
    {
        return ALLOCATORS_NULL;
    }
    if (!x || !y)
    {
        return TENSOR_NULL;
    }
    if (!x->data || !y->data)
    {
        return TENSOR_DATA_NULL;
    }
    if (x->data_size != y->data_size)
    {
        return TENSOR_DATA_SIZE_MISMATCH;
    }
    if (!tensor_same_shape(x, y))
    {
        return TENSOR_SHAPE_MISMATCH;
    }
    if (x->dtype != y->dtype)
    {
        return TENSOR_DTYPE_MISMATCH;
    }

    (*out) = tensor_allocator_alloc(&env->tensor_alloc, x->shape, x->shape_size, x->dtype);

    cgrad_error err = tensor_add_dispatch(x, y, *out);
    if (err != NO_ERROR)
    {
        return err;
    }

    if (track_grad)
    {
        return tensor_add_update_graph(x, y, out, env);
    }

    return NO_ERROR;
}

static inline cgrad_error tensor_add_update_graph(struct tensor *const x, struct tensor *const y, struct tensor **const out, struct cgrad_env *const env)
{
    cgrad_error err = add_computational_graph_link(x, LHS_TENSOR, *out, &tensor_add_backpropagate, env);
    if (err != NO_ERROR)
    {
        return err;
    }

    err = add_computational_graph_link(y, RHS_TENSOR, *out, &tensor_add_backpropagate, env);

    return err;
}

static cgrad_error tensor_add_backpropagate(const struct backpropagation_context *const ctx, const struct tensor *const grad_wrt_out, struct tensor *grad_wrt_operand)
{
    /**
     * Given the symmetry of the addition operation, the gradient with respect to both operands is the same.
     * Therefore, we can use the same gradient for both operands.
     * The gradient with respect to both operands is the gradient with respect to the output.
     */
    return tensor_copy(grad_wrt_out, grad_wrt_operand);
}

static inline cgrad_error tensor_add_dispatch(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    switch (x->dtype)
    {
    case DTYPE_FLOAT64:
        return tensor_add_f64(x, y, out);
    case DTYPE_FLOAT32:
        return tensor_add_f32(x, y, out);
    case DTYPE_INT32:
        return tensor_add_i32(x,y,out);
    case DTYPE_INT16:
        return tensor_add_i16(x,y,out);
    default:
        return OPERATION_INVALID_TENSOR_DTYPE;
    }
}

static cgrad_error tensor_add_f64(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    double *restrict out_data = (double *)out->data;
    double *restrict A_data = (double *)x->data;
    double *restrict B_data = (double *)y->data;

    for (size_t i = 0; i < x->data_size; i++)
    {
        out_data[i] = A_data[i] + B_data[i];
    }

    return NO_ERROR;
}

static cgrad_error tensor_add_f32(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    float *restrict out_data = (float *)out->data;
    float *restrict A_data = (float *)x->data;
    float *restrict B_data = (float *)y->data;

    for (size_t i = 0; i < x->data_size; i++)
    {
        out_data[i] = A_data[i] + B_data[i];
    }

    return NO_ERROR;
}
static cgrad_error tensor_add_i32(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    int32_t *restrict out_data = (int32_t *)out->data;
    int32_t *restrict A_data = (int32_t *)x->data;
    int32_t *restrict B_data = (int32_t *)y->data;

    for (size_t i = 0; i < x->data_size; i++)
    {
        out_data[i] = A_data[i] + B_data[i];
    }

    return NO_ERROR;
}
static cgrad_error tensor_add_i16(const struct tensor *const x, const struct tensor *const y, struct tensor *const out)
{
    int16_t *restrict out_data = (int16_t *)out->data;
    int16_t *restrict A_data = (int16_t *)x->data;
    int16_t *restrict B_data = (int16_t *)y->data;

    for (size_t i = 0; i < x->data_size; i++)
    {
        out_data[i] = A_data[i] + B_data[i];
    }

    return NO_ERROR;
}

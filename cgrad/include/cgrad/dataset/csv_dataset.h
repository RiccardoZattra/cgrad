#ifndef CSV_DATASET_H
#define CSV_DATASET_H

#include "cgrad/dataset/indexes_permutation.h"
#include "cgrad/tensor/tensor.h"
#include "cgrad/cgrad_env.h"
#include "cgrad/error.h"
#include <stddef.h>

/**
 * @struct csv_dataset
 * @brief Stores a dataset loaded from a CSV file.
 *
 * The first column is assumed to be the label, and the remaining columns are features.
 */
struct csv_dataset 
{
    size_t rows;    /**< Number of data rows (excluding header). */
    size_t cols;    /**< Number of columns (features + label). */
    double *data;   /**< Flattened row-major array of data. */
};

/**
 * @brief Loads a CSV file into a csv_dataset structure.
 *
 * @param csv_path Path to the CSV file.
 * @return Pointer to the allocated csv_dataset, or NULL if allocation failed.
 */
struct csv_dataset *csv_dataset_alloc(const char *csv_path);

/**
 * @brief Extracts a batch of samples from a CSV dataset and converts them into input and target tensors.
 *
 * This function takes a subset of rows from a CSV dataset, selected using the indices
 * provided in @p ixs_batch, and creates two tensors:
 * - @p inputs: containing the features of each sample (all columns except the first).
 * - @p targets: containing the label of each sample (the first column).
 *
 * Both tensors are allocated using the memory allocator provided in @p env.
 *
 * @param[in]  dataset    Pointer to the CSV dataset from which to extract the data.
 * @param[out] inputs     Pointer to the tensor that will hold the batch features
 *                        (dimensions: [batch_size, cols-1]).
 * @param[out] targets    Pointer to the tensor that will hold the batch labels
 *                        (dimensions: [batch_size, 1]).
 * @param[in]  ixs_batch  Structure containing the row indices of the dataset to include in the batch.
 * @param[in]  dtype      Data type to use for the tensors 
 * @param[in]  env        Computation environment containing the tensor allocator.
 *
 * @return A ::cgrad_error error code:
 * - ::NO_ERROR if the operation completed successfully.
 * - ::INDEXES_BATCH_NULL if @p ixs_batch is NULL.
 * - ::TENSOR_ALLOCATION_FAILED if tensor allocation failed.
 * - Other error codes if @p dataset is invalid.
 *
 * @note The first column of the dataset is interpreted as the label (target), while the remaining
 * columns are treated as input features.
 */
cgrad_error csv_dataset_sample_batch(const struct csv_dataset *const dataset, struct tensor **const inputs, struct tensor **const targets, const struct indexes_batch *const ixs_batch, const cgrad_dtype dtype, struct cgrad_env *const env);

/**
 * @brief Applies standard scaling (zero mean, unit variance) to the dataset features.
 *
 * The first column (label) is not scaled.
 *
 * @param dataset Pointer to the csv_dataset.
 * @return NO_ERROR on success, or an error code on failure.
 */
cgrad_error csv_dataset_standard_scale(struct csv_dataset *dataset);

/**
 * @brief Checks if the dataset or its data pointer is NULL.
 *
 * @param dataset Pointer to the csv_dataset.
 * @return DATASET_NULL if dataset or data is NULL, NO_ERROR otherwise.
 */
static inline cgrad_error csv_dataset_check_null(const struct csv_dataset *const dataset);

static inline cgrad_error csv_dataset_check_null(const struct csv_dataset *const dataset)
{
    if (!dataset)
    {
        return DATASET_NULL;
    }
    if (!dataset->data)
    {
        return DATASET_NULL;
    }
    return NO_ERROR;
}

#endif
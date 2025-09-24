#include "cgrad/dataset/indexes_batch.h"
#include <stdlib.h>


struct indexes_batch *indexes_batch_alloc(const size_t capacity)
{
    struct indexes_batch *ixs_batch = malloc(sizeof(struct indexes_batch));
    if (!ixs_batch)
    {
        return NULL;
    }

    ixs_batch->indexes = calloc(capacity, sizeof(size_t));
    if (!ixs_batch->indexes)
    {
        free(ixs_batch);
        return NULL;
    }

    /***
     * Set capacity to the size of allocated memory,
     * while setting initial size to be zero, meaning the
     * container is empty.
     */
    ixs_batch->capacity = capacity;
    ixs_batch->size = 0;

    return ixs_batch;
}

void indexes_batch_free(struct indexes_batch *ixs_batch)
{
    free(ixs_batch->indexes);
    free(ixs_batch);
}
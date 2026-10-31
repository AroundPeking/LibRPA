#pragma once

/**
 * @file librpa_handler.h
 * @brief Handler management for LibRPA instances.
 */

// MPI is required to compile LibRPA.
// TODO: wrap around with LIBRPA_USE_MPI when non-MPI usage is maturer (for compatibility)
#include <mpi.h>

// C APIs
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to a LibRPA instance.
 *
 * This structure encapsulates the internal state of a LibRPA computation.
 * Users should not access its members directly; use the provided API functions.
 *
 * @note Do not create this structure manually. Use librpa_create_handler() instead.
 */
typedef struct
{
    /** Internal instance ID used by the manager. Do not modify manually. */
    const int instance_id_;
} LibrpaHandler;

/** Opaque QSGW execution state; create and delete it through librpa_compute.h. */
typedef struct LibrpaQsgwTask LibrpaQsgwTask;

/** Indexed cRPA and orbital result handles; numerical views and accessors are
 * declared in librpa_compute.h. Standalone factory results own snapshots that
 * survive handler changes or destruction. Release each owned result on its
 * rank with librpa_delete_crpa_result or librpa_delete_orbital_result.
 * Accessor arrays and strings are borrowed, read-only, and valid until the
 * owning result is deleted. The handles passed to LibrpaCrpaOutputSink are
 * borrowed instead: do not delete them or retain them or their views after
 * that callback returns. */
typedef struct LibrpaCrpaResult LibrpaCrpaResult;
typedef struct LibrpaOrbitalResult LibrpaOrbitalResult;

/** Independently owned continuation result, valid after its source cRPA result
 * is deleted. Release with librpa_delete_crpa_continued_result. Arrays returned
 * by its accessors remain borrowed and valid only until this result is deleted. */
typedef struct LibrpaCrpaContinuedResult LibrpaCrpaContinuedResult;

/**
 * @brief Create a new LibRPA handler instance.
 *
 * Allocates and initializes a new LibRPA handler associated with the given
 * MPI communicator. This handle is used in subsequent API calls to
 * perform RPA/EXX/G0W0 calculations.
 *
 * @param[in] comm MPI communicator (e.g., MPI_COMM_WORLD).
 *
 * @return Pointer to newly created LibrpaHandler. Must be freed with
 *         librpa_destroy_handler() when no longer needed.
 *
 * @see librpa_destroy_handler
 */
LibrpaHandler* librpa_create_handler(MPI_Comm comm);

/**
 * @brief Destroy a LibRPA handler instance.
 *
 * Frees all internal resources associated with the handler. After this call,
 * the handler pointer becomes invalid and should not be used.
 *
 * @param[in] h Pointer to the LibrpaHandler to destroy. If NULL, no action is taken.
 *
 * @see librpa_create_handler
 */
void librpa_destroy_handler(LibrpaHandler *h);

#ifdef __cplusplus
}
#endif

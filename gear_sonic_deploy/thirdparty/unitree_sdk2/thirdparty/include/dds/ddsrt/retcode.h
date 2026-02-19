#ifndef DDS_RETCODE_H
#define DDS_RETCODE_H

#include <stdint.h>

#include "dds/export.h"

#if defined (__cplusplus)
extern "C" {
#endif

typedef int32_t dds_return_t;

/*
  State is unchanged following a function call returning an error
  other than UNSPECIFIED, OUT_OF_RESOURCES and ALREADY_DELETED.

  Error handling functions. Three components to returned int status value.

  1 - The DDS_ERR_xxx error number
  2 - The file identifier
  3 - The line number

  All functions return >= 0 on success, < 0 on error
*/

/**
 * @defgroup retcode (Return Codes)
 * @ingroup dds
 * @{
 */
#define DDS_RETCODE_OK                   (0) /**< Success */
#define DDS_RETCODE_ERROR                (-1) /**< Non specific error */
#define DDS_RETCODE_UNSUPPORTED          (-2) /**< Feature unsupported */
#define DDS_RETCODE_BAD_PARAMETER        (-3) /**< Bad parameter value */
#define DDS_RETCODE_PRECONDITION_NOT_MET (-4) /**< Precondition for operation not met */
#define DDS_RETCODE_OUT_OF_RESOURCES     (-5) /**< When an operation fails because of a lack of resources */
#define DDS_RETCODE_NOT_ENABLED          (-6) /**< When a configurable feature is not enabled */
#define DDS_RETCODE_IMMUTABLE_POLICY     (-7) /**< When an attempt is made to modify an immutable policy */
#define DDS_RETCODE_INCONSISTENT_POLICY  (-8) /**< When a policy is used with inconsistent values */
#define DDS_RETCODE_ALREADY_DELETED      (-9) /**< When an attempt is made to delete something more than once */
#define DDS_RETCODE_TIMEOUT              (-10) /**< When a timeout has occurred */
#define DDS_RETCODE_NO_DATA              (-11) /**< When expected data is not provided */
#define DDS_RETCODE_ILLEGAL_OPERATION    (-12) /**< When a function is called when it should not be */
#define DDS_RETCODE_NOT_ALLOWED_BY_SECURITY (-13) /**< When credentials are not enough to use the function */


/* Extended return codes are not in the DDS specification and are meant
   exclusively for internal use and must not be returned by the C API. */
#define DDS_XRETCODE_BASE (-50) /**< Base offset for extended return codes */
#define DDS_XRETCODE(x) (DDS_XRETCODE_BASE - (x)) /**< Extended return code generator */

/** Requested resource is busy */
#define DDS_RETCODE_IN_PROGRESS         DDS_XRETCODE(1)
/** Resource unavailable, try again */
#define DDS_RETCODE_TRY_AGAIN           DDS_XRETCODE(2)
/** Operation was interrupted */
#define DDS_RETCODE_INTERRUPTED         DDS_XRETCODE(3)
/** Permission denied */
#define DDS_RETCODE_NOT_ALLOWED         DDS_XRETCODE(4)
/** Host not found */
#define DDS_RETCODE_HOST_NOT_FOUND      DDS_XRETCODE(5)
/** Network is not available */
#define DDS_RETCODE_NO_NETWORK          DDS_XRETCODE(6)
/** Connection is not available */
#define DDS_RETCODE_NO_CONNECTION       DDS_XRETCODE(7)
/* Host not reachable is used to indicate that a connection was refused
   (ECONNREFUSED), reset by peer (ECONNRESET) or that a host or network cannot
   be reached (EHOSTUNREACH, ENETUNREACH). Generally all system errors that
   indicate a connection cannot be made or that a message cannot be delivered,
   map onto host not reachable. */
/** Not enough space available */
#define DDS_RETCODE_NOT_ENOUGH_SPACE    DDS_XRETCODE(8)
/* Not enough space is used to indicate there is not enough buffer space to
   read a message from the network (i.e. EMSGSIZE), but is also used to
   indicate there is not enough space left on a device, etc. */
/** Result too large */
#define DDS_RETCODE_OUT_OF_RANGE        DDS_XRETCODE(9)
/** Not found */
#define DDS_RETCODE_NOT_FOUND           DDS_XRETCODE(10)


/**
 * @brief Takes the error value and outputs a string corresponding to it.
 *
 * @param[in]  ret  Error value to be converted to a string
 *
 * @returns  String corresponding to the error value
 */
DDS_EXPORT const char *dds_strretcode(dds_return_t ret);
/**
 * @}
 */
#if defined (__cplusplus)
}
#endif

#endif /* DDS_RETCODE_H */

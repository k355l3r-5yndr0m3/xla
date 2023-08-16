#pragma once
#include <stddef.h>
#include "xla/pjrt/c/pjrt_c_api.h"


#ifdef __cplusplus 
extern "C" {
#endif
const PJRT_Api* GetPjrtApi();
// const void *SerializeCompileOptions(size_t *compile_options_size);

#ifdef __cplusplus 
}
#endif


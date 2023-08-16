#include "compile_options.h"

#include "xla/pjrt/pjrt_executable.h"

const void *SerializeCompileOptions(size_t *compile_options_size) {
    // xla::CompileOptions options;
    xla::CompileOptions options;
    
    auto result = options.ToProto();
    if (!result.ok())
        return nullptr;

    auto proto = result.value();
    size_t size = proto.ByteSizeLong();
    void *buffer = malloc(size);

    if (!proto.SerializeToArray(buffer, size)) {
        free(buffer);
        return nullptr;
    }
    
    *compile_options_size = size;
    return buffer;
}

#include <stdio.h>
#include <stdlib.h>

#include "plugin.h"
#include "xla/pjrt/c/pjrt_c_api.h"

static inline void assert_pjrt_ok(const PJRT_Api *api, PJRT_Error *err) {
    if (err == NULL) return;
    PJRT_Error_Message_Args error_message_args = {
        .struct_size = PJRT_Error_Message_Args_STRUCT_SIZE, 
        .priv = NULL, 
        .error = err,
    };

    api->PJRT_Error_Message(&error_message_args);
    fprintf(stderr, "ERROR MSG: %*s\n", error_message_args.message_size, error_message_args.message);
    
    PJRT_Error_Destroy_Args error_destroy_args = {
        .struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE, 
        .priv = NULL, 
        .error = err
    };
    api->PJRT_Error_Destroy(&error_destroy_args);
    exit(-1);
}

int main() {
    const PJRT_Api *api = GetPjrtApi();
    PJRT_Client *client;
    {
        PJRT_Client_Create_Args client_create_args = {
            .struct_size = PJRT_Client_Create_Args_STRUCT_SIZE, 
            .priv = NULL, 
            .create_options = NULL, .num_options = 0,
        };
        assert_pjrt_ok(api, api->PJRT_Client_Create(&client_create_args));
        client = client_create_args.client;
    }
    {
        PJRT_Client_PlatformName_Args client_platformname_args = {
            .struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE, .priv = NULL, 
            .client = client
        };
        assert_pjrt_ok(api, api->PJRT_Client_PlatformName(&client_platformname_args));
        printf("Client platform :%*s\n", client_platformname_args.platform_name_size, client_platformname_args.platform_name);
    }
    {
        PJRT_Client_Destroy_Args client_destroy_args = {
            .struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE,
            .priv = NULL,
            .client = client,
        };
        assert_pjrt_ok(api, api->PJRT_Client_Destroy(&client_destroy_args));
    }
    return 0;
}

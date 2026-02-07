#ifndef TOTTON_AUDIO_DE_MIRRORING_DSP_MULTISTAGE_UPSAMPLER_C_API_H
#define TOTTON_AUDIO_DE_MIRRORING_DSP_MULTISTAGE_UPSAMPLER_C_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void* tadm_create_multistage_upsampler_from_dir(const char* config_dir, size_t num_stages,
                                                char* error_buffer, size_t error_buffer_length);

void tadm_destroy_multistage_upsampler(void* handle);

size_t tadm_multistage_output_length(size_t input_length, size_t num_stages);

size_t tadm_multistage_process_block(void* handle, const double* input, size_t input_length,
                                     double* output, size_t output_length, char* error_buffer,
                                     size_t error_buffer_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TOTTON_AUDIO_DE_MIRRORING_DSP_MULTISTAGE_UPSAMPLER_C_API_H

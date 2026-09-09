#ifndef CELESTEBENCH_AUDIO_H
#define CELESTEBENCH_AUDIO_H

#include <stdint.h>

struct lua_State;

void audio_init_api(struct lua_State* L);
void audio_reset(void);
void audio_destroy(void);
void audio_frame(unsigned fps);
uint32_t audio_samples(void);
void audio_copy(int16_t* out);

#endif

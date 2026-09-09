#include "audio.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "memory.h"
#include "z8lua/fix32.h"
#include "z8lua/lauxlib.h"
#include "z8lua/lua.h"

#define RATE 22050

typedef struct {
    int sfx, note, end, previous_pitch;
    float age, phase;
    bool active;
} channel;

static channel channels[4];
static int16_t* pcm;
static uint32_t pcm_len, pcm_cap, sample_remainder;
static unsigned noise = 1;
static int music_pattern = -1, music_loop = -1, music_mask = 15;
static float music_age, music_length, filtered;

static int argument(lua_State* L, int index, int fallback)
{
    return lua_isnoneornil(L, index) ? fallback : fix32_to_int(luaL_checkinteger(L, index));
}

static void play_sfx(int sfx, int slot, int offset, int length)
{
    if (slot < 0 || slot >= 4)
    {
        for (slot = 0; slot < 4 && channels[slot].active; slot++);
        if (slot == 4) slot = 0;
    }
    if (sfx < 0 || sfx >= 64)
    {
        channels[slot].active = false;
        return;
    }
    if (offset < 0) offset = 0;
    if (offset > 31) offset = 31;
    if (length < 0) length = 0;
    if (length > 32 - offset) length = 32 - offset;
    channels[slot] = (channel){sfx, offset, length ? offset + length : 32, -1, 0, 0, true};
    if (channels[slot].note >= channels[slot].end) channels[slot].active = false;
}

static void play_music(int pattern)
{
    music_pattern = pattern;
    music_age = 0;
    music_length = 0;
    if (pico8_ram[0x3100 + pattern * 4] & 0x80) music_loop = pattern;
    for (int slot = 0; slot < 4; slot++)
    {
        unsigned entry = pico8_ram[0x3100 + pattern * 4 + slot];
        if (entry & 0x40)
        {
            channels[slot].active = false;
            continue;
        }
        int sfx = entry & 0x3f;
        unsigned speed = pico8_ram[0x3200 + sfx * 68 + 65];
        if (!speed) speed = 16;
        float length = (float)speed * 32.0f / 120.0f;
        if (length > music_length) music_length = length;
        if (music_mask & (1 << slot)) play_sfx(sfx, slot, 0, 0);
    }
}

static int pico8_sfx(lua_State* L)
{
    int sfx = argument(L, 1, -1);
    int slot = argument(L, 2, -1);
    int offset = argument(L, 3, 0);
    int length = argument(L, 4, 0);
    if (sfx < 0 && slot < 0)
    {
        memset(channels, 0, sizeof(channels));
    }
    else
    {
        play_sfx(sfx, slot, offset, length);
    }
    return 0;
}

static int pico8_music(lua_State* L)
{
    int pattern = argument(L, 1, -1);
    int mask = argument(L, 3, 15);
    if (pattern < 0 || pattern >= 64)
    {
        memset(channels, 0, sizeof(channels));
        music_pattern = -1;
        return 0;
    }
    music_loop = -1;
    music_mask = mask;
    play_music(pattern);
    return 0;
}

void audio_init_api(lua_State* L)
{
    lua_pushcfunction(L, pico8_music);
    lua_setglobal(L, "music");
    lua_pushcfunction(L, pico8_sfx);
    lua_setglobal(L, "sfx");
}

void audio_reset(void)
{
    memset(channels, 0, sizeof(channels));
    pcm_len = 0;
    sample_remainder = 0;
    noise = 1;
    music_pattern = music_loop = -1;
    music_mask = 15;
    music_age = music_length = filtered = 0;
}

void audio_destroy(void)
{
    free(pcm);
    pcm = NULL;
    pcm_len = pcm_cap = 0;
}

static float wave(int instrument, float phase)
{
    float x = phase * 2.0f - 1.0f;
    switch (instrument)
    {
    case 0: return 1.0f - 2.0f * fabsf(x);
    case 1: return phase < 0.75f ? phase * (8.0f / 3.0f) - 1.0f : 7.0f - phase * 8.0f;
    case 2: return x;
    case 3: return phase < 0.5f ? 1.0f : -1.0f;
    case 4: return phase < 0.25f ? 1.0f : -1.0f;
    case 5: return sinf(phase * 12.5663706f) * 0.5f + sinf(phase * 6.2831853f) * 0.5f;
    case 6: noise = noise * 1103515245u + 12345u; return (float)((noise >> 16) & 0x7fff) / 16384.0f - 1.0f;
    default: return x < 0 ? -1.0f : 1.0f;
    }
}

static float voice(channel* voice, float seconds)
{
    uint8_t* sfx = &pico8_ram[0x3200 + voice->sfx * 68];
    unsigned speed = sfx[65] ? sfx[65] : 16;
    voice->age += seconds;
    float duration = (float)speed / 120.0f;
    while (voice->active && voice->age >= duration)
    {
        unsigned word = sfx[voice->note * 2] | ((unsigned)sfx[voice->note * 2 + 1] << 8);
        voice->previous_pitch = word & 63;
        voice->age -= duration;
        if (++voice->note >= voice->end) voice->active = false;
    }
    if (!voice->active) return 0;
    unsigned word = sfx[voice->note * 2] | ((unsigned)sfx[voice->note * 2 + 1] << 8);
    unsigned pitch = word & 63;
    unsigned instrument = (word >> 6) & 7;
    unsigned volume = (word >> 9) & 7;
    unsigned effect = (word >> 12) & 7;
    if (volume == 0) return 0;
    float progress = voice->age / duration;
    float note = pitch;
    float gain = (float)volume / 7.0f;
    if (effect == 1 && voice->previous_pitch >= 0)
        note = voice->previous_pitch + (note - voice->previous_pitch) * progress;
    else if (effect == 2)
        note += sinf(voice->age * 50.265482f) * 0.5f;
    else if (effect == 3)
        note -= progress * 8.0f;
    else if (effect == 4)
        gain *= progress;
    else if (effect == 5)
        gain *= 1.0f - progress;
    else if (effect == 6 || effect == 7)
    {
        int arp = (int)(voice->age * (effect == 6 ? 60.0f : 30.0f)) % 4;
        int index = (voice->note & ~3) + arp;
        unsigned arp_word = sfx[index * 2] | ((unsigned)sfx[index * 2 + 1] << 8);
        note = arp_word & 63;
    }
    float frequency = 440.0f * powf(2.0f, (note - 33.0f) / 12.0f);
    voice->phase += frequency / RATE;
    voice->phase -= floorf(voice->phase);
    return wave(instrument, voice->phase) * gain;
}

static void advance_music(float seconds)
{
    if (music_pattern < 0) return;
    music_age += seconds;
    if (music_age < music_length) return;
    int next = music_pattern + 1;
    uint8_t* pattern = &pico8_ram[0x3100 + music_pattern * 4];
    if (pattern[2] & 0x80)
        next = -1;
    else if (pattern[1] & 0x80)
        next = music_loop >= 0 ? music_loop : -1;
    if (next < 0 || next >= 64) music_pattern = -1;
    else play_music(next);
}

void audio_frame(unsigned fps)
{
    if (fps == 0) return;
    sample_remainder += RATE;
    unsigned samples = sample_remainder / fps;
    sample_remainder %= fps;
    if (pcm_len + samples > pcm_cap)
    {
        uint32_t cap = pcm_cap ? pcm_cap : RATE;
        while (cap < pcm_len + samples) cap *= 2;
        int16_t* next = realloc(pcm, cap * sizeof(*pcm));
        if (!next) return;
        pcm = next;
        pcm_cap = cap;
    }
    for (unsigned i = 0; i < samples; i++)
    {
        float sample = 0;
        advance_music(1.0f / RATE);
        for (int slot = 0; slot < 4; slot++)
            if (channels[slot].active) sample += voice(&channels[slot], 1.0f / RATE);
        filtered += 0.75f * (sample - filtered);
        sample = filtered;
        sample *= 14000.0f;
        if (sample > 32767.0f) sample = 32767.0f;
        if (sample < -32768.0f) sample = -32768.0f;
        pcm[pcm_len++] = (int16_t)sample;
    }
}

uint32_t audio_samples(void)
{
    return pcm_len;
}

void audio_copy(int16_t* out)
{
    if (pcm_len) memcpy(out, pcm, pcm_len * sizeof(*pcm));
    pcm_len = 0;
}

#include <SDL3/SDL.h>
// Access the upstream frame loop and VM without its interactive main().
#include "core.c"

static SDL_Renderer* renderer;

static void run_frame(void)
{
    pico8_frame_start = SDL_GetTicks();
    pico8_frame_ms = has_update60 ? 1000u / 60u : 1000u / 30u;

    if (has_update)
    {
        update_input(renderer);
        call_pico8_function(vm, "_update");
    }
    else if (has_update60)
    {
        update_input(renderer);
        call_pico8_function(vm, "_update60");
    }

    if (has_draw)
    {
        call_pico8_function(vm, "_draw");
    }

    update_time();
    update_from_virtual_memory(renderer);
}

// The PRNG survives VM destruction. Seed a fresh VM before using upstream's
// cartridge runner, which creates its own VM but preserves that seed.
static int run_cart_deterministic(void)
{
    destroy_vm();
    if (!init_vm(renderer) || luaL_dostring(vm, "srand(0)"))
    {
        return false;
    }
    touch_button_state_player_0 = 0;
    touch_button_state_player_1 = 0;
    update_input(renderer);
    return run_cartridge(renderer);
}

int shim_init(void)
{
    if (renderer != NULL)
    {
        return -1;
    }

    SDL_setenv_unsafe("SDL_VIDEODRIVER", "dummy", 0);
    SDL_setenv_unsafe("SDL_AUDIODRIVER", "dummy", 0);

    if (!init_app(&renderer, NULL))
    {
        return -1;
    }

    screen_rect.x = 0.0f;
    screen_rect.y = 0.0f;
    screen_rect.w = (float)128;
    screen_rect.h = (float)128;

    if (!init_memory(renderer))
    {
        return -2;
    }

    return 0;
}

void shim_quit(void)
{
    if (renderer == NULL)
    {
        return;
    }

    destroy_vm();
    destroy_cart(get_cart());
    destroy_memory();
    destroy_app();
    renderer = NULL;

    SDL_Quit();
}

int shim_load_cart(const char* path)
{
    if (!renderer)
    {
        return -1;
    }

    state = STATE_MENU;
    destroy_cart(get_cart());
    if (!load_cart(renderer, path, get_cart()))
    {
        return -2;
    }

    if (!run_cart_deterministic())
    {
        return -3;
    }

    return 0;
}

static int cart_loaded(void)
{
    return (renderer != NULL && state == STATE_EMULATOR) ? 1 : 0;
}

int shim_step(uint32_t frames, uint8_t buttons)
{
    if (!cart_loaded())
    {
        return 0;
    }

    for (uint32_t i = 0; i < frames; i++)
    {
        touch_button_state_player_0 = buttons;
        run_frame();
    }

    return (int)frames;
}

uint32_t shim_frame_ms(void)
{
    if (!cart_loaded())
    {
        return 0;
    }

    return has_update60 ? 16u : 33u;
}

void shim_framebuffer(uint8_t* out)
{
    for (int y = 0; y < 128; y++)
    {
        const uint8_t* src = &pico8_ram[0x6000 + (y << 6)];

        for (int x = 0; x < 128; x++)
        {
            uint8_t byte = src[x >> 1];
            uint8_t color = (x & 1) ? (byte >> 4) : (byte & 0x0F);
            uint8_t hw = pico8_ram[0x5f10 + color];
            uint8_t r, g, b;

            color_lookup(hw, &r, &g, &b);

            uint8_t* pixel = &out[(y * 128 + x) * 4];
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
            pixel[3] = 0xFF;
        }
    }
}

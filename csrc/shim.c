#include <SDL3/SDL.h>
#include <math.h>
#include <stdint.h>
// Access the upstream frame loop and VM without its interactive main().
#include "core.c"

void open8_init_api(lua_State* L);

void init_api(lua_State* L)
{
    open8_init_api(L);
}

static SDL_Renderer* renderer;

typedef struct shim_game_state
{
    int32_t room;
    int32_t alive;
    float feet_y;
    int32_t grounded;
    float spawn_feet_y;
    float exit_feet_y;
    int32_t deaths;
} shim_game_state_t;

static shim_game_state_t game_state;
static int celeste_cart;
static float spawn_feet[31];

static int global_is_function(const char* name)
{
    lua_getglobal(vm, name);
    int result = lua_isfunction(vm, -1);
    lua_pop(vm, 1);
    return result;
}

static void find_spawn_feet(void)
{
    for (int room = 0; room < 31; room++)
    {
        int x = room % 8;
        int y = room / 8;
        spawn_feet[room] = NAN;
        for (int ty = 0; ty < 16; ty++)
        {
            for (int tx = 0; tx < 16; tx++)
            {
                int row = y * 16 + ty;
                int address = row < 32
                    ? 0x2000 + row * 128 + x * 16 + tx
                    : 0x1000 + (row - 32) * 128 + x * 16 + tx;
                if (pico8_ram[address] == 1)
                {
                    spawn_feet[room] = (float)((ty + 1) * 8);
                    ty = 16;
                    break;
                }
            }
        }
    }
}

static int find_player(void)
{
    int top = lua_gettop(vm);
    lua_getglobal(vm, "objects");
    int objects = lua_gettop(vm);
    lua_getglobal(vm, "player");
    int player_type = lua_gettop(vm);

    size_t count = lua_rawlen(vm, objects);
    for (size_t i = 1; i <= count; i++)
    {
        lua_rawgeti(vm, objects, (int)i);
        if (lua_istable(vm, -1))
        {
            lua_getfield(vm, -1, "type");
            int match = lua_rawequal(vm, -1, player_type);
            lua_pop(vm, 1);
            if (match)
            {
                lua_remove(vm, objects);
                lua_remove(vm, player_type - 1);
                return lua_gettop(vm);
            }
        }
        lua_pop(vm, 1);
    }

    lua_settop(vm, top);
    return 0;
}

static float player_field(int player, const char* name)
{
    lua_getfield(vm, player, name);
    float value = (float)fix32_to_double((fix32_t)lua_tonumber(vm, -1));
    lua_pop(vm, 1);
    return value;
}

static int player_is_grounded(int player)
{
    lua_getfield(vm, player, "is_solid");
    if (!lua_isfunction(vm, -1))
    {
        lua_pop(vm, 1);
        return 0;
    }

    lua_pushnumber(vm, fix32_from_int(0));
    lua_pushnumber(vm, fix32_from_int(1));
    if (lua_pcall(vm, 2, 1, 0) != LUA_OK)
    {
        lua_pop(vm, 1);
        return 0;
    }

    int grounded = lua_toboolean(vm, -1);
    lua_pop(vm, 1);
    return grounded;
}

const shim_game_state_t* shim_game_state(void)
{
    if (!celeste_cart)
    {
        return NULL;
    }

    int top = lua_gettop(vm);
    lua_getglobal(vm, "room");
    if (!lua_istable(vm, -1))
    {
        lua_settop(vm, top);
        return NULL;
    }
    lua_getfield(vm, -1, "x");
    int room_x = fix32_to_int((fix32_t)lua_tointeger(vm, -1));
    lua_pop(vm, 1);
    lua_getfield(vm, -1, "y");
    int room_y = fix32_to_int((fix32_t)lua_tointeger(vm, -1));
    lua_pop(vm, 1);
    lua_pop(vm, 1);

    int room = room_x % 8 + room_y * 8;
    int player = find_player();
    int alive = player != 0;
    float feet_y = NAN;
    int grounded = 0;
    if (alive)
    {
        feet_y = player_field(player, "y") + 8.0f;
        grounded = player_is_grounded(player);
    }

    game_state.room = room;
    game_state.alive = alive;
    game_state.feet_y = feet_y;
    game_state.grounded = grounded;
    game_state.spawn_feet_y = room >= 0 && room < 31 ? spawn_feet[room] : NAN;
    game_state.exit_feet_y = 4.0f;
    lua_getglobal(vm, "deaths");
    game_state.deaths = fix32_to_int((fix32_t)lua_tointeger(vm, -1));
    lua_pop(vm, 1);
    lua_settop(vm, top);
    return &game_state;
}

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
    renderer = NULL;
    celeste_cart = 0;

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

    celeste_cart = global_is_function("level_index") &&
                   global_is_function("load_room") &&
                   global_is_function("solid_at");
    if (celeste_cart)
    {
        find_spawn_feet();
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

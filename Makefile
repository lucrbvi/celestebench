UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
DYLIB := build/libopen8env.dylib
SDL3_LIB := deps/open8-build/_deps/sdl3-build/libSDL3.0.dylib
RPATH := -Wl,-rpath,@loader_path/../deps/open8-build/_deps/sdl3-build
else
DYLIB := build/libopen8env.so
SDL3_LIB := deps/open8-build/_deps/sdl3-build/libSDL3.so
RPATH := -Wl,-rpath,'$$ORIGIN/../deps/open8-build/_deps/sdl3-build'
endif

SDL3_INC := deps/open8-build/_deps/sdl3-src/include

CC      := cc
CFLAGS  := -O2 -std=c11 -fPIC -MMD -MP -Ideps/open8/src -I$(SDL3_INC)
LDFLAGS := -shared -Ldeps/open8-build/_deps/sdl3-build -lSDL3 -lm $(RPATH)

Z8LUA := $(filter-out %/lua.c %/ltests.c,$(wildcard deps/open8/src/z8lua/*.c))
SRC   := \
    csrc/shim.c \
    deps/open8/src/api.c \
    deps/open8/src/app.c \
    deps/open8/src/auxiliary.c \
    deps/open8/src/memory.c \
    deps/open8/src/p8scii.c \
    deps/open8/src/lexaloffle/p8_compress.c \
    deps/open8/src/lexaloffle/pxa_compress_snippets.c \
    $(Z8LUA)
OBJ := $(addprefix build/obj/,$(SRC:.c=.o))

.PHONY: all wasm site clean
all: $(DYLIB)

$(SDL3_LIB):
	cmake -S deps/open8 -B deps/open8-build -DCMAKE_BUILD_TYPE=Release
	cmake --build deps/open8-build --target SDL3

$(DYLIB): $(OBJ) $(SDL3_LIB)
	@mkdir -p $(@D)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

build/obj/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

build/obj/csrc/shim.o: deps/open8/src/core.c
$(OBJ): $(SDL3_LIB)
-include $(OBJ:.o=.d)

# Emscripten build of the same shim, bypassing open8's CMake. SDL3 comes from
# Emscripten's own port. Usage: make wasm  (needs emsdk on PATH).
EMCC  ?= emcc
CART  := deps/open8/export/carts/1CELESTE.PNG
WEB   := build/web
EXPORTS := _shim_init,_shim_quit,_shim_load_cart,_shim_step,_shim_frame_ms,_shim_framebuffer,_shim_game_state,_malloc,_free

EMCFLAGS := -O2 -std=c11 -MMD -MP -Ideps/open8/src -sUSE_SDL=3 \
    -Wno-error=incompatible-pointer-types -Wno-error=int-conversion \
    -Wno-error=implicit-function-declaration -Wno-error=implicit-int
EMLDFLAGS := -sUSE_SDL=3 \
    -sEXPORTED_FUNCTIONS=$(EXPORTS) \
    -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,HEAP32,HEAPF32,stringToNewUTF8 \
    -sMODULARIZE=1 -sEXPORT_NAME=createOpen8 -sENVIRONMENT=web \
    -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=1048576 \
    --preload-file $(CART)@/1CELESTE.PNG \
    -o $(WEB)/open8.js

WASM_OBJ := $(addprefix build/wasm/,$(SRC:.c=.o))

wasm: $(WEB)/open8.js

$(WEB)/open8.js: $(WASM_OBJ) $(CART) Makefile
	@mkdir -p $(WEB)
	$(EMCC) $(WASM_OBJ) $(EMLDFLAGS)

build/wasm/%.o: %.c
	@mkdir -p $(@D)
	$(EMCC) $(EMCFLAGS) -c $< -o $@

build/wasm/csrc/shim.o: deps/open8/src/core.c
-include $(WASM_OBJ:.o=.d)

# Build the WASM player and export the public bundle into site/public.
# Then: cd site && npx wrangler dev
site: wasm
	uv run python -m web.publish

clean:
	rm -rf build

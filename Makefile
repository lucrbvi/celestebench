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
    csrc/audio.c \
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

.PHONY: all clean
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
build/obj/deps/open8/src/api.o: CFLAGS += -Dinit_api=open8_init_api
$(OBJ): $(SDL3_LIB)
-include $(OBJ:.o=.d)

clean:
	rm -rf build

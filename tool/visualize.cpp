#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

void visualize_sdl(float *inputs, int *labels, int total_data) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL_Init Error: %s\n", SDL_GetError());
        return;
    }

    SDL_Window *win = SDL_CreateWindow("Label Visualization",
                                       SDL_WINDOWPOS_CENTERED,
                                       SDL_WINDOWPOS_CENTERED,
                                       WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    if (!win) {
        printf("SDL_CreateWindow Error: %s\n", SDL_GetError());
        SDL_Quit();
        return;
    }

    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    if (!ren) {
        printf("SDL_CreateRenderer Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(win);
        SDL_Quit();
        return;
    }

    SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
    SDL_RenderClear(ren);

    for (int i = 0; i < total_data; ++i) {
        // 將 [-1,1] 轉換到視窗座標
        int x = (inputs[i * 2 + 0] + 1.0) * (WINDOW_WIDTH / 2);
        int y = (1.0 - (inputs[i * 2 + 1] + 1.0) / 2) * WINDOW_HEIGHT;

        if (labels[i] == 0)
            SDL_SetRenderDrawColor(ren, 255, 0, 0, 255);  // 類別 0：紅色
        else if (labels[i] == 1)
            SDL_SetRenderDrawColor(ren, 0, 255, 0, 255);  // 類別 1：綠色
        else if (labels[i] == 2)
            SDL_SetRenderDrawColor(ren, 0, 0, 255, 255);  // 類別 2：藍色
        else
            SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);    // 其他：黑色

        SDL_Rect rect = {x - 2, y - 2, 4, 4};  // 小正方形
        SDL_RenderFillRect(ren, &rect);
    }

    SDL_RenderPresent(ren);

    printf("Press any key to exit...\n");
    getchar();

    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
}

void show_image_sdl(float *image_data, int width, int height) {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow("Image", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                          width * 4, height * 4, SDL_WINDOW_SHOWN);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24,
                                             SDL_TEXTUREACCESS_STREAMING, width, height);

    uint8_t *rgb_buffer = (uint8_t *)malloc(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        rgb_buffer[i * 3 + 0] = (uint8_t)(image_data[i] * 255);         // R
        rgb_buffer[i * 3 + 1] = (uint8_t)(image_data[i + width * height] * 255); // G
        rgb_buffer[i * 3 + 2] = (uint8_t)(image_data[i + 2 * width * height] * 255); // B
    }

    SDL_UpdateTexture(texture, NULL, rgb_buffer, width * 3);
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);

    SDL_Delay(1000);  // 顯示 1 秒
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    free(rgb_buffer);
}
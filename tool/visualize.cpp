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

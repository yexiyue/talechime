/* SPDX-License-Identifier: GPL-3.0-or-later
 * Standalone eSpeak NG adapter for Misaki's TextToPhonemes protocol. */
#include <espeak-ng/speak_lib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    char text[258];
    if (argc != 2 || strncmp(argv[1], "--path=", 7) != 0 || argv[1][7] == '\0') return 2;
    if (!fgets(text, sizeof(text), stdin)) return 3;
    size_t len = strlen(text);
    if (len == 0 || text[len - 1] != '\n' || len > 256) return 3;
    text[--len] = '\0';
    if (!len || strchr(text, '\r')) return 3;
    if (espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, argv[1] + 7, espeakINITIALIZE_DONT_EXIT) < 0) return 4;
    if (espeak_SetVoiceByName("en-us") != EE_OK) return 5;
    const void *cursor = text;
    int first = 1;
    while (cursor != NULL) {
        const char *phonemes = espeak_TextToPhonemes(&cursor, 1, 0x02 | 0x80 | ('^' << 8));
        if (phonemes && *phonemes) {
            if (!first && putchar(' ') == EOF) return 6;
            if (fputs(phonemes, stdout) == EOF) return 6;
            first = 0;
        }
    }
    if (first || putchar('\n') == EOF || fflush(stdout) == EOF) return 6;
    espeak_Terminate();
    return 0;
}

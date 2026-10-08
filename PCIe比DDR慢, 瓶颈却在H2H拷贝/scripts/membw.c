// Single-core memory bandwidth breakdown: read, write (regular / non-temporal), copy, latency.
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define N (1UL << 30)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double best(double (*f)(char*, char*), char* a, char* b) { double m = 1e9; for (int i = 0; i < 3; i++) { double t = f(a, b); if (t < m) m = t; } return N / m / 1e9; }
static volatile uint64_t sink;
static double rd(char* a, char* b) { double t = now(); __m256i acc = _mm256_setzero_si256(); for (size_t i = 0; i < N; i += 32) acc = _mm256_xor_si256(acc, _mm256_load_si256((__m256i*)(a + i))); sink = _mm256_extract_epi64(acc, 0); return now() - t; }
static double wr(char* a, char* b) { double t = now(); __m256i v = _mm256_set1_epi8(3); for (size_t i = 0; i < N; i += 32) _mm256_store_si256((__m256i*)(a + i), v); return now() - t; }
static double wr_nt(char* a, char* b) { double t = now(); __m256i v = _mm256_set1_epi8(5); for (size_t i = 0; i < N; i += 32) _mm256_stream_si256((__m256i*)(a + i), v); _mm_sfence(); return now() - t; }
static double cp(char* a, char* b) { double t = now(); for (size_t i = 0; i < N; i += 32) _mm256_store_si256((__m256i*)(b + i), _mm256_load_si256((__m256i*)(a + i))); return now() - t; }
static double cp_nt(char* a, char* b) { double t = now(); for (size_t i = 0; i < N; i += 32) _mm256_stream_si256((__m256i*)(b + i), _mm256_load_si256((__m256i*)(a + i))); _mm_sfence(); return now() - t; }
static double cp_libc(char* a, char* b) { double t = now(); memcpy(b, a, N); return now() - t; }
int main(void) {
  char *a = aligned_alloc(64, N), *b = aligned_alloc(64, N);
  memset(a, 1, N); memset(b, 2, N);
  printf("read only            %6.1f GB/s\n", best(rd, a, b));
  printf("write, regular       %6.1f GB/s\n", best(wr, a, b));
  printf("write, non-temporal  %6.1f GB/s\n", best(wr_nt, a, b));
  printf("copy, regular store  %6.1f GB/s\n", best(cp, a, b));
  printf("copy, non-temporal   %6.1f GB/s\n", best(cp_nt, a, b));
  printf("copy, glibc memcpy   %6.1f GB/s\n", best(cp_libc, a, b));
  // latency: dependent pointer chase over random cache lines in 1 GiB
  size_t lines = N / 64; size_t* perm = malloc(lines * sizeof(size_t));
  for (size_t i = 0; i < lines; i++) perm[i] = i;
  srand(1); for (size_t i = lines - 1; i > 0; i--) { size_t j = ((size_t)rand() << 16 ^ rand()) % (i + 1); size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
  for (size_t i = 0; i < lines; i++) *(size_t*)(a + perm[i] * 64) = perm[(i + 1) % lines] * 64;
  size_t p = perm[0] * 64, hops = 20000000; double t = now();
  for (size_t i = 0; i < hops; i++) p = *(size_t*)(a + p);
  sink = p; printf("memory latency       %6.1f ns per dependent load\n", (now() - t) / hops * 1e9);
  return 0;
}

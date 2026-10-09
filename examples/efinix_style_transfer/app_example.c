/*
 * app_example.c -- bare-metal usage sketch of an NNgen accelerator on a RISC-V SoC
 * (e.g. Efinix Sapphire), with CPU-managed double buffering of input/output frames.
 *
 * Build check on a host:  gcc -Wall -Wextra -c -I<build_dir> app_example.c
 * Platform specifics (register window, DDR addresses, cache, IRQ wiring) are placeholders.
 */
#include <stdint.h>
#include <string.h>
#include "nngen_driver.h"
#include "style_net_nngen.h"     /* generated next to style_net.v */

/* ---- platform placeholders ---- */
#ifndef NNGEN_REG_BASE
#define NNGEN_REG_BASE   0xF8010000u   /* AXI-Lite window (e.g. APB3 -> AXI-Lite bridge) */
#endif
#ifndef NN_DDR_BASE
#define NN_DDR_BASE      0x01000000u   /* GLOBAL_OFFSET: start of the NN DDR region */
#endif
/* two input (RGBX, written by camera pre-processing) and two output frame buffers */
#define IN_BUF(i)   (NN_DDR_BASE + 0x00400000u + (uint32_t)(i) * 0x00100000u)
#define OUT_BUF(i)  (NN_DDR_BASE + 0x00800000u + (uint32_t)(i) * 0x00100000u)

static nngen_dev_t nn;
static volatile int nn_done;

/* call from the platform interrupt handler wired to userInterruptA */
void nn_irq_handler(void)
{
  uint32_t s = nngen_irq_status(&nn);
  nngen_irq_ack(&nn, s);
  nn_done = 1;
}

/* load the parameter image to DDR (here: memcpy through a CPU DDR window) */
static void load_params(const uint8_t *img, uint32_t size)
{
  uint32_t dst = NN_DDR_BASE + STYLE_NET_PARAMS_DEFAULT_OFFSET;
  memcpy((void *)(uintptr_t)dst, img, size);
  /* flush D-cache here if the CPU path is cached */
}

int nn_setup(const uint8_t *param_image)
{
  if (nngen_init(&nn, NNGEN_REG_BASE, NN_DDR_BASE, &style_net_model) != 0)
    return -1;                                   /* bitstream/header mismatch */
  load_params(param_image, STYLE_NET_PARAM_IMAGE_SIZE);
  nngen_irq_enable(&nn, 1);
  return 0;
}

/* run one frame: input buffer k -> output buffer k (blocking or IRQ-driven) */
int nn_run_frame(int k, int use_irq)
{
  nngen_set_region_addr(&nn, STYLE_NET_IN_ACT_REG, IN_BUF(k));
  nngen_set_region_addr(&nn, STYLE_NET_OUT_HEAD_CONV_REG, OUT_BUF(k));
  nn_done = 0;
  nngen_start(&nn);
  if (use_irq) {
    while (!nn_done) { /* wfi */ }
    return 0;
  }
  return nngen_wait(&nn, 0);
}

/*
 * Frame loop with double buffering: while the accelerator processes buffer k,
 * the camera pre-processor fills buffer k^1 and the display reads output k^1.
 */
void nn_loop(volatile uint32_t *preproc_done_buf, volatile uint32_t *display_buf)
{
  int k = 0;
  for (;;) {
    while (*preproc_done_buf != (uint32_t)k) { }  /* wait until input k is complete */
    nn_run_frame(k, 1);
    *display_buf = OUT_BUF(k);                    /* hand output k to the overlay/HDMI */
    k ^= 1;
  }
}

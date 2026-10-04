# nngen 加速器集成到 fpga_2026_efinix（Ti60F225）交接文档

> 读者：负责把 nngen 生成的 IP 集成进 `fpga_2026_efinix` 的人。
> nngen 侧代码：本仓库 `efinix-infra` 分支（基于 `mobilenet-support`）。参考交付物在
> `examples/efinix_style_transfer/ip_128x128/`。
> FPGA 工程信息取自 `Fuwaki/fpga_2026_efinix@1caea05`（2026-10-04），本文**没有修改** FPGA 仓库，
> 也**没有做** FPGA 侧仿真或上板。凡是涉及 FPGA 工程的内容都属于建议，以集成时的实际情况为准。
>
> 标注约定：**【实测】** RTL 仿真、生成器或脚本实际跑出的数字；**【估计】** 静态统计或通用综合器给出的近似值；
> **【建议】** 设计取舍；**【待测】** 只有 Efinity 或上板才能确认。

---

## 0. 一页结论

1. **交付物**是一个单文件 Verilog-2001 IP：顶层 `style_net`。它带三组接口：
   - 1 个 AXI4 master（`maxi_*`，16 bit 数据、32 bit 地址、无 ID）；
   - 1 个 AXI4-Lite slave（`saxi_*`，32 bit，用作寄存器口）；
   - 1 根电平中断 `irq`。

   配套文件：C 头文件 `style_net_nngen.h`、模型无关的裸机驱动 `nngen_driver.[ch]`、参数镜像 `style_net_params.bin/.h`（44,288 B）。
2. **参考模型**是一个编码器-解码器结构的“风格迁移”小网络（39,779 个参数）：
   - 输入：128×128 **RGBX int8**（`q = pixel >> 1`，X 通道填 0）。
   - 输出：128×128 RGB int8，内存中按 RGBX 排布（X 为填充）。
   - 【实测】RTL 仿真（Verilator）的输出与 `ng.eval` 逐字节一致。
   - 【实测】int8 与浮点结果的相关系数为 0.988–0.996。
3. **性能**：
   - 【实测】par=1、axi16 配置每帧 **25,538,410 周期**。在 100 MHz 下约 255 ms，即约 3.9 fps。这是理想存储器下的数字，实际 DDR 会更慢。
   - par=2 时为 7.29 M 周期（约 73 ms），par=4 时为 3.6 M 周期（约 36 ms）。但 par≥2 会超出 BRAM 预算（见第 6 节）。
4. **BRAM**：
   - 【估计】交付配置（axi16、`min_onchip_ram_capacity=1024`、par=1）约需 **70 个 RAM10K**。FPGA 工程现在剩余约 90 个。
   - 用 nngen 默认的 axi32 配置要 164 个，放不下。
5. **逻辑量是最大风险**：
   - 【估计】按 yosys 通用流程（LUT4）统计，32×32 par=1 的版本约需 **50.6k LUT4 和 27.8k FF**。128×128 的版本见第 6.2 节，与此接近。
   - FPGA 工程现在只剩约 28.9k XLR。yosys 的通用估计一般偏高，但**很可能放不下**。
   - **必须先用 Efinity 单独综合这个 IP，拿到实测 XLR 和 Fmax**（第 8 节第 1 步）。
6. **系统接法**【建议】：
   - 数据通路：nngen `maxi` → 16→128 位宽转换 → 在 `ddr3_top` 前新增的 N:1 AXI 互联（现在 DDR 控制器只有 frame_buffer 一个 master）。
   - 控制通路：`saxi` ← APB3（0xF8010000）→ AXI-Lite 桥。
   - 中断：`irq` → `userInterruptA`。
   - CPU 负责输入/输出双缓冲：每帧只需改写 2 个地址寄存器，再写一次 START。

---

## 1. 交付物与目录

```
examples/efinix_style_transfer/
├── style_net_nngen.py      # PyTorch -> ONNX -> nngen -> 量化 -> Verilog + 头文件 + 驱动 + 参数 (+ RTL 仿真)
├── style_net_3.pt          # 训练好的参考权重（脚本默认加载，保证可复现）
├── ramest_efinix.py        # Ti60 RAM10K 用量估算（按真双口形状）
├── make_release.sh         # 重新生成 ip_128x128/
├── app_example.c           # 裸机用法示例：初始化、双缓冲、中断
└── ip_128x128/             # 参考交付物（由 make_release.sh 生成，已提交）
    ├── style_net.v                 # IP，48,408 行 / 2.47 MB，顶层 style_net
    ├── style_net_nngen.h           # 寄存器表、内存映射、张量形状/scale、参数镜像布局
    ├── nngen_driver.h / .c         # 模型无关的驱动（rv32imc -Os 下 186 B text）
    ├── style_net_params.bin        # 参数镜像（44,288 B），原样写进 DDR
    ├── style_net_params.h          # 同一份镜像的 C 数组版本
    ├── style_net_nngen_report.txt  # nngen 原始报告（调度表、内存映射、寄存器映射）
    ├── summary.json                # 周期数、RAM 估计、精度、验证结果
    └── SHA256SUMS
```

【实测】`make_release.sh` 生成的结果是确定的：`style_net.v` 和 `style_net_params.bin` 的 sha256 与通过 RTL 仿真的那次构建（`build_128_a16_cap1k_sim`）完全相同。

---

## 2. IP 接口

### 2.1 端口

| 信号组 | 方向 | 宽度 | 说明 |
|---|---|---|---|
| `CLK` | in | 1 | 单时钟，AXI master、AXI-Lite 和内部逻辑都在这个时钟域 |
| `RESETN` | in | 1 | **低有效，同步复位**（在 `always @(posedge CLK)` 里采样）。复位期间时钟必须在跑，建议保持 ≥16 个周期 |
| `irq` | out | 1 | **电平、高有效**，寄存器输出。`irq = OR(ISR & IER)` |
| `maxi_aw*` / `maxi_w*` / `maxi_b*` | | | AXI4 写通道 |
| `maxi_ar*` / `maxi_r*` | | | AXI4 读通道 |
| `saxi_*` | | | AXI4-Lite slave：`awaddr/araddr` 32 bit，`wdata/rdata` 32 bit，另有 `awcache/awprot/arcache/arprot`（可接 0） |

### 2.2 AXI4 master（`maxi`）

| 项 | 值 | 对集成的影响 |
|---|---|---|
| 数据宽度 | **16 bit**（`maxi_datawidth=16`），`wstrb` 2 bit | DDR 控制器是 128 bit，需要 16→128 upsizer |
| 地址宽度 | 32 bit，字节地址 | |
| ID | **无** ID 信号 | 互联侧给这个口分配一个常量 ID，所有事务按顺序完成 |
| `awsize/arsize` | 常量 1（2 B/beat） | 窄于 DDR 位宽，由 upsizer 处理 |
| burst | INCR（`awburst=1`），`awlen` ≤ 255，即最多 **256 beat = 512 B** | 突发**不会跨 4 KB 边界**（veriloggen 会自动切分）。如果 DDR 侧要求更短的突发，就在互联或 upsizer 里切分 |
| outstanding | 写：最多 6 个未完成的 AW；读：请求 FIFO 深度 8 | 互联或 upsizer 要能正确处理多个 outstanding 事务（同一 ID，按序） |
| `bready` | 常量 1 | 内部会计数 outstanding 写，并在依赖写回数据的读之前等待写响应 |
| `awcache/arcache` | 常量 3；`awprot/arprot=0`、`awqos/arqos=0`、`awlock=0` | |
| `awuser/aruser` | 2 bit 常量 0 | 不接即可 |
| `bresp/rresp` | 输入但**被忽略** | 错误不会上报。调试时请在互联侧加错误计数 |

### 2.3 AXI4-Lite slave（`saxi`）

- 寄存器都是 32 bit 宽，按字对齐。
- 地址只用 `addr[7:2]` 译码（6 bit 字索引，共 256 B），高位被忽略，所以寄存器在任意 256 B 对齐的窗口内会重复出现。【建议】给它分配一个 4 KB 窗口。
- 支持 AW 先于 W，也支持 AW 和 W 同时到达。`bresp/rresp` 恒为 0（OKAY）。**不检查 `wstrb`**，只能做整 32 bit 写。
- 吞吐：每次访问要几个周期，对控制面来说足够。

### 2.4 中断

- ISR bit0：`BUSY` 的下降沿，即一次推理结束。ISR bit1：extern 中断，本模型不用。
- 使用流程：IER 写 1 打开 bit0；中断到来后读 ISR，再向 IAR 写入要清除的位。IAR 的 bit i 清除 ISR 的 bit i，写入后 IAR 自动归零。
- `irq` 是电平信号，直到 ISR 被清零之前都保持高电平。接 `userInterruptA` 时不需要再做脉冲展宽。

---

## 3. 寄存器表与驱动方法

以 `ip_128x128/style_net_nngen.h` 为准，下表是本模型的值。寄存器的数目和顺序由模型决定：每个 input、每个 output 各占一个地址寄存器。所以**换模型就必须换头文件**；驱动会用 `ADDRESS_AMOUNT` 校验头文件与比特流是否匹配。

| 偏移 | R/W | 名称 | 说明 |
|---|---|---|---|
| 0x00–0x0C | R | HEADER0–3 | 保留 |
| 0x10 | W/R | START | 写 1 启动。读回非 0 表示主 FSM 还没接受启动请求 |
| 0x14 | R | BUSY | 1 = 正在推理 |
| 0x18 | W | RESET | 软复位内部逻辑：先等 AXI master 空闲，再复位 2–3 个周期。**AXI-Lite 寄存器（地址寄存器等）不会被清掉** |
| 0x1C / 0x20 | R / W | EXTERN_SEND / RECV | 本模型不用 |
| 0x24 | R | ISR | bit0 = 推理结束 |
| 0x28 | RW | IER | 中断使能 |
| 0x2C | W | IAR | 写 1 清除对应的 ISR 位 |
| 0x30 | R | COUNT | 周期计数（`measurable_main_fsm`）：主 FSM 处于 `COUNT_STATE` 状态时，每 `COUNT_DIV+1` 个周期加 1 |
| 0x34 / 0x38 | W | COUNT_STATE / COUNT_DIV | 见上一行 |
| 0x7C | R | ADDRESS_AMOUNT | 默认内存映射的总大小。本模型为 **1,789,184**（0x1B4D00），用于校验 |
| 0x80 | RW | GLOBAL_OFFSET | 所有区域共用的 DDR 基址 |
| 0x84 | RW | TEMPORAL | 中间结果区，1,613,824 B |
| 0x88 | RW | OUT `/head/Conv` | 输出张量，65,536 B |
| 0x8C | RW | IN `act` | 输入张量，65,536 B |
| 0x90 | RW | PARAMS | 参数镜像，44,288 B |

**DMA 地址 = GLOBAL_OFFSET + 区域寄存器的值**（`verilog.py` 中的 `maxi.global_base_addr(offset_reg)`，已在 RTL 仿真中走过）。驱动的 `nngen_set_region_addr()` 接收 DDR 绝对地址，内部减去 GLOBAL_OFFSET 再写入，所以所有区域都必须位于 GLOBAL_OFFSET **之上**。

### 3.1 驱动调用顺序

```c
#include "nngen_driver.h"
#include "style_net_nngen.h"           // 必须在 nngen_driver.h 之后 include

nngen_dev_t nn;
nngen_init(&nn, 0xF8010000u /*AXI-Lite 窗口*/, 0x01000000u /*GLOBAL_OFFSET*/, &style_net_model);
//   = 写 RESET -> 等 BUSY=0 -> 写 GLOBAL_OFFSET -> 把每个地址寄存器写成默认偏移
//     -> 校验 ADDRESS_AMOUNT（返回 -1 表示头文件与比特流不匹配）-> 清 ISR
/* 把 style_net_params.bin 写到 0x01000000 + STYLE_NET_PARAMS_DEFAULT_OFFSET（只需一次） */
nngen_irq_enable(&nn, 1);

/* 每帧 */
nngen_set_region_addr(&nn, STYLE_NET_IN_ACT_REG,        in_buf[k]);   // DDR 绝对地址
nngen_set_region_addr(&nn, STYLE_NET_OUT_HEAD_CONV_REG, out_buf[k]);
nngen_start(&nn);              // 写 START=1，等待 START 读回 0
/* 中断：s = nngen_irq_status(&nn); nngen_irq_ack(&nn, s);   或轮询：nngen_wait(&nn, 0); */
```

`RESET → START → 等 BUSY → 读 ISR → 写 IAR` 这一序列与 RTL 仿真 testbench 完全相同。【待测】`nngen_init` 中的软复位在 RTL 里没有单独仿真过，只做过代码审查：RESET 寄存器会在主 FSM 回到初始状态时自动清零。

**注意：**

- 推理进行中（BUSY=1）**不要改写地址寄存器**。主 FSM 在每层开始时才读取这些寄存器，中途改写会导致后续层读写错误的地址。
- 一次推理会读参数、读输入、读写 temporal 区、写输出。temporal 区只给这一个 IP 使用，不能与其他缓冲重叠。
- CPU 的 D-cache：如果 CPU 通过带 cache 的路径访问 DDR，那么启动前要 flush 输入和参数，结束后要 invalidate 输出。Sapphire 现在配置为 `DDR=0`，不存在这个问题。

---

## 4. 内存映射与张量布局

### 4.1 默认内存映射（相对 GLOBAL_OFFSET，按 64 B 对齐）

| 区域 | 默认偏移 | 大小 | 说明 |
|---|---|---|---|
| OUT `/head/Conv` | 0x0000_0000 | 0x1_0000（64 KB） | 输出张量 |
| IN `act` | 0x0001_0000 | 0x1_0000（64 KB） | 输入张量 |
| PARAMS | 0x0002_0000 | 0xAD00（44,288 B） | 参数镜像 |
| TEMPORAL | 0x0002_AD00 | 0x18_A000（1,613,824 B） | 中间特征图，nngen 自己管理 |
| 合计 | | 0x1B_4D00（1,789,184 B） | = ADDRESS_AMOUNT |

每帧可以把输入和输出指向任意 DDR 地址（满足 64 B 对齐、并且在 GLOBAL_OFFSET 之上）。参数和 temporal 区通常保持默认偏移。

### 4.2 参数镜像

- 由 `ng.export_ndarray(objs, chunk_size=64)` 生成，`style_net_params.bin` 就是它的原样字节。**整块写入 PARAMS 区域即可，不需要任何转换。**
- 镜像内有 51 个张量，按 64 B 对齐依次排列：int8 权重是 OHWI 布局，int32 是 bias 和 scale。各张量的偏移列在头文件注释里，只用于调试，驱动不需要知道。
- 参数镜像和比特流必须来自**同一次生成**：量化 shift 量固化在 RTL 里，镜像里只放权重、bias 和 scale。

### 4.3 输入张量（RGBX）

- 形状 (1, 128, 128, 4)，NHWC，int8，行优先，无行填充。像素 (y, x) 的通道 c 位于字节偏移 `(y*128 + x)*4 + c`。
- c=0..2 依次是 **R、G、B**，c=3 必须为 **0**。
- 量化方式：`q = pixel_u8 >> 1`，取值 0..127（校准用的是 `round(pixel/2)` 再截到 127，与右移的差别 ≤ 1 LSB）。对应浮点输入 `pixel/256`，scale_factor = 128（头文件里的 `STYLE_NET_IN_ACT_SCALE_FACTOR`）。
- 按 32 bit 小端读，一个字就是 `0x00BBGGRR`。

### 4.4 输出张量

- 形状 (1, 128, 128, 3)，内存中对齐为 (1, 128, 128, **4**)。布局与输入相同，c=3 是**未定义的填充字节**，不要使用。
- 值是 int8（有符号）。浮点值 = `q / STYLE_NET_OUT_HEAD_CONV_SCALE_FACTOR`（本次构建为 59.7778），对应像素为 `pixel = clamp(round(q * 256 / 59.7778), 0, 255)`，约等于 `clamp((q*1097) >> 8, 0, 255)`。
- **scale_factor 在每次重新生成后都可能变化**（来自校准），硬件后处理请把它做成寄存器，或从头文件取值。

---

## 5. 推荐系统架构（fpga_2026_efinix）

```
 SC431HAI ─ CSI-2 RX ─ frame_buffer ──────────────┐ AXI 128b (M0, 最高优先级)
                     │                               ▼
                     └─(读出) debayer ─┬─ HDMI     ┌──────────────┐
                                        │           │ N:1 AXI 互联  │──► ddr3_top (唯一 slave, 128b, ID 4b)
                       预处理(裁剪+下采样+量化) ─────►│ M1 预处理写   │
                                                    │ M2 CPU/邮箱   │
  nngen maxi 16b ─► axi_adapter 16→128 ────────────►│ M3 nngen      │
                                                    └──────────────┘
  Sapphire APB3 0xF8010000 ─► 译码 ─► APB→AXI-Lite 桥 ─► nngen saxi
  nngen irq (| 预处理 done) ─────────────────────────► userInterruptA
```

| 部件 | 建议 | 说明 |
|---|---|---|
| AXI 互联 | 在 `ddr3_top` 前加 N:1 互联或仲裁器。frame_buffer 优先级最高，nngen 最低 | 【事实】`ddr3_top` 只有一个 128 bit AXI4 slave（ID 4 bit），现在被 frame_buffer 独占。可以直接用 Forencich `verilog-axi` 的 `axi_interconnect`/`axi_crossbar`（MIT 许可，Verilog-2001），也可以手写一个固定优先级仲裁器。AW/W/AR 分别仲裁，W 跟随 AW 的授权顺序，不交织 |
| 位宽转换 | `axi_adapter`（S 16 → M 128） | 【建议】同时在这里把突发长度限制在 ≤ 64 拍（128 bit），并把 nngen 的 ID 映射成常量，例如 `4'b1000` |
| 控制口 | APB3 `io_apbSlave_0`（0xF8010000，64 KB，现在悬空）→ 译码 → APB→AXI-Lite 桥 → `saxi` | 把 0xF801_0000–0xF801_0FFF 分给 nngen。桥要等 AXI-Lite 的 B 或 R 响应返回后再拉高 PREADY |
| 中断 | `irq` → `userInterruptA`（现在接 0） | 有多个中断源时，先相或再接入，并加一个 SYS 状态寄存器区分来源 |
| 时钟 | 首选 core_clk 100 MHz，与 DDR AXI 同域 | 【待测】Fmax。如果 100 MHz 收敛不了，就给 nngen 单独一个时钟（50–66 MHz），在 maxi 上加异步 AXI FIFO，在 saxi 上加 `axil_cdc`，irq 做 2 级同步 |
| 复位 | 同步低有效。接 DDR 校准完成与 SoC 复位“与”起来、再同步到 core_clk 的复位 | 软件也可以用 RESET 寄存器复位 |
| DDR 访问（参数、结果） | 第一版用 APB“DDR 邮箱”（单拍 AXI master），或开 Sapphire 的 `DDR=1` | 参数 44 KB 在开机时只写一次。如果输出直接由 HDMI 叠加模块从 DDR 读取，CPU 就不需要读输出 |

### 5.1 DDR 分区建议（GLOBAL_OFFSET = 0x0100_0000）

| 地址 | 大小 | 用途 |
|---|---|---|
| 0x0000_0000–0x00BD_DDFF | 11.9 MB | frame_buffer 3 帧（现状，`START_ADDR=0`） |
| 0x0100_0000–0x011B_4CFF | 1.7 MB | nngen 默认映射：OUT0 占位、IN 占位、PARAMS（+0x2_0000）、TEMPORAL（+0x2_AD00） |
| 0x0140_0000 / 0x0150_0000 | 各 64 KB | 输入 RGBX 双缓冲 IN0 / IN1（预处理写，nngen 读） |
| 0x0180_0000 / 0x0190_0000 | 各 64 KB | 输出 RGBX 双缓冲 OUT0 / OUT1（nngen 写，叠加或 HDMI 读） |

`app_example.c` 使用的就是这套地址。

### 5.2 CPU 管理的双缓冲

```
帧 n:   预处理写 IN[k^1]      nngen: IN[k] -> OUT[k]      显示读 OUT[k^1]
帧 n+1: 预处理写 IN[k]        nngen: IN[k^1] -> OUT[k^1]  显示读 OUT[k]
```

- 预处理每写完一帧就产生 done（中断或状态位），并给出 buf 号。nngen 空闲时，CPU 把 IN 和 OUT 寄存器指向对应的缓冲后写 START。nngen 还忙就丢弃这一帧（推理只有约 4 fps，比摄像头慢得多）。
- 收到 nngen 中断后，CPU 把 OUT[k] 交给显示：在叠加模块的影子寄存器里写入新地址，vsync 时生效，避免画面撕裂。
- 只有 IN 和 OUT 需要双缓冲。PARAMS 只读，TEMPORAL 只在单次推理内部使用，都不需要。

### 5.3 预处理要求（nngen 的输入契约）

1. 分辨率：从 1920×1080 中取一个 **1024×1024** 的 ROI，做 **8×8 盒式平均**得到 128×128。也可以取 1080×1080 后做非整数缩放，但那样更复杂。
2. 颜色：使用白平衡后的 RGB888，即 debayer 之后 `rgb_datax2` 上的数据，与屏幕显示的画面一致。
3. 量化：`q = avg_u8 >> 1`。这里**不需要乘法器**，因为 1/64 平均再右移 1 位，等于 8×8 的和右移 7 位。
4. 打包：每像素 4 B `{0, B, G, R}`（小端字）。一个 128 bit 字装 4 个像素；一行 512 B，可以用 1 个或多个 INCR 突发写进 DDR，地址按 64 B 对齐。
5. 每帧写完 64 KB 后翻转 buf 并产生 done。
6. 资源【估计】：行累加器 128×3×14 bit，约 1 个 RAM10K；外加异步 FIFO 1–2 个 RAM10K，逻辑约 1–2k XLR。

如果换模型，输入的 scale_factor、通道数和归一化方式都可能改变，以新头文件里的 `*_IN_*_SCALE_FACTOR`、`*_C` 为准。

---

## 6. Efinix 注意事项与资源估计

### 6.1 代码结构检查（对 32×32 和 128×128 生成的 Verilog 做的静态扫描）

| 检查项 | 结果 |
|---|---|
| 宽 RAM 字内按字节通道写（`mem[a][8*i+:8] <=`，这是 `engine/README.md` 记录的 Titanium 综合崩溃写法） | **0 处** |
| `initial`、`$readmem*`、SystemVerilog 关键字 | 无（默认配置下） |
| RAM 写法 | 全部是 8 bit（另有 2 个 32 bit）整字写、真双口、write-first（`mem[addr] <= wdata`，同拍 `rdata <= wdata`），属于可推断为 RAM10K 的写法 |
| 异步读存储器 | 2 个 137×8（AXI 请求 FIFO），会被实现为 FF 或 LUT（共 2,192 bit） |
| 乘法 | `multiplier_core_*`：10 个 8×8、1 个 8×10、**1 个 32×32**（可能要 4 个 DSP）。`madd_core_*`：11 个 8×8+c 或 8×10+c。合计约 26–30 个 DSP（剩余 151 个）。地址运算里的常数乘会被映射成逻辑（EFX-0666 警告），属于正常现象 |
| 复位 | 同步复位 |
| 模块名 | `style_net` 及 88 个子模块，子模块名是**通用名**：`madd_0`、`multiplier_core_3`、`ram_w8_l4096_id5_1`、`_maxi_read_req_fifo` 等。**同一工程里放两个 nngen IP，或与其他代码重名时会冲突**。必要时用 sed 给子模块统一加前缀 |
| `use_param_ram=True` 选项 | 会生成带 `initial` 初始化的宽 ROM（例如 420 bit × 8），会浪费 RAM10K。**Efinix 上不要开**（默认关闭） |

### 6.2 资源估计

**RAM10K**（脚本 `ramest_efinix.py`）：

- 按 Ti60 真双口形状计算：1024×8/10、2048×4/5、4096×2、8192×1；512×16/20 只有简单双口模式才有。
- 这是悲观估计：假定每个叶子 RAM 单独占块，不做合并。

| 配置（参考模型） | RAM10K | 周期数（理想存储器） | 100 MHz 下的时间 |
|---|---|---|---|
| 128×128，axi16，cap 1024，par1（**交付配置**） | **70** | 25,538,410 | 255 ms |
| 128×128，axi16，nngen 默认 cap 4096，par1 | 90 | — | — |
| 128×128，axi32，默认，par1 | 164 | 24,766,818 | 248 ms |
| 128×128，axi16，cap 1024，par2 | 119 | 7,289,082 | 73 ms |
| 128×128，axi32，par4 | 236 | 3,596,698 | 36 ms |
| 32×32，axi16，cap 1024，par1 | 66 | 1,596,602 | 16 ms |
| 32×32，axi32，par4 | 237 | 193,026 | 1.9 ms |

- FPGA 工程现在剩余约 90 个 RAM10K（166/256 已用）。另外，预处理需要 2–3 个，互联和 upsizer 的 FIFO 也要几个。所以**只有 par=1 的配置放得下**。
- 如果要 par≥2，就得压缩 frame_buffer 的 FIFO（各 1024×128 bit），或者换更大的器件。

**逻辑**：

- 【估计】方法：yosys 0.52 通用流程（`proc; opt; memory -nomap; techmap; abc -lut 4`），不展平层次，multiplier/madd 核作为黑盒。脚本见本节末尾。
- 32×32、par1、axi32：**50,582 个 LUT4、27,827 个 FF**、80 个 RAM，加 23 个乘法核（黑盒）。
- 128×128、par1、axi16：见第 6.4 节补充数字。
- 逻辑的主体不在 MAC 阵列，而在各层控制 FSM、地址生成和参数选择用的大量 32 bit 多路选择器。按 yosys 的门级统计：`$mux` 约 9.3 万 bit，`$eq` 约 2.2 万 bit，`$add` 约 1.7 万 bit。
- Efinity 的结果通常会比 yosys 通用 LUT4 统计低一些，但**即使打七折，约 35k XLR 也已超过剩余的 28.9k**。这必须在第 8 节第 1 步用 Efinity 实测。如果确实放不下，可选方案：
  1. 精简模型，减少不同形状和不同种类的层，逻辑量大致与“层数 × 算子种类”成正比；
  2. 压缩 FPGA 工程现有的逻辑；
  3. 换 Ti120 或 Ti180 器件；
  4. 在 nngen 侧做逻辑优化，例如把控制参数表放进 ROM 或 RAM、缩窄地址寄存器（这需要 nngen 开发，目前还没做）。

**Fmax**：【待测】仓库里 `engine/` 自研引擎实测只有 43.6 MHz，瓶颈是组合仲裁路径。nngen 的控制 FSM 也有长组合路径的风险。**不要假定能跑 100 MHz。**

### 6.3 Efinity 使用提示（摘自 `fpga_2026_efinix/engine/README.md` 的经验）

- `efx_run` 用 `--flist` 时必须同时加 `--root style_net`，否则不能锁定顶层。
- 本机的 `-f full` 因为 `ivl` ABI 问题不可用，请用 `-f compile`。
- `--pnr_opts=--sdc_file` 会被拒绝，SDC 要通过工程 xml 指定。
- 跨时钟域路径在现有 SDC 里属于 exclusive 组，不做时序检查。新增的 CDC 必须结构化实现（异步 FIFO 或握手），不能直接跨域。

### 6.4 补充数字

（128×128 axi16 的 yosys 结果见文末附录 A。）

---

## 7. 如何从模型重新生成 IP 和头文件

### 7.1 环境

- Python 3 + 本仓库（`pip install -e .`，或在仓库根目录下运行）、veriloggen、numpy、torch、onnx。
- **ONNX 导出必须使用旧版 exporter**：`torch.onnx.export(..., dynamo=False)`。新的 dynamo exporter 依赖 onnxscript，并且会生成 nngen 不支持的图结构。
- RTL 仿真需要 Verilator 5.x（或 iverilog，但要慢得多）。

### 7.2 参考模型

```sh
cd examples/efinix_style_transfer
./make_release.sh                      # 128x128，只生成，不仿真（约 30 s）
./make_release.sh --sim verilator      # 外加完整 RTL 仿真，并与 ng.eval 比对（约 2 min）
python style_net_nngen.py --size 32    # 32x32 快速版：生成 + RTL 仿真（约 40 s）
python style_net_nngen.py --size 128 --par 2 --sim none --outdir build_p2   # 其他配置
python style_net_nngen.py --size 128 --config maxi_datawidth=32 ...        # 任意 nngen config
```

每个输出目录都包含：`style_net.v`、`style_net_v1_0/`（IP-XACT 包）、头文件、驱动、参数、报告、`summary.json`（周期数、`verify: PASSED`、RAM 估计）。

### 7.3 自己的模型

```python
import nngen as ng
# 1) PyTorch -> ONNX（旧版 exporter，opset 11–17；输入 NCHW (1,4,H,W)，nngen 内部转成 NHWC）
torch.onnx.export(model, torch.zeros(1, 4, H, W), 'mynet.onnx', input_names=['act'],
                  output_names=['out'], opset_version=13, dynamo=False)
# 2) ONNX -> nngen
(outputs, placeholders, variables, constants, operators) = ng.from_onnx(
    'mynet.onnx', value_dtypes={},
    default_placeholder_dtype=ng.int8, default_variable_dtype=ng.int8,
    default_constant_dtype=ng.int8, default_operator_dtype=ng.int8,
    default_scale_dtype=ng.int32, default_bias_dtype=ng.int32)
# 3) 量化：校准数据必须与实际预处理一致（NHWC、RGBX、X=0、q = pixel>>1）
outputs['out'].quant_range_rate = 0.9          # 图像输出：使用约 90% 的 int8 范围
ng.quantize(outputs, {'act': 128}, input_generators={'act': calib_gen}, num_samples=4)
for op in operators.values():                  # 可选：并行度
    if isinstance(op, ng.conv2d): op.attribute(par_ich=1, par_och=1)
# 4) 生成：只调用一次 to_ipxact/to_verilog，并在目标目录内执行
config = {'maxi_datawidth': 16, 'offchipram_chunk_bytes': 64, 'min_onchip_ram_capacity': 1024}
ng.to_ipxact([outputs['out']], 'mynet', config=config)
# 5) 头文件 + 驱动 + 参数镜像：必须在 to_ipxact/to_verilog 之后调用（要用到已分配的地址）
ng.export_driver_files([outputs['out']], 'mynet', 'outdir', config=config, params_c_array=True)
```

细节可以照抄 `style_net_nngen.py`：其中有 RGBX 输入、校准、`ng.eval` 软件对比，以及按驱动顺序写的 RTL testbench。

已支持、并且在 Verilator 下测过的算子：

- Conv（含 stride 2 和 group/depthwise）
- Clip → relu6、ReLU
- Add（scaled_add）、Concat
- **Resize/Upsample（nearest，整数倍）**
- GlobalAveragePool、Gemm 等 nngen 原有算子

**不支持**：bilinear Resize、ConvTranspose。

调优要点：

- `maxi_datawidth=16` 加 `min_onchip_ram_capacity=1024`，是目前 Ti60 上最省 RAM10K 的组合。axi8 也能用（71 块），但 DMA 带宽会减半。
- `par_ich/par_och`（卷积）这类并行度每翻一倍，周期数大致减到 1/2–1/4，但 RAM 会明显增加，因为宽 RAM 无法使用真双口的窄形状。
- 输出的 `quant_range_rate`、校准集都会影响 int8 精度。生成后先看 `float vs int8 corrcoef`。参考模型是 0.99 左右；如果低于 0.95，就要检查量化。

---

## 8. 上板 / 验证检查清单（按顺序）

1. **IP 单独综合（必做，先于一切集成）**
   - 新建 Efinity 工程，器件 Ti60F225，顶层 `style_net`，文件只放 `ip_128x128/style_net.v`，约束 `create_clock -period 10 CLK`。
   - 记录 XLR、RAM10K、DSP、Fmax，并确认所有 `ram_w*` 都被推断为 RAM10K（看综合报告的 memory 段），没有展开成 FF。
   - 如果 XLR 超过约 28k，或 Fmax < 100 MHz，就回到第 6.2 节的方案，或决定使用独立时钟。
2. **视频通路基线**：现有 camera → DDR → HDMI 工程先上板跑通。README 写明还没上板验证过。
3. **互联**：在 `ddr3_top` 前插入 N:1 互联，先只接 frame_buffer 和一个调试 master（邮箱）。确认画面不受影响后，用调试 master 对 0x0100_0000 起的 NN 区做 memtest。
4. **控制面**：APB 译码 + APB→AXI-Lite 桥，先接 nngen 的 `saxi`。nngen 的 `maxi` 可以暂时接到一个 AXI RAM 或空 slave 上。CPU 读 `0x7C`，应读到 **1,789,184**（0x1B4D00）；读写 `0x80–0x90` 应能正确回读。
5. **接上 maxi**：nngen `maxi` → 16→128 adapter → 互联 M3。`irq` → `userInterruptA`。
6. **静态图推理（比特级验证）**：
   - 把 `style_net_params.bin` 写到 0x0102_0000。
   - 写入测试输入。可以用 `style_net_nngen.py` 的 `synth_images` 生成，也可以从 PC 上的图片按第 4.3 节的格式生成。
   - 运行 `nngen_init` → 设置 IN/OUT 地址 → `nngen_start` → 等中断或 BUSY。
   - 读出输出，与 PC 上 `ng.eval` 的结果做**逐字节**比较，忽略 X 通道。
   - 【建议】额外导出一份测试输入/期望输出对，可以在 `style_net_nngen.py` 的 `vact/vout` 处保存为 .bin。
7. **性能**：在 START 和中断之间用 CPU 的 mcycle 计时，并与 RTL 的 25.54 M 周期对比，比值就是 DDR 带来的开销。也可以用 `COUNT_STATE/COUNT_DIV/COUNT` 测某个主 FSM 状态（即某一层）的周期数，用来找瓶颈层。
8. **实时预处理**：加入第 5.3 节的预处理写口。先单次抓一帧张量导出，与软件预处理对比（误差 ≤ 1 LSB）。再进入连续模式，按第 5.2 节做双缓冲。
9. **显示**：叠加模块从 OUT[k] 读 128×128 RGBX，按第 4.4 节换算成像素，放大后贴到 HDMI 上。地址切换只在 vsync 时发生。
10. **压力测试**：视频和推理同时连续跑数小时，确认 HDMI 读不断流（建议在互联里统计 frame_buffer 的等待周期）、推理结果稳定、没有 AXI 错误（统计 bresp/rresp）。

---

## 9. 已知限制与待办

- Efinity 在这台开发机上不可用（需要 Efinix 账号），所以**没有做过 Efinity 综合**，逻辑量和 Fmax 都还没确认。
- 交付配置（par=1）只有约 3.9 fps（128×128，100 MHz，理想存储器）。par=2 约 14 fps，但需要 119 个 RAM10K，超出预算。
- `maxi` 会忽略 `bresp/rresp`；没有 AXI ID；没有可配置的最大突发长度。
- 子模块名是通用名，有重名风险（见第 6.1 节）。
- 软复位路径没有单独仿真过。
- 头文件和比特流靠 `ADDRESS_AMOUNT` 做弱校验。HEADER0–3 寄存器目前没有填入模型哈希。

---

## 附录 A：yosys 通用综合估计

脚本（`hierarchy -top style_net` 之后把 `multiplier_core_*`、`madd_core_*` 设为黑盒）：

```
read_verilog style_net.v
hierarchy -top style_net
blackbox multiplier_core_* madd_core_*
proc; opt -fast; memory -nomap; opt -fast
techmap; opt -fast; abc -lut 4; opt_clean; stat
```

| 构建 | $lut (LUT4) | FF | $mem | 黑盒乘法核 | 运行时间 |
|---|---|---|---|---|---|
| 32×32 par1 axi32（默认 cap） | 50,582 | 27,827 | 80 | 23 | 11 min |

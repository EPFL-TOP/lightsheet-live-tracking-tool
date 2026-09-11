# Replacing the microscope PC

Findings from `tools/audit_windows_hardware.ps1`, run 2026-09-10, and
what they mean for specifying a replacement.

## The current machine

| | |
|---|---|
| Model | **HP Z840 Workstation** (serial CZC8297J3W) |
| CPU | 2 × Intel Xeon E5-2623 v3, 4 cores each |
| RAM | 128 GB |
| BIOS | M60 v01.65, **November 2015** |
| OS | **Windows 10 Enterprise 2015 LTSB, build 10240** |

The OS matters: **Windows 10 LTSB 2015 left support in October 2025**,
so this machine is already running an unsupported OS. That is a fair
reason to replace it, but see the driver risk below before ordering.

## The two cards that make this machine non-standard

### 1. `MicoIf (CZMI)` — the Zeiss card, and the important one

```
VEN_10EE  DEV_0303  SUBSYS_030210EE
vendor    Xilinx (FPGA)
driver    MicoIfDrv
location  PCI bus 132, device 0, function 0
```

**CZMI = Carl Zeiss Micro Imaging.** This is a Zeiss-proprietary
Xilinx FPGA card with a proprietary kernel driver (`MicoIfDrv`). It is
almost certainly the microscope's **real-time trigger/IO interface** —
the "Rt system" that MTB refers to. The very first MTB error we ever
saw on this machine was:

```
ConnectToRtSystem(): OpenRtNet("localhost", 1966, 5.0, "RtDll")
```

and MTB's configuration carries matching trigger wiring
(`SVB1_Camera1Ports`, `Camera 1 Trigger Out`,
`DG4 Strobe Colibri Increment`, and an `MTBRTComponents_IOCard.xml`).

This is the card to worry about. There is no retail equivalent and no
public driver.

### 2. `PVCam PCIe Camera` — the Prime 95B interface

```
VEN_1B6B  DEV_0001  SUBSYS_00011B6B  REV_04
vendor    Photometrics
driver    PMPCIEWDF
location  PCI bus 8, device 0, function 0
runtime   C:\Program Files\Photometrics\PVCam
```

The Prime 95B is the **PCIe** variant, so this host card is required —
a USB Prime would not need it.

### And a PCIe switch fabric worth identifying physically

```
PCI-to-PCI Bridge   VEN_10B5 DEV_8609 SUBSYS_03451954   bus 6, dev 0, fn 0
PCI-to-PCI Bridge   VEN_10B5 DEV_8609 SUBSYS_03451954   bus 7, dev 1, fn 0
Base System Device  VEN_10B5 DEV_8609 SUBSYS_860910B5   bus 6, dev 0, fn 1  [ERROR]
```

`VEN_10B5` is a PLX/Broadcom **PEX 8609** PCIe switch. The subsystem
vendor `0x1954` is **One Stop Systems**, who make PCIe expansion
hardware. The camera sits on bus 8, i.e. **behind both bridges**
(root → bus 6 → bus 7 → bus 8).

So either the Photometrics host card carries these switches, or there
is a separate PCIe expansion card in the chain. **The audit cannot
tell these apart — open the case and count the cards.** It changes
whether two or three boards have to move.

The `[ERROR]` entry is bus 6 / device 0 / **function 1**, the same
physical device as the working bridge at function 0. It is a second
function of that switch with no driver bound. The camera works, so it
is benign — but it is also why an earlier scan flagged an
"unidentified card" that turned out not to be one.

## Slots

`Win32_SystemSlot` reports WMI enum codes, not lane counts. Decoded
(`5`=x1, `7`=x4, `8`=x8, `10`=x16; usage `3`=free, `4`=in use):

| Slot | Width | State |
|---|---|---|
| 1 | x4 | in use |
| 2 | x16 | in use |
| 3 | x8 | free |
| 4 | x16 | in use |
| 5 | x8 | free |
| 6 | x16 | in use |
| 7 | x1 | free |

Four of seven occupied. Occupants are the Quadro M4000 (x16), the two
interface cards, and a 10 GbE NIC (`ixgbn`).

## Specifying the replacement

**Minimum:** a workstation chassis with at least **four usable PCIe
slots**, including one x16 for the GPU and at least two x8 for the
interface cards. Full-height, and deep enough for full-length boards —
measure the MicoIf card before ordering, as FPGA boards are often long.

**Licence dongle:** one Sentinel HL / HASP key on USB
(`VID_0529 PID_0001`, enumerated three ways). USB, so it moves with
the machine. ZEN and MTB licences may still be machine-tied — check
with Zeiss.

## The real risk, and it is not the hardware

**`MicoIfDrv` is a Zeiss proprietary driver for a 2015-era card, and
`MTB 2011` is 2018-era software.** Neither is guaranteed to have a
build for a current Windows. If Zeiss cannot supply a `MicoIfDrv` for
the OS you intend to install, **the new machine cannot drive the
microscope at all**, regardless of its specification.

Settle this **before** raising a purchase order:

1. Ask Zeiss whether `MicoIfDrv` and MTB 2011 (or a successor) are
   supported on your target Windows version. Quote the card's
   `VEN_10EE DEV_0303 SUBSYS_030210EE`.
2. Ask Teledyne/Photometrics the same for `PMPCIEWDF` with the PCIe
   Prime 95B interface (`VEN_1B6B DEV_0001`).
3. Ask Zeiss whether the ZEN 2 and MTB licences transfer, and what
   the lead time is for re-issuing.

If either driver is unavailable, the pragmatic options are to buy a
machine that runs the **same OS generation** (Windows 10 LTSC is
supported far longer than LTSB 2015), or to keep this PC as the
instrument controller and do analysis elsewhere.

## What this project needs from the new machine

Our stack is undemanding, so it does not constrain the choice:

- Python with `pythonnet`, `pymmcore-plus`, `torch` (CPU is fine)
- The MTB 2011 stack and PVCAM runtime
- No GPU requirement of our own — tracking offloads to the CoTracker
  server at `upoates-tethys.epfl.ch:8000`

The existing Quadro M4000 is for ZEN's display, not for us.

## Reproducing this audit

```powershell
powershell -ExecutionPolicy Bypass -File tools\audit_windows_hardware.ps1 > hardware_audit.txt
```

The script is deliberately **pure ASCII**: Windows PowerShell 5.1
reads a BOM-less `.ps1` as ANSI, so a single em dash once corrupted
the parse and produced a page of unrelated errors.
`tests/test_shell_scripts.py` guards against that.

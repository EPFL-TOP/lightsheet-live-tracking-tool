<#
    Hardware audit of the microscope PC.

    Purpose: identify the non-standard interface cards so a replacement
    machine can be specified with the right slots and the cards moved
    across. Run in an ELEVATED PowerShell and keep the output.

        powershell -ExecutionPolicy Bypass -File tools\audit_windows_hardware.ps1 > hardware_audit.txt

    What we already know about this machine, for context:
      - the Prime 95B reports as 'PMPCIECam00', i.e. a PCIe camera, so
        there is a Photometrics/Teledyne PCIe interface card
      - a 'Base System Device' PCI\VEN_10B5&DEV_8609 (PLX/Broadcom
        PEX 8609 PCIe switch) appeared in ERROR state; PLX bridges are
        common on frame-grabber cards, so this is probably a function
        of the camera card with no driver bound
      - the microscope itself is on USB (CAN29 over USB into CZCanSrv),
        NOT on a card
#>

# ASCII ONLY in this file. Windows PowerShell 5.1 reads a .ps1 with no
# byte-order mark as ANSI, so a UTF-8 multi-byte character arrives as
# several bytes of nonsense, terminates whatever string it sits in and
# cascades into a wall of unrelated parse errors. An em dash on one
# comment line was enough to break the whole script.

$ErrorActionPreference = "Continue"

function Section($title) {
    ""
    "=" * 78
    $title
    "=" * 78
}

# Vendor IDs worth recognising on sight. Everything else gets reported
# raw for lookup at https://pcilookup.com
$Vendors = @{
    "10B5" = "PLX Technology / Broadcom (PCIe switches, common on frame grabbers)"
    "8086" = "Intel"
    "1022" = "AMD"
    "10DE" = "NVIDIA"
    "1002" = "AMD/ATI"
    "1912" = "Renesas (USB 3 controllers)"
    "1B21" = "ASMedia"
    "104C" = "Texas Instruments (IEEE-1394 / FireWire cameras)"
    "12D8" = "Pericom"
    "1103" = "HighPoint"
    "11AB" = "Marvell"
    "14E4" = "Broadcom"
    "1077" = "QLogic"
    "1000" = "LSI / Broadcom"
    "1957" = "Freescale"
    "1409" = "Timedia/SUNIX (serial cards)"
    "135E" = "Sealevel (serial I/O)"
    "14D2" = "Titan/Oxford (serial cards)"
    "1592" = "Sitecom"
    "10EE" = "Xilinx (custom FPGA acquisition cards)"
    "1204" = "Lattice"
    "1FC9" = "Teledyne / e2v"
    "1AE8" = "Silicon Software (frame grabbers)"
    "1B66" = "Kaya Instruments (frame grabbers)"
    "1C2C" = "Euresys (frame grabbers)"
}

Section "1. MACHINE IDENTITY"
Get-CimInstance Win32_ComputerSystem |
    Select-Object Manufacturer, Model, SystemFamily, TotalPhysicalMemory,
                  NumberOfProcessors | Format-List
Get-CimInstance Win32_BIOS |
    Select-Object Manufacturer, SMBIOSBIOSVersion, ReleaseDate,
                  SerialNumber | Format-List
Get-CimInstance Win32_BaseBoard |
    Select-Object Manufacturer, Product, Version, SerialNumber |
    Format-List
Get-CimInstance Win32_Processor |
    Select-Object Name, NumberOfCores, NumberOfLogicalProcessors |
    Format-List
"OS:"
Get-CimInstance Win32_OperatingSystem |
    Select-Object Caption, Version, BuildNumber, OSArchitecture |
    Format-List

Section "2. PHYSICAL SLOTS  (what a replacement chassis must provide)"
$slots = Get-CimInstance Win32_SystemSlot
if ($slots) {
    $slots | Select-Object SlotDesignation, CurrentUsage, Status,
                           MaxDataWidth, Tag |
        Format-Table -AutoSize
} else {
    "Win32_SystemSlot returned nothing (common on OEM boards) -"
    "use the LocationInfo in section 3 and a physical inspection."
}

Section "3. ALL PCI / PCIe DEVICES  (the important section)"
"Devices are listed with vendor/device IDs. Match VEN_xxxx below."
""
$pci = Get-PnpDevice | Where-Object { $_.InstanceId -like "PCI\*" }
foreach ($d in ($pci | Sort-Object Class, FriendlyName)) {
    $id = $d.InstanceId
    $ven = if ($id -match "VEN_([0-9A-Fa-f]{4})") { $Matches[1].ToUpper() } else { "?" }
    $dev = if ($id -match "DEV_([0-9A-Fa-f]{4})") { $Matches[1].ToUpper() } else { "?" }
    $sub = if ($id -match "SUBSYS_([0-9A-Fa-f]{8})") { $Matches[1].ToUpper() } else { "" }
    $vendorName = if ($Vendors.ContainsKey($ven)) { $Vendors[$ven] } else { "" }

    $loc = ""
    $svc = ""
    try {
        $loc = (Get-PnpDeviceProperty -InstanceId $id `
                -KeyName "DEVPKEY_Device_LocationInfo" `
                -ErrorAction Stop).Data
    } catch {}
    try {
        $svc = (Get-PnpDeviceProperty -InstanceId $id `
                -KeyName "DEVPKEY_Device_Service" `
                -ErrorAction Stop).Data
    } catch {}

    $flag = ""
    if ($d.Status -ne "OK") { $flag = "   *** STATUS " + $d.Status }
    $subText = ""
    if ($sub) { $subText = " SUBSYS_" + $sub }

    "[" + $d.Status + "] " + $d.FriendlyName + $flag
    "        class    : " + $d.Class
    "        VEN_" + $ven + " DEV_" + $dev + $subText
    if ($vendorName) { "        vendor   : $vendorName" }
    if ($loc) { "        location : $loc" }
    if ($svc) { "        driver   : $svc" }
    ""
}

Section "4. DEVICES WITH PROBLEMS  (missing drivers = unidentified cards)"
Get-PnpDevice | Where-Object { $_.Status -ne "OK" } |
    Select-Object Status, Class, FriendlyName, InstanceId |
    Format-List

Section "5. THE CAMERA"
"Photometrics PCIe cameras present as PMPCIECam*; a USB one would not."
Get-PnpDevice | Where-Object {
    $_.FriendlyName -match "Photometrics|PVCAM|Prime|PMPCIE|Teledyne" -or
    $_.InstanceId  -match "PMPCIE"
} | Select-Object Status, Class, FriendlyName, InstanceId | Format-List
""
"PVCAM runtime:"
Get-ChildItem "C:\Windows\System32\pvcam*.dll" -ErrorAction SilentlyContinue |
    Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
Get-ChildItem "C:\Program Files\Photometrics",
              "C:\Program Files\Teledyne",
              "C:\ProgramData\Photometrics" -ErrorAction SilentlyContinue |
    Select-Object FullName | Format-Table -AutoSize

Section "6. SERIAL / USB PATHS  (how the microscope is attached)"
"The stand uses CAN29 over USB into CZCanSrv, so expect no card here."
Get-PnpDevice -Class Ports -ErrorAction SilentlyContinue |
    Select-Object Status, FriendlyName, InstanceId | Format-Table -AutoSize
""
"USB devices (excluding hubs and controllers):"
Get-PnpDevice -Class USB -ErrorAction SilentlyContinue |
    Where-Object { $_.FriendlyName -notmatch "Hub|Host Controller|Root" } |
    Select-Object Status, FriendlyName, InstanceId | Format-Table -AutoSize

Section "7. DRIVERS FOR NON-MICROSOFT HARDWARE"
"Third-party drivers must exist for the new OS before migrating."
Get-CimInstance Win32_PnPSignedDriver |
    Where-Object {
        $_.Manufacturer -and
        $_.Manufacturer -notmatch "Microsoft|Standard|\(Standard" -and
        $_.DeviceClass -match "SYSTEM|MEDIA|IMAGE|PORTS|USB|NET|HDC|SCSIADAPTER|UNKNOWN"
    } |
    Select-Object DeviceName, Manufacturer, DriverVersion, DriverDate,
                  InfName |
    Sort-Object Manufacturer, DeviceName | Format-Table -AutoSize

Section "8. ZEISS AND ACQUISITION SOFTWARE"
Get-CimInstance Win32_Service |
    Where-Object { $_.PathName -match "Zeiss|MTB|CZCan" } |
    Select-Object Name, DisplayName, State, StartMode, PathName |
    Format-List
""
Get-ChildItem "C:\Program Files\Carl Zeiss",
              "C:\Program Files (x86)\Common Files\Carl Zeiss" `
              -ErrorAction SilentlyContinue -Depth 0 |
    Select-Object FullName | Format-Table -AutoSize

Section "9. LICENCE DONGLES  (these must move with the software)"
Get-PnpDevice | Where-Object {
    $_.FriendlyName -match "Sentinel|HASP|Aladdin|CodeMeter|Dongle|WIBU"
} | Select-Object Status, Class, FriendlyName, InstanceId | Format-List

Section "SUMMARY FOR PROCUREMENT"
@"
Read section 3 and note, for every card that is NOT a chipset function:

  * VEN/DEV/SUBSYS  -> look up the exact model at https://pcilookup.com
  * location        -> which physical slot and its width
  * driver          -> the service name; confirm a build exists for the
                       OS you intend to install

Then specify the replacement with:
  * at least as many PCIe slots of at least those widths,
  * a chassis deep enough for full-length cards if any are,
  * and the same OS generation unless every vendor has a newer driver.

Cards seen on this machine so far:
  * a Photometrics/Teledyne PCIe interface for the Prime 95B
    (the camera enumerates as PMPCIECam00)
  * PCI\VEN_10B5&DEV_8609 - PLX/Broadcom PEX 8609 PCIe switch, which
    was in ERROR state. PLX switches are usually a bridge function on
    a frame-grabber card rather than a card in their own right, so
    check whether its 'location' matches the camera's.

Also worth confirming before ordering:
  * whether the camera card needs a specific slot (some frame grabbers
    require x4/x8 electrical, not just physical),
  * whether the Sentinel/HASP dongles in section 9 are USB (portable)
    or a card,
  * that ZEN and MTB are licensed for the new machine - a licence tied
    to hardware may need re-issuing by Zeiss.
"@

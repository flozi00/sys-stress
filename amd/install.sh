#!/usr/bin/env bash
# AMD GPU (ROCm) driver + Docker install for the gpu-stress test.
#
# Installs the AMDGPU DKMS kernel driver + ROCm userspace from the official
# Radeon repos, installs Docker (matching the NVIDIA install.sh flow), then
# builds the vendor-neutral Triton gpu-stress image for AMD GPUs.
#
# Tested on Ubuntu 24.04 (Noble) with an AMD Radeon AI PRO R9700 (RDNA 4,
# gfx1201).  The ROCm 6.4 image also covers Instinct MI300X/MI325 (gfx942)
# and other ROCm-supported Radeon and Instinct cards.
#
# Run as root:
#
#   sudo bash install.sh
set -euo pipefail

# Detect the Ubuntu codename so we register the right repo line. ROCm 6.4 ships
# packages for noble (24.04) and jammy (22.04); falling back to jammy on a noble
# host breaks apt.
. /etc/os-release
CODENAME="${VERSION_CODENAME:-noble}"

echo "=== Installing AMDGPU + ROCm for Ubuntu ${CODENAME} (${ID:-ubuntu}) ==="

# APT keyring directory recommended by the distribution maintainers.
mkdir --parents --mode=0755 /etc/apt/keyrings

# Download the Radeon signing key.
wget -q https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Register the kernel-mode (amdgpu) driver repo for the detected codename.
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.4/ubuntu ${CODENAME} main" \
    | tee /etc/apt/sources.list.d/amdgpu.list
apt-get update -y

# Register the ROCm userspace packages repo.
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4 ${CODENAME} main" \
    | tee /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | tee /etc/apt/preferences.d/rocm-pin-600
apt-get update -y

# Install the AMDGPU DKMS kernel driver + ROCm SMI.  amdgpu-dkms brings the
# up-to-date kernel module and the matching firmware blobs required by recent
# ASICs (e.g. RDNA 4 / gfx1201 or Instinct gfx942) that the stock
# linux-firmware may not yet include.  rocm-smi gives us the SMI command used
# by the stress test monitor.
apt-get install -y amdgpu-dkms rocm-smi

# Some AMD GPUs have no display pipeline, or the amdgpu display core (DCN)
# crashes with a divide-by-zero during dm_hw_init:
#   dcn401_get_memclk_states_from_smu ... divide error
# Disabling the display core (amdgpu.dc=0) avoids this on headless cards.
# Harmless on consumer cards that are also headless (no monitor attached).
echo "options amdgpu dc=0" > /etc/modprobe.d/amdgpu.conf
update-initramfs -u -k all

# --- Docker ---------------------------------------------------------------
# Remove conflicting packages, then install the official Docker CE repo.
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker \
           containerd runc; do
    apt-get remove -y "$pkg" || true
done

apt-get update -y
apt-get install -y ca-certificates curl

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  ${CODENAME} stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin \
                   docker-compose-plugin

systemctl enable --now docker

# --- Build the gpu-stress ROCm image --------------------------------------
# The same Triton-based stress test runs on AMD GPUs via the ROCm PyTorch
# image which already bundles a matching ROCm-enabled PyTorch + Triton.
cd "$(dirname "$0")/../gpu-stress"
docker build -f Dockerfile.rocm -t gpu-stress-rocm .

echo "=== AMD install complete ==="
echo "Reboot if the amdgpu driver was already loaded, then run:"
echo "  docker run --rm --device=/dev/kfd --device=/dev/dri \\"
echo "    --security-opt seccomp=unconfined --group-add video \\"
echo "    -v \"\$PWD/stress-logs:/workspace/gpu-stress/stress-logs\" \\"
echo "    gpu-stress-rocm --duration 120"
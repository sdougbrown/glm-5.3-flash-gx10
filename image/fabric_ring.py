"""NCCL and arx settings for a rank that mentat placed on a ring of boxes.

Runs at interpreter start (fabric_ring.pth) and does nothing unless mentat
gave this process MENTAT_FABRIC_LAYOUT=ring, which it does for each rank of a
TP=RING4 or TP=RING3 claim. On a ring each box cables one port to the
previous rank and the other to the next, so a rank sends to next and receives
from prev on different ports, and the diagonal ranks share no cable. This
sets:

- NCCL_IB_HCA to the two PCIe roots' functions of both ports, NCCL_ALGO=Ring
  so NCCL only talks to neighbours, and NCCL_GRAPH_FILE to a ring whose
  channels receive on the ports toward prev and send on the ports toward
  next. NCCL cannot infer that wiring, since it assumes every NIC reaches
  every peer.
- the GID from each port's own address, since each cable has its own subnet
  and no one GID index fits every device.
- VLLM_ARX_RING, ARX_RING_PREV_HCAS and ARX_RING_NEXT_HCAS for arx, unless
  FABRIC_RING_ARX=0.

NCCL reads its environment on first use, so this must run before vLLM
imports it.
"""
import os
import socket
import struct
from typing import Optional

# Where /sys is mounted; the tests point this at a fake tree.
SYS_ROOT = "/sys"


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _gid_ipv4(gid: str) -> Optional[str]:
    """The IPv4 behind a v4-mapped GID such as
    0000:0000:0000:0000:0000:ffff:0a63:0301, or None for a link-local slot.

    The driver widens the IPv4-mapped form to eight groups: five empty, the
    ffffff prefix, then the address in the last two groups.
    """
    groups = gid.split(":")
    if len(groups) != 8 or groups[5] != "ffff" or any(g != "0000" for g in groups[:5]):
        return None
    try:
        return socket.inet_ntoa(
            struct.pack(">HH", int(groups[-2], 16), int(groups[-1], 16)))
    except (ValueError, OSError):
        return None


def _device_ipv4s(dev: str, netdevs: set[str]) -> list[str]:
    """The IPv4 addresses of dev's RoCE v2 GIDs, one per netdev address.

    The GID table is filled from the netdevs' addresses, and ndevs names the
    netdev an entry came from, so a VLAN or other upper device's entries are
    skipped by requiring it to be one of this RDMA device's own netdevs. The
    address derived from the GID is the IPv4 configured on that netdev, so a
    device with no such GID sits on an unaddressed netdev.
    """
    out = []
    port = f"{SYS_ROOT}/class/infiniband/{dev}/ports/1"
    try:
        names = sorted(os.listdir(f"{port}/gids"), key=int)
    except OSError:
        return out
    for name in names:
        if _read(f"{port}/gid_attrs/types/{name}") != "RoCE v2":
            continue
        if _read(f"{port}/gid_attrs/ndevs/{name}") not in netdevs:
            continue
        addr = _gid_ipv4(_read(f"{port}/gids/{name}"))
        if addr is not None:
            out.append(addr)
    return out


def _port_devices(iface: str) -> list[str]:
    """RDMA devices behind the port that carries iface — one per PCIe root
    when both roots carry an address, only the first root's otherwise.

    Each ConnectX-7 port is two PCI functions on two roots with the same
    bus:device.function (0000:01:00.1 and 0002:01:00.1), and each function
    has its own RDMA device and netdev. A device counts only when one of its
    netdev's addresses shows up as a RoCE v2 GID: the device of an
    unaddressed netdev has no such GID, cannot open a QP to the neighbour,
    and is left out. Whatever is kept keeps the old order: sorted by PCI
    address, which is also the order libibverbs and NCCL list them in.
    """
    pci = os.path.basename(os.path.realpath(f"{SYS_ROOT}/class/net/{iface}/device"))
    bdf = pci.split(":", 1)[1]
    devs = []
    for fn in sorted(os.listdir(f"{SYS_ROOT}/bus/pci/devices")):
        if fn.split(":", 1)[1] != bdf:
            continue
        ib = f"{SYS_ROOT}/bus/pci/devices/{fn}/infiniband"
        if os.path.isdir(ib):
            devs += [(fn, d) for d in os.listdir(ib)]
    out = []
    for _, dev in sorted(devs):
        try:
            netdevs = set(os.listdir(f"{SYS_ROOT}/class/infiniband/{dev}/device/net"))
        except OSError:
            continue
        if _device_ipv4s(dev, netdevs):
            out.append(dev)
    return out


def _graph_xml(nccl_index: dict[str, int], prev: list[str], nxt: list[str], nchannels: int) -> str:
    """A ring graph for one GPU per node: channel c receives on prev[c % len(prev)]
    and sends on next[c % len(nxt)], alternating PCIe roots across channels
    when a port has two devices."""
    chans = "".join(
        f'<channel><net dev="{nccl_index[prev[c % len(prev)]]}"/><gpu dev="0"/>'
        f'<net dev="{nccl_index[nxt[c % len(nxt)]]}"/></channel>'
        for c in range(nchannels)
    )
    # The speeds and path types are what NCCL computes for its own ring on
    # these boxes. It picks protocols and chunk sizes from them, which set
    # the order its reductions sum in.
    return (
        '<graphs version="1">'
        f'<graph id="0" pattern="4" crossnic="1" nchannels="{nchannels}" speedintra="0.24" '
        'speedinter="0.24" latencyinter="0" typeintra="LOC" typeinter="P2C" samechannels="1">'
        f"{chans}</graph></graphs>\n"
    )


def _setup() -> None:
    if os.environ.get("MENTAT_FABRIC_LAYOUT") != "ring":
        return
    prev_iface = os.environ["MENTAT_FABRIC_PREV_IFACE"]
    nxt_iface = os.environ["MENTAT_FABRIC_NEXT_IFACE"]
    prev = _port_devices(prev_iface)
    nxt = _port_devices(nxt_iface)
    for side, iface, port_devs in (("prev", prev_iface, prev), ("next", nxt_iface, nxt)):
        if not port_devs:
            raise RuntimeError(
                f"fabric_ring: no RDMA device behind {iface} ({side} port) has a "
                "RoCE v2 GID for an IPv4 address. The port cabled to the "
                "neighbour needs its netdev's address up, or there is nothing "
                "this rank can send that neighbour over."
            )
    # On a switch both "ports" can be the same one; list each device once.
    devs = sorted(set(prev + nxt),
                  key=lambda d: os.path.realpath(f"{SYS_ROOT}/class/infiniband/{d}/device"))
    index = {d: i for i, d in enumerate(devs)}
    nchannels = int(os.environ.get("NCCL_MAX_NCHANNELS") or 8)
    path = f"/tmp/nccl-ring-graph.{os.getpid()}.xml"
    with open(path, "w") as f:
        f.write(_graph_xml(index, prev, nxt, nchannels))
    env = {
        "NCCL_IB_HCA": "=" + ",".join(devs),
        "NCCL_IB_MERGE_NICS": "0",
        "NCCL_CROSS_NIC": "1",
        "NCCL_IB_ADDR_FAMILY": "AF_INET",
        "NCCL_IB_ROCE_VERSION_NUM": "2",
    }
    if os.environ.get("FABRIC_RING_GRAPH", "1") != "0":
        env.update({"NCCL_ALGO": "Ring", "NCCL_GRAPH_FILE": path})
    else:
        # NCCL takes one device per channel for both directions of a ring graph,
        # so a graph cannot send and receive on different cables. Subnet-aware
        # routing instead opens each peer's queue pairs on the device in that
        # peer's subnet (FABRIC_RING_GRAPH=0; entrypoint: TP=RING3).
        env["NCCL_IB_SUBNET_AWARE_ROUTING"] = "1"
    if os.environ.get("FABRIC_RING_ARX", "1") != "0":
        env.update({
            "VLLM_ARX_RING": "1",
            "ARX_RING_PREV_HCAS": ",".join(prev),
            "ARX_RING_NEXT_HCAS": ",".join(nxt),
        })
    os.environ.update(env)
    os.environ.pop("NCCL_IB_GID_INDEX", None)


_setup()
# Subnet allocation for 10.20.0.0/16

Rule used: usable hosts = 2^(32 − prefix) − 2. For each requirement pick the **largest prefix
(smallest subnet)** whose usable-host count is at least the requirement. Allocate from the largest
requirement to the smallest, starting at 10.20.0.0, each subnet placed directly after the previous
one. Because the subnets are allocated largest first, every network address falls on its own
block boundary automatically.

## Table

| Subnet     | Hosts needed | Network address | Prefix | Usable host range          | Broadcast     | Usable hosts |
|------------|-------------:|-----------------|-------:|----------------------------|---------------|-------------:|
| app        | 1,000        | 10.20.0.0       | /22    | 10.20.0.1 – 10.20.3.254    | 10.20.3.255   | 1,022 |
| public     | 250          | 10.20.4.0       | /24    | 10.20.4.1 – 10.20.4.254    | 10.20.4.255   | 254   |
| data       | 100          | 10.20.5.0       | /25    | 10.20.5.1 – 10.20.5.126    | 10.20.5.127   | 126   |
| management | 20           | 10.20.5.128     | /27    | 10.20.5.129 – 10.20.5.158  | 10.20.5.159   | 30    |

Space used: 10.20.0.0 – 10.20.5.159. Free for later: 10.20.5.160 – 10.20.255.255.

## Working

**app (1,000 hosts).** /23 gives 2^9 − 2 = 510 < 1,000, which is too small. /22 gives 2^10 − 2 = 1,022 ≥ 1,000, so the prefix is **/22**
(block size 1,024 addresses = 4 × 256, so 4 third-octet values).
Network 10.20.0.0; the block covers 10.20.0.0 – 10.20.3.255.
First usable 10.20.0.1, last usable 10.20.3.254, broadcast 10.20.3.255.

**public (250 hosts).** /25 gives 126 < 250. /24 gives 2^8 − 2 = 254 ≥ 250, so the prefix is **/24** (block 256).
Next free address after app: 10.20.3.255 + 1 = **10.20.4.0**. Block 10.20.4.0 – 10.20.4.255.
Usable 10.20.4.1 – 10.20.4.254, broadcast 10.20.4.255.

**data (100 hosts).** /26 gives 62 < 100. /25 gives 2^7 − 2 = 126 ≥ 100, so the prefix is **/25** (block 128).
Next free: **10.20.5.0**. Block 10.20.5.0 – 10.20.5.127 (0 + 128 − 1 = 127).
Usable 10.20.5.1 – 10.20.5.126, broadcast 10.20.5.127.

**management (20 hosts).** /28 gives 14 < 20. /27 gives 2^5 − 2 = 30 ≥ 20, so the prefix is **/27** (block 32).
Next free: **10.20.5.128** (128 is a multiple of 32, so it is aligned). Block 10.20.5.128 – 10.20.5.159 (128 + 32 − 1).
Usable 10.20.5.129 – 10.20.5.158, broadcast 10.20.5.159.

The table was also checked with Python's `ipaddress` module (`ip_network('10.20.5.128/27')`, etc.).

## How these subnets are used by `nacl.json` (public subnet NACL)

| Rule purpose | CIDR used |
|---|---|
| SSH only from management | 10.20.5.128/27 |
| Forward to / replies from the app tier | 10.20.0.0/22 |
| Abusive range | 198.51.100.0/24 |

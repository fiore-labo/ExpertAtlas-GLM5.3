// From b12x (https://github.com/local-inference-lab/b12x): b12x/comm/roce/_roce_proxy.c at commit 1a7e3ec2, Apache-2.0,
// Copyright (c) 2025 by the b12x authors; modified by Fiore-lab: a GID index for each RDMA device (picked from the
// device's GID table when the caller passes a negative index), the build note below, and the share mode of a doorbell
// (top bit of its byte count: peer p gets share p of the payload, for the two-shot all-reduce; ABI 4), and the
// stripes of each peer over the HCAs that reach it (up to four HCAs; ABI 5, below), and a restart after an idle park
// (a failure on the way out kept, a failed context not started again, the thread id cleared when the thread ends).
//
// Stripes (ABI 5): the ranks exchange each HCA's IPv4 address and prefix, and a peer's stripes are the pairs of
// HCAs, one on each side, in the same subnet (matched in the lower rank's HCA order, each HCA of the higher rank
// used once). On a switch every HCA reaches every peer and the pairs are (0, 0), (1, 1), ... as before ABI 5. On
// three nodes cabled as a triangle each link is its own subnet and a peer gets only the HCAs of its link. Every
// peer must get the same number of stripes (the GPU waits for that many flags from each peer). An HCA whose
// prefix is unknown (0) matches any HCA.
//
// Ring of four (ABI 6, roce_set_ring4 before roce_connect; glm53/R3RT-102.md): ranks in ring order, rank r's
// neighbours r ^ 1 and r ^ 3 reached over exactly two stripes each, its opposite r ^ 2 over none (every HCA's
// address and prefix known, else the connect fails). Nothing is written to the opposite rank. Its payload reaches
// this rank through both neighbours instead: the sender's r ^ 1 forwards the sender's stripe 0, its r ^ 3 stripe 1,
// each into this rank's recv[sender][slot] at the same offset and then flag[sender][slot][stripe] on the same QP,
// so the GPU kernel waits for and reads the opposite rank's slot as any other. A neighbour forwards op N once it
// has posted its own op N (that gives N's byte count) and has seen the sender's stripe flag for N. The receiving
// rank completes N only after the relays arrived, and the sender writes that slot again (N + 2) only after the
// receiver sent N + 1, so a relay's source is read before it is overwritten. Share mode is not relayed.
//
// RDMA proxy for the b12x RoCE one-shot all-reduce.
//
// One rank owns one pinned host region laid out as:
//
//   recv[src][slot]  (world * SLOTS * slot_bytes)  filled by peers' RDMA writes
//   flag[src][slot][stripe] sequence number written after that stripe's payload,
//                         on the stripe's HCA
//   send[slot]       (SLOTS * slot_bytes)          staged by the local GPU kernel
//   ctrl             (FLAG_STRIDE)                 {u32 seq, u32 nbytes, u32 error,
//                                                   u32 missing_peer} doorbell; the
//                                                  last two are set by the kernel
//                                                  when a wait times out
//
// A byte count with its top bit set (ROCE_SHARES) splits the payload into world equal shares: peer p gets share p
// alone, written at the same offset of its receive slot (the reduce-scatter of a two-shot all-reduce).
//
// The GPU kernel stages its input into send[seq & 1], publishes nbytes and seq
// in ctrl, then spins on flag[peer][seq & 1][stripe] for every peer and stripe.
// The proxy thread
// below spins on ctrl.seq and, for every peer, stripes the payload across that
// peer's HCAs.  Each stripe is followed by its own 4-byte seq write on the same reliable
// QP, so its flag cannot become visible before its payload.  The GPU waits for
// every stripe flag before consuming the receive slot.  Nothing on the receive
// path involves the host.
//
// This file is built into a shared library with the host gcc and libibverbs and
// loaded by the engine; it must stay plain C with no CUDA dependency.

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <ifaddrs.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define ROCE_MAX_PEERS 16
#define ROCE_MAX_HCAS 4     /* changed from b12x (2): both ports' two PCIe halves */
#define ROCE_SLOTS 2
#define ROCE_FLAG_STRIDE 128
#define ROCE_PORT 1
#define ROCE_SEND_DEPTH 256
#define ROCE_ABI_VERSION 7  /* 6: the ring of four (roce_set_ring4, relays); 7: roce_pin, roce_tid */
#define ROCE_SHARES 0x80000000u
// changed from b12x (ABI 6): before a relay reads what a peer's NIC wrote (its stripe flag seen), order the device's
// writes against the other NIC's reads (outer shareable, loads and stores: stronger than rdma-core's oshld)
#if defined(__aarch64__)
#define ROCE_FROM_DEVICE_BARRIER() __asm__ volatile("dmb osh" ::: "memory")
#else
#define ROCE_FROM_DEVICE_BARRIER() __atomic_thread_fence(__ATOMIC_SEQ_CST)
#endif
// Model graphs leave sub-millisecond gaps between collectives.  Keep the
// proxy hot across those gaps; sleeping there adds one scheduler wakeup to
// every collective on the graph's critical path.
#define ROCE_IDLE_SPINS 20000000

typedef struct {
    uint64_t region_addr;
    uint32_t rkey[ROCE_MAX_HCAS];
    uint16_t lid[ROCE_MAX_HCAS];
    uint8_t gid[ROCE_MAX_HCAS][16];
    uint32_t mtu[ROCE_MAX_HCAS];
    uint32_t qp_num[ROCE_MAX_HCAS][ROCE_MAX_PEERS];
    uint32_t ipv4[ROCE_MAX_HCAS];   /* changed from b12x (ABI 5): the GID's IPv4 address, network order */
    uint8_t prefix[ROCE_MAX_HCAS];  /* its prefix length on the HCA's net device (0: unknown) */
    uint32_t n_hca;
} roce_blob_t;

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp[ROCE_MAX_PEERS];
    uint32_t outstanding[ROCE_MAX_PEERS];
    uint64_t writes_completed;
    uint64_t bytes_posted;
    union ibv_gid gid;
    uint16_t lid;
    int gid_index;          /* changed from b12x: per HCA (picked when the global index is < 0) */
    enum ibv_mtu mtu;
    uint32_t ipv4;          /* changed from b12x (ABI 5): the GID's IPv4 address (network order, 0: none) */
    uint8_t prefix;         /* and its prefix length (0: unknown) */
    int8_t gid_v2;          /* changed from b12x (ABI 6): the GID's type is RoCE v2 (1), another (0), unknown (-1) */
} roce_hca_t;

typedef struct {
    int world;
    int rank;
    int n_hca;
    int gid_index;
    roce_hca_t hca[ROCE_MAX_HCAS];
    uint8_t *region;
    size_t region_bytes;
    size_t slot_bytes;
    size_t recv_off;
    size_t flag_off;
    size_t send_off;
    size_t ctrl_off;
    int started;
    int n_stripe;                                      /* changed from b12x (ABI 5): stripes a peer (after connect) */
    int8_t stripe_hca[ROCE_MAX_PEERS][ROCE_MAX_HCAS];  /* the local HCA of each peer's stripe */
    uint64_t peer_addr[ROCE_MAX_PEERS];
    uint32_t peer_rkey[ROCE_MAX_HCAS][ROCE_MAX_PEERS]; /* by stripe */
    /* changed from b12x (ABI 6): the ring of four. Relay j forwards stripe j of relay_src[j]'s payload to
       relay_dst[j]; relay[slot] holds the op (seq, byte count) this rank posted in that slot and which relays are due */
    int ring4;
    int relay_src[2], relay_dst[2];
    struct {
        uint32_t seq;
        uint32_t nbytes;
        uint8_t due[2];
    } relay[ROCE_SLOTS];
    uint64_t relays_posted;
    pthread_t thread;
    atomic_int running;
    atomic_int failed;
    uint32_t last_seq;
    uint64_t ops_posted;
    uint64_t writes_completed;
    atomic_int tid; /* changed from b12x (ABI 7): the proxy thread's id, 0 before it runs */
    char err[512];
} roce_ctx_t;

void roce_destroy(roce_ctx_t *c);

static void set_err(roce_ctx_t *c, const char *what, int e) {
    snprintf(c->err, sizeof(c->err), "%s: %s", what, e ? strerror(e) : "failed");
}

int roce_abi_version(void) { return ROCE_ABI_VERSION; }

int roce_layout(int world, uint64_t slot_bytes, uint64_t *out) {
    // out = {recv_off, flag_off, send_off, ctrl_off, total_bytes, flag_stride, slots}
    if (world < 2 || world > ROCE_MAX_PEERS || slot_bytes == 0 || (slot_bytes % 4096) != 0) {
        return -1;
    }
    // Reject a layout whose arithmetic would wrap; the caller sizes slots from
    // configuration, so a wrapped region must fail here rather than at the NIC.
    uint64_t recv_bytes, flag_bytes, send_bytes, send_off, ctrl_off, total;
    if (slot_bytes > ((uint64_t)1 << 40) ||
        __builtin_mul_overflow((uint64_t)world * ROCE_SLOTS, slot_bytes, &recv_bytes) ||
        __builtin_mul_overflow(
            (uint64_t)world * ROCE_SLOTS * ROCE_MAX_HCAS,
            (uint64_t)ROCE_FLAG_STRIDE,
            &flag_bytes) ||
        __builtin_mul_overflow((uint64_t)ROCE_SLOTS, slot_bytes, &send_bytes) ||
        __builtin_add_overflow(recv_bytes, flag_bytes, &send_off) ||
        __builtin_add_overflow(send_off, send_bytes, &ctrl_off) ||
        __builtin_add_overflow(ctrl_off, (uint64_t)ROCE_FLAG_STRIDE, &total)) {
        return -1;
    }
    uint64_t recv_off = 0;
    uint64_t flag_off = recv_off + recv_bytes;
    out[0] = recv_off;
    out[1] = flag_off;
    out[2] = send_off;
    out[3] = ctrl_off;
    out[4] = total;
    out[5] = ROCE_FLAG_STRIDE;
    out[6] = ROCE_SLOTS;
    return 0;
}

uint64_t roce_blob_bytes(void) { return sizeof(roce_blob_t); }

static const uint8_t v4_mapped[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};

/* changed from b12x (ABI 5): "a.b.c.d/n" -> host-order network and mask; NULL or "" -> mask 0 (any) */
static int parse_range(const char *s, uint32_t *net, uint32_t *mask) {
    *net = 0;
    *mask = 0;
    if (s == NULL || *s == 0) return 0;
    char buf[32];
    const char *slash = strchr(s, '/');
    if (slash == NULL || (size_t)(slash - s) >= sizeof(buf)) return -1;
    memcpy(buf, s, (size_t)(slash - s));
    buf[slash - s] = 0;
    struct in_addr a;
    char *end = NULL;
    long n = strtol(slash + 1, &end, 10);
    if (inet_pton(AF_INET, buf, &a) != 1 || end == slash + 1 || *end != 0 || n < 1 || n > 32) return -1;
    *mask = n == 32 ? 0xffffffffu : ~(0xffffffffu >> n);
    *net = ntohl(a.s_addr) & *mask;
    return 0;
}

/* changed from b12x (ABI 5): the prefix length of `ipv4` (network order) on the net device of GID `index` of
   RDMA device `name` (sysfs gid_attrs/ndevs), 0 when it cannot be found */
static uint8_t gid_prefix(const char *name, int index, uint32_t ipv4) {
    char path[256], ndev[64] = {0};
    snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%d/gid_attrs/ndevs/%d", name, ROCE_PORT, index);
    FILE *f = fopen(path, "r");
    if (f == NULL) return 0;
    int got = fgets(ndev, sizeof(ndev), f) != NULL;
    fclose(f);
    if (!got) return 0;
    ndev[strcspn(ndev, "\n")] = 0;
    struct ifaddrs *list = NULL;
    if (getifaddrs(&list) != 0) return 0;
    uint8_t prefix = 0;
    for (struct ifaddrs *i = list; i != NULL; i = i->ifa_next) {
        if (i->ifa_addr == NULL || i->ifa_netmask == NULL || i->ifa_addr->sa_family != AF_INET) continue;
        if (strcmp(i->ifa_name, ndev) != 0) continue;
        if (((struct sockaddr_in *)i->ifa_addr)->sin_addr.s_addr != ipv4) continue;
        prefix = (uint8_t)__builtin_popcount(((struct sockaddr_in *)i->ifa_netmask)->sin_addr.s_addr);
        break;
    }
    freeifaddrs(list);
    return prefix;
}

/* changed from b12x (ABI 5): whether two HCAs are in the same subnet (an unknown prefix matches any) */
static int same_subnet(uint32_t a, uint8_t pa, uint32_t b, uint8_t pb) {
    if (pa == 0 || pb == 0) return 1;
    if (pa != pb) return 0;
    uint32_t m = pa >= 32 ? 0xffffffffu : ~(0xffffffffu >> pa);
    return (ntohl(a) & m) == (ntohl(b) & m);
}

/* changed from b12x (ABI 6): whether GID `index` of RDMA device `name` is RoCE v2 (sysfs gid_attrs/types): 1, 0, or
   -1 when the type cannot be read */
static int gid_type_v2(const char *name, int index) {
    char path[256], type[32] = {0};
    snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", name, ROCE_PORT, index);
    FILE *f = fopen(path, "r");
    if (f == NULL) return -1;
    int got = fgets(type, sizeof(type), f) != NULL;
    fclose(f);
    if (!got) return -1;
    type[strcspn(type, "\n")] = 0;
    return strcmp(type, "RoCE v2") == 0;
}

static int open_hca(roce_ctx_t *c, int h, const char *name) {
    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    if (list == NULL) {
        set_err(c, "ibv_get_device_list", errno);
        return -1;
    }
    struct ibv_device *dev = NULL;
    for (int i = 0; i < num; i++) {
        if (strcmp(ibv_get_device_name(list[i]), name) == 0) {
            dev = list[i];
            break;
        }
    }
    if (dev == NULL) {
        ibv_free_device_list(list);
        snprintf(c->err, sizeof(c->err), "RDMA device %s not found", name);
        return -1;
    }
    roce_hca_t *hca = &c->hca[h];
    hca->ctx = ibv_open_device(dev);
    ibv_free_device_list(list);
    if (hca->ctx == NULL) {
        set_err(c, "ibv_open_device", errno);
        return -1;
    }
    struct ibv_port_attr port;
    if (ibv_query_port(hca->ctx, ROCE_PORT, &port) != 0) {
        set_err(c, "ibv_query_port", errno);
        return -1;
    }
    if (port.state != IBV_PORT_ACTIVE) {
        snprintf(c->err, sizeof(c->err), "RDMA device %s port %d is not active", name, ROCE_PORT);
        return -1;
    }
    hca->lid = port.lid;
    hca->mtu = port.active_mtu;
    hca->gid_index = c->gid_index;
    if (hca->gid_index < 0) {               /* changed from b12x: the last IPv4-mapped (::ffff:a.b.c.d) GID = RoCE v2 */
        /* ABI 5: with TF_ROCE_ADDR_RANGE=a.b.c.d/n only an address in that range (a net device with several) */
        uint32_t range = 0, range_mask = 0;
        if (parse_range(getenv("TF_ROCE_ADDR_RANGE"), &range, &range_mask) != 0) {
            snprintf(c->err, sizeof(c->err), "TF_ROCE_ADDR_RANGE must be a.b.c.d/n");
            return -1;
        }
        int n_gid = port.gid_tbl_len > 0 && port.gid_tbl_len < 256 ? port.gid_tbl_len : 16;
        for (int i = 0; i < n_gid; i++) {
            union ibv_gid g;
            if (ibv_query_gid(hca->ctx, ROCE_PORT, i, &g) != 0) break;
            if (memcmp(g.raw, v4_mapped, 12) != 0) continue;
            if (gid_type_v2(name, i) == 0) continue;  /* changed from b12x (ABI 6): never a RoCE v1 entry */
            uint32_t a;
            memcpy(&a, g.raw + 12, 4);
            if (range_mask != 0 && (ntohl(a) & range_mask) != range) continue;
            hca->gid_index = i;
        }
        if (hca->gid_index < 0) {
            snprintf(c->err, sizeof(c->err), "RDMA device %s: no IPv4 RoCE GID%s", name,
                     range_mask != 0 ? " in TF_ROCE_ADDR_RANGE" : "");
            return -1;
        }
    }
    if (ibv_query_gid(hca->ctx, ROCE_PORT, hca->gid_index, &hca->gid) != 0) {
        set_err(c, "ibv_query_gid", errno);
        return -1;
    }
    if (memcmp(hca->gid.raw, v4_mapped, 12) == 0) {
        memcpy(&hca->ipv4, hca->gid.raw + 12, 4);
        hca->prefix = gid_prefix(name, hca->gid_index, hca->ipv4);
    }
    hca->gid_v2 = (int8_t)gid_type_v2(name, hca->gid_index);
    hca->pd = ibv_alloc_pd(hca->ctx);
    if (hca->pd == NULL) {
        set_err(c, "ibv_alloc_pd", errno);
        return -1;
    }
    hca->mr = ibv_reg_mr(hca->pd, c->region, c->region_bytes,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (hca->mr == NULL) {
        set_err(c, "ibv_reg_mr(pinned region)", errno);
        return -1;
    }
    hca->cq = ibv_create_cq(hca->ctx, ROCE_SEND_DEPTH * ROCE_MAX_PEERS, NULL, NULL, 0);
    if (hca->cq == NULL) {
        set_err(c, "ibv_create_cq", errno);
        return -1;
    }
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        struct ibv_qp_init_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.send_cq = hca->cq;
        attr.recv_cq = hca->cq;
        attr.qp_type = IBV_QPT_RC;
        attr.cap.max_send_wr = ROCE_SEND_DEPTH;
        attr.cap.max_recv_wr = 1;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        attr.cap.max_inline_data = 16;
        hca->qp[p] = ibv_create_qp(hca->pd, &attr);
        if (hca->qp[p] == NULL) {
            set_err(c, "ibv_create_qp", errno);
            return -1;
        }
        struct ibv_qp_attr init;
        memset(&init, 0, sizeof(init));
        init.qp_state = IBV_QPS_INIT;
        init.pkey_index = 0;
        init.port_num = ROCE_PORT;
        init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        int rc = ibv_modify_qp(hca->qp[p], &init,
                               IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (rc != 0) {
            set_err(c, "ibv_modify_qp(INIT)", rc);
            return -1;
        }
    }
    return 0;
}

roce_ctx_t *roce_create(int world, int rank, const char *const *hca_names, int n_hca,
                        int gid_index, void *region, uint64_t region_bytes,
                        uint64_t slot_bytes, char *err, uint64_t err_len) {
    uint64_t layout[7];
    if (roce_layout(world, slot_bytes, layout) != 0 || layout[4] > region_bytes ||
        rank < 0 || rank >= world || n_hca < 1 || n_hca > ROCE_MAX_HCAS) {
        snprintf(err, err_len, "invalid roce runtime geometry");
        return NULL;
    }
    roce_ctx_t *c = calloc(1, sizeof(*c));
    if (c == NULL) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    c->world = world;
    c->rank = rank;
    c->n_hca = n_hca;
    c->gid_index = gid_index;
    c->region = region;
    c->region_bytes = region_bytes;
    c->slot_bytes = slot_bytes;
    c->recv_off = layout[0];
    c->flag_off = layout[1];
    c->send_off = layout[2];
    c->ctrl_off = layout[3];
    for (int h = 0; h < n_hca; h++) {
        if (open_hca(c, h, hca_names[h]) != 0) {
            snprintf(err, err_len, "%s", c->err);
            roce_destroy(c);
            return NULL;
        }
    }
    return c;
}

int roce_local_blob(roce_ctx_t *c, void *out, uint64_t out_len) {
    if (out_len < sizeof(roce_blob_t)) {
        return -1;
    }
    roce_blob_t blob;
    memset(&blob, 0, sizeof(blob));
    blob.region_addr = (uint64_t)(uintptr_t)c->region;
    for (int h = 0; h < c->n_hca; h++) {
        blob.rkey[h] = c->hca[h].mr->rkey;
        blob.lid[h] = c->hca[h].lid;
        blob.mtu[h] = (uint32_t)c->hca[h].mtu;
        memcpy(blob.gid[h], c->hca[h].gid.raw, 16);
        for (int p = 0; p < c->world; p++) {
            blob.qp_num[h][p] = (p == c->rank) ? 0 : c->hca[h].qp[p]->qp_num;
        }
        blob.ipv4[h] = c->hca[h].ipv4;
        blob.prefix[h] = c->hca[h].prefix;
    }
    blob.n_hca = (uint32_t)c->n_hca;
    memcpy(out, &blob, sizeof(blob));
    return 0;
}

/* changed from b12x (ABI 5): local HCA h's QP for peer p to the peer's HCA hr */
static int connect_qp(roce_ctx_t *c, int h, int p, const roce_blob_t *peer, int hr) {
    roce_hca_t *hca = &c->hca[h];
    struct ibv_qp_attr rtr;
    memset(&rtr, 0, sizeof(rtr));
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = (enum ibv_mtu)(peer->mtu[hr] < (uint32_t)hca->mtu ? peer->mtu[hr] : (uint32_t)hca->mtu);
    rtr.dest_qp_num = peer->qp_num[hr][c->rank];
    rtr.rq_psn = 0;
    rtr.max_dest_rd_atomic = 1;
    rtr.min_rnr_timer = 12;
    rtr.ah_attr.is_global = 1;
    rtr.ah_attr.dlid = peer->lid[hr];
    rtr.ah_attr.sl = 0;
    rtr.ah_attr.src_path_bits = 0;
    rtr.ah_attr.port_num = ROCE_PORT;
    memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid[hr], 16);
    rtr.ah_attr.grh.sgid_index = (uint8_t)hca->gid_index;
    rtr.ah_attr.grh.hop_limit = 64;
    rtr.ah_attr.grh.traffic_class = 0;
    rtr.ah_attr.grh.flow_label = 0;
    int rc = ibv_modify_qp(hca->qp[p], &rtr,
                           IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                               IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc != 0) {
        set_err(c, "ibv_modify_qp(RTR)", rc);
        return -1;
    }
    struct ibv_qp_attr rts;
    memset(&rts, 0, sizeof(rts));
    rts.qp_state = IBV_QPS_RTS;
    rts.timeout = 14;
    rts.retry_cnt = 7;
    rts.rnr_retry = 7;
    rts.sq_psn = 0;
    rts.max_rd_atomic = 1;
    rc = ibv_modify_qp(hca->qp[p], &rts,
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                           IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc != 0) {
        set_err(c, "ibv_modify_qp(RTS)", rc);
        return -1;
    }
    return 0;
}

int roce_connect(roce_ctx_t *c, const void *blobs, uint64_t blobs_len) {
    if (blobs_len < sizeof(roce_blob_t) * (uint64_t)c->world) {
        snprintf(c->err, sizeof(c->err), "peer blob buffer too small");
        return -1;
    }
    const roce_blob_t *all = (const roce_blob_t *)blobs;
    // changed from b12x (ABI 6): a ring of four takes no guess: every HCA's address and prefix must be known (an
    // unknown prefix would match the opposite rank's HCAs and connect a pair no link joins)
    if (c->ring4) {
        for (int h = 0; h < c->n_hca; h++) {
            if (c->hca[h].gid_v2 != 1) {
                snprintf(c->err, sizeof(c->err), "ring of four: HCA %d's GID index %d is not a RoCE v2 entry (type %s)", h,
                         c->hca[h].gid_index, c->hca[h].gid_v2 == 0 ? "other" : "unreadable");
                return -1;
            }
        }
        for (int r = 0; r < c->world; r++) {
            for (int h = 0; h < (int)all[r].n_hca && h < ROCE_MAX_HCAS; h++) {
                if (all[r].ipv4[h] == 0 || all[r].prefix[h] == 0) {
                    snprintf(c->err, sizeof(c->err),
                             "ring of four: rank %d's HCA %d has no IPv4 address or prefix (net device not found?)", r, h);
                    return -1;
                }
            }
        }
    }
    // changed from b12x (ABI 5): each peer's stripes = the HCA pairs in one subnet, found the same way on both
    // sides: the lower rank's HCAs in order, each matched to the first unused HCA of the higher rank
    int8_t remote_hca[ROCE_MAX_PEERS][ROCE_MAX_HCAS];
    c->n_stripe = -1;
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        if (all[p].n_hca < 1 || all[p].n_hca > ROCE_MAX_HCAS) {
            snprintf(c->err, sizeof(c->err), "rank %d sent %u HCAs", p, all[p].n_hca);
            return -1;
        }
        const roce_blob_t *lo = c->rank < p ? &all[c->rank] : &all[p];
        const roce_blob_t *hi = c->rank < p ? &all[p] : &all[c->rank];
        int used[ROCE_MAX_HCAS] = {0};
        int n = 0;
        for (int a = 0; a < (int)lo->n_hca; a++) {
            for (int b = 0; b < (int)hi->n_hca; b++) {
                if (used[b] || !same_subnet(lo->ipv4[a], lo->prefix[a], hi->ipv4[b], hi->prefix[b])) {
                    continue;
                }
                used[b] = 1;
                c->stripe_hca[p][n] = (int8_t)(c->rank < p ? a : b);
                remote_hca[p][n] = (int8_t)(c->rank < p ? b : a);
                n++;
                break;
            }
        }
        if (c->ring4 && p == (c->rank ^ 2)) {  /* changed from b12x (ABI 6): the opposite rank, reached by relays */
            if (n != 0) {
                snprintf(c->err, sizeof(c->err),
                         "ring of four: rank %d (the opposite) shares a subnet with rank %d: not cabled as a ring in rank order",
                         p, c->rank);
                return -1;
            }
            for (int s = 0; s < ROCE_MAX_HCAS; s++) c->stripe_hca[p][s] = -1;
            continue;
        }
        if (n == 0) {
            snprintf(c->err, sizeof(c->err), "no HCA reaches rank %d (no HCA pair in one subnet%s)", p,
                     c->ring4 ? "; the ranks must be in ring order" : "");
            return -1;
        }
        if (c->ring4 && n != 2) {
            snprintf(c->err, sizeof(c->err), "ring of four: rank %d is reached over %d HCA pairs, a ring link has 2", p, n);
            return -1;
        }
        if (c->n_stripe >= 0 && n != c->n_stripe) {
            snprintf(c->err, sizeof(c->err),
                     "the peers are reached over different numbers of HCAs (%d and %d, rank %d)", c->n_stripe, n, p);
            return -1;
        }
        c->n_stripe = n;
    }
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        c->peer_addr[p] = all[p].region_addr;
        if (c->ring4 && p == (c->rank ^ 2)) {
            continue;  /* changed from b12x (ABI 6): no QP to the opposite rank leaves INIT */
        }
        for (int s = 0; s < c->n_stripe; s++) {
            const int hr = remote_hca[p][s];
            c->peer_rkey[s][p] = all[p].rkey[hr];
            if (connect_qp(c, c->stripe_hca[p][s], p, &all[p], hr) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

/* changed from b12x (ABI 6): this rank is one of a ring of four in ring order (before roce_connect) */
int roce_set_ring4(roce_ctx_t *c, int on) {
    if (on && c->world != 4) {
        snprintf(c->err, sizeof(c->err), "a ring of four needs four ranks, not %d", c->world);
        return -1;
    }
    c->ring4 = on != 0;
    c->relay_src[0] = c->rank ^ 1;  /* this rank is its r ^ 1: forwards its stripe 0 to its opposite, our r ^ 3 */
    c->relay_dst[0] = c->rank ^ 3;
    c->relay_src[1] = c->rank ^ 3;  /* this rank is its r ^ 3: forwards its stripe 1 to our r ^ 1 */
    c->relay_dst[1] = c->rank ^ 1;
    memset(c->relay, 0, sizeof(c->relay));
    return 0;
}

/* changed from b12x (ABI 5): the stripes a peer (after roce_connect), and the local HCA of one (-1: the opposite rank
   of a ring of four, reached by relays) */
int roce_stripes(roce_ctx_t *c) { return c->n_stripe; }

int roce_stripe_hca(roce_ctx_t *c, int peer, int stripe) {
    if (peer < 0 || peer >= c->world || peer == c->rank || stripe < 0 || stripe >= c->n_stripe) {
        return -1;
    }
    return c->stripe_hca[peer][stripe];
}

static int drain_cq(roce_ctx_t *c, int h) {
    struct ibv_wc wc[32];
    int n = ibv_poll_cq(c->hca[h].cq, 32, wc);
    if (n < 0) {
        set_err(c, "ibv_poll_cq", errno);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            snprintf(c->err, sizeof(c->err),
                     "RDMA write to rank %u failed: %s (vendor_err 0x%x, seq %u)",
                     (unsigned)wc[i].wr_id, ibv_wc_status_str(wc[i].status),
                     wc[i].vendor_err, c->last_seq);
            return -1;
        }
        c->hca[h].outstanding[wc[i].wr_id] -= 1;
        c->hca[h].writes_completed += 1;
        c->writes_completed += 1;
    }
    return 0;
}

static int post_op(roce_ctx_t *c, uint32_t seq, uint32_t nbytes) {
    // share mode: peer p gets bytes [p * share, (p + 1) * share) alone
    const int shares = (nbytes & ROCE_SHARES) != 0;
    nbytes &= ~ROCE_SHARES;
    if (nbytes == 0 || nbytes % 16 != 0 || (shares && nbytes % (16u * (uint32_t)c->world) != 0)) {
        snprintf(c->err, sizeof(c->err),
                 "RoCE payload size must be a positive multiple of 16 bytes (of 16 x world in share mode), got %u",
                 nbytes);
        return -1;
    }
    if (nbytes > c->slot_bytes) {
        snprintf(c->err, sizeof(c->err), "RoCE payload of %u bytes is larger than a slot (%zu)", nbytes, c->slot_bytes);
        return -1;
    }
    uint32_t slot = seq & 1u;
    if (c->ring4) {  /* changed from b12x (ABI 6) */
        if (shares) {
            snprintf(c->err, sizeof(c->err), "ring of four: share mode (the two-shot all-reduce) is not relayed (seq %u)", seq);
            return -1;
        }
        if (c->relay[slot].due[0] || c->relay[slot].due[1]) {
            snprintf(c->err, sizeof(c->err), "ring of four: seq %u's relays still due when seq %u rings", c->relay[slot].seq, seq);
            return -1;
        }
        c->relay[slot].seq = seq;
        c->relay[slot].nbytes = nbytes;
    }
    uint8_t *send = c->region + c->send_off + (size_t)slot * c->slot_bytes;
    uint32_t seq_copy = seq;
    uint32_t total_packs = shares ? nbytes / 16 / (uint32_t)c->world : nbytes / 16;
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank || (c->ring4 && p == (c->rank ^ 2))) {
            continue;  /* changed from b12x (ABI 6): the opposite rank gets this payload through the relays */
        }
        // the stripes cover [pack_offset, pack_offset + total_packs) at the same offset on both sides
        uint32_t pack_offset = shares ? total_packs * (uint32_t)p : 0;
        for (int st = 0; st < c->n_stripe; st++) {
            const int h = c->stripe_hca[p][st];
            roce_hca_t *hca = &c->hca[h];
            // Two work requests per stripe; keep the queue at most a quarter
            // full so a provider that needs extra entries can never fail a post.
            while (hca->outstanding[p] >= ROCE_SEND_DEPTH / 4) {
                if (drain_cq(c, h) != 0) {
                    return -1;
                }
                // A peer that stopped acknowledging keeps the QP retrying for
                // a long time; honour a stop request instead of blocking
                // roce_stop (and so teardown) behind it.
                if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
                    snprintf(c->err, sizeof(c->err),
                             "RoCE proxy stopped with %u writes outstanding to rank %d",
                             hca->outstanding[p], p);
                    return -1;
                }
            }
            uint32_t stripe_packs = total_packs / (uint32_t)c->n_stripe;
            if ((uint32_t)st < total_packs % (uint32_t)c->n_stripe) {
                stripe_packs += 1;
            }
            uint32_t stripe_bytes = stripe_packs * 16;
            uint64_t byte_offset = (uint64_t)pack_offset * 16;
            uint64_t remote = c->peer_addr[p];
            struct ibv_sge flag_sge = {
                .addr = (uint64_t)(uintptr_t)&seq_copy,
                .length = 4,
                .lkey = 0,
            };
            struct ibv_send_wr flag_wr;
            memset(&flag_wr, 0, sizeof(flag_wr));
            flag_wr.wr_id = (uint64_t)p;
            flag_wr.sg_list = &flag_sge;
            flag_wr.num_sge = 1;
            flag_wr.opcode = IBV_WR_RDMA_WRITE;
            flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            flag_wr.wr.rdma.remote_addr =
                remote + c->flag_off +
                (((uint64_t)c->rank * ROCE_SLOTS + slot) * (uint64_t)c->n_stripe +
                 (uint64_t)st) * ROCE_FLAG_STRIDE;
            flag_wr.wr.rdma.rkey = c->peer_rkey[st][p];

            struct ibv_send_wr data_wr;
            struct ibv_sge data_sge;
            struct ibv_send_wr *first_wr = &flag_wr;
            if (stripe_bytes != 0) {
                data_sge = (struct ibv_sge){
                    .addr = (uint64_t)(uintptr_t)(send + byte_offset),
                    .length = stripe_bytes,
                    .lkey = hca->mr->lkey,
                };
                memset(&data_wr, 0, sizeof(data_wr));
                data_wr.wr_id = (uint64_t)p;
                data_wr.next = &flag_wr;
                data_wr.sg_list = &data_sge;
                data_wr.num_sge = 1;
                data_wr.opcode = IBV_WR_RDMA_WRITE;
                data_wr.wr.rdma.remote_addr =
                    remote + c->recv_off +
                    ((uint64_t)c->rank * ROCE_SLOTS + slot) * c->slot_bytes +
                    byte_offset;
                data_wr.wr.rdma.rkey = c->peer_rkey[st][p];
                first_wr = &data_wr;
            }
            struct ibv_send_wr *bad = NULL;
            int rc = ibv_post_send(hca->qp[p], first_wr, &bad);
            if (rc != 0) {
                set_err(c, "ibv_post_send", rc);
                return -1;
            }
            hca->outstanding[p] += 1;
            hca->bytes_posted += stripe_bytes;
            pack_offset += stripe_packs;
        }
    }
    c->ops_posted += 1;
    if (c->ring4) {
        c->relay[slot].due[0] = 1;
        c->relay[slot].due[1] = 1;
    }
    for (int h = 0; h < c->n_hca; h++) {
        if (drain_cq(c, h) != 0) {
            return -1;
        }
    }
    return 0;
}

/* changed from b12x (ABI 6): the relays of a ring of four whose source stripe has arrived. Relay j forwards stripe j of
   relay_src[j]'s payload for op relay[slot].seq into relay_dst[j]'s recv[src][slot] at the same offset, then writes
   flag[src][slot][j] there on the same QP. A relay whose queue is full waits for the next call. Returns -1 on failure,
   else the number of relays still due. */
static int progress_relays(roce_ctx_t *c) {
    int due = 0;
    for (uint32_t slot = 0; slot < ROCE_SLOTS; slot++) {
        for (int j = 0; j < 2; j++) {
            if (!c->relay[slot].due[j]) {
                continue;
            }
            const uint32_t seq = c->relay[slot].seq;
            const int src = c->relay_src[j], dst = c->relay_dst[j];
            const uint64_t flag_index = ((uint64_t)src * ROCE_SLOTS + slot) * (uint64_t)c->n_stripe + (uint64_t)j;
            volatile uint32_t *arrived = (volatile uint32_t *)(c->region + c->flag_off + flag_index * ROCE_FLAG_STRIDE);
            if (__atomic_load_n(arrived, __ATOMIC_ACQUIRE) != seq) {
                due++;
                continue;
            }
            const int h = c->stripe_hca[dst][j];
            roce_hca_t *hca = &c->hca[h];
            if (hca->outstanding[dst] >= ROCE_SEND_DEPTH / 4) {
                if (drain_cq(c, h) != 0) {
                    return -1;
                }
                due++;
                continue;
            }
            ROCE_FROM_DEVICE_BARRIER();
            // stripe j's packs, split as the sender's post_op split them
            const uint32_t total_packs = c->relay[slot].nbytes / 16;
            const uint32_t s = (uint32_t)c->n_stripe;
            uint32_t pack_offset = 0;
            for (int st = 0; st < j; st++) {
                pack_offset += total_packs / s + ((uint32_t)st < total_packs % s ? 1 : 0);
            }
            const uint32_t stripe_bytes = (total_packs / s + ((uint32_t)j < total_packs % s ? 1 : 0)) * 16;
            const uint64_t byte_offset = (uint64_t)pack_offset * 16;
            const uint64_t slot_off = c->recv_off + ((uint64_t)src * ROCE_SLOTS + slot) * c->slot_bytes + byte_offset;
            uint32_t seq_copy = seq;
            struct ibv_sge flag_sge = {.addr = (uint64_t)(uintptr_t)&seq_copy, .length = 4, .lkey = 0};
            struct ibv_send_wr flag_wr;
            memset(&flag_wr, 0, sizeof(flag_wr));
            flag_wr.wr_id = (uint64_t)dst;
            flag_wr.sg_list = &flag_sge;
            flag_wr.num_sge = 1;
            flag_wr.opcode = IBV_WR_RDMA_WRITE;
            flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            flag_wr.wr.rdma.remote_addr = c->peer_addr[dst] + c->flag_off + flag_index * ROCE_FLAG_STRIDE;
            flag_wr.wr.rdma.rkey = c->peer_rkey[j][dst];
            struct ibv_send_wr data_wr;
            struct ibv_sge data_sge;
            struct ibv_send_wr *first_wr = &flag_wr;
            if (stripe_bytes != 0) {
                data_sge = (struct ibv_sge){.addr = (uint64_t)(uintptr_t)(c->region + slot_off), .length = stripe_bytes, .lkey = hca->mr->lkey};
                memset(&data_wr, 0, sizeof(data_wr));
                data_wr.wr_id = (uint64_t)dst;
                data_wr.next = &flag_wr;
                data_wr.sg_list = &data_sge;
                data_wr.num_sge = 1;
                data_wr.opcode = IBV_WR_RDMA_WRITE;
                data_wr.wr.rdma.remote_addr = c->peer_addr[dst] + slot_off;
                data_wr.wr.rdma.rkey = c->peer_rkey[j][dst];
                first_wr = &data_wr;
            }
            struct ibv_send_wr *bad = NULL;
            int rc = ibv_post_send(hca->qp[dst], first_wr, &bad);
            if (rc != 0) {
                set_err(c, "ibv_post_send (relay)", rc);
                return -1;
            }
            hca->outstanding[dst] += 1;
            hca->bytes_posted += stripe_bytes;
            c->relays_posted += 1;
            c->relay[slot].due[j] = 0;
        }
    }
    return due;
}

static void *proxy_run(roce_ctx_t *c) {
    atomic_store(&c->tid, (int)syscall(SYS_gettid));
    volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
    // Spin while ops are flowing.  After ROCE_IDLE_SPINS polls without a
    // doorbell, request a short nanosleep between polls (the OS decides the
    // actual delay) so an idle runtime does not hold a core next to the
    // serving process.  The missed-doorbell catch-up below keeps the protocol
    // correct however long the thread is away.
    uint64_t idle = 0;
    const struct timespec nap = {0, 20000};
    while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
        uint32_t seq = __atomic_load_n(&ctrl[0], __ATOMIC_ACQUIRE);
        // changed from b12x (ABI 6): a ring of four's relays go on whether or not a doorbell rang (the opposite rank
        // waits for them before it can ring its next one)
        int relays_due = 0;
        if (c->ring4) {
            relays_due = progress_relays(c);
            if (relays_due < 0) {
                atomic_store(&c->failed, 1);
                return NULL;
            }
        }
        if (seq == c->last_seq) {
            idle++;
            if (idle % 64 == 0) {
                for (int h = 0; h < c->n_hca; h++) {
                    if (drain_cq(c, h) != 0) {
                        atomic_store(&c->failed, 1);
                        return NULL;
                    }
                }
            }
            if (idle >= ROCE_IDLE_SPINS && relays_due == 0) {
                nanosleep(&nap, NULL);
            }
            continue;
        }
        idle = 0;
        // The doorbell holds only the newest sequence.  Our kernel for op N
        // completes on the peers' payloads alone, so op N+1 can ring before
        // this thread has seen op N (it slept, or the scheduler moved it).
        // Peers cannot get further than one op ahead of us, so at most
        // ROCE_SLOTS doorbells are pending and every send slot is intact:
        // post each missed sequence in order using its per-slot byte count.
        uint32_t pending = seq - c->last_seq;
        if (pending > ROCE_SLOTS) {
            snprintf(c->err, sizeof(c->err),
                     "doorbell skipped %u ops (last %u, now %u)", pending, c->last_seq, seq);
            atomic_store(&c->failed, 1);
            return NULL;
        }
        for (uint32_t s = c->last_seq + 1; pending > 0; s++, pending--) {
            uint32_t nbytes = ctrl[4 + (s & 1u)];
            if (post_op(c, s, nbytes) != 0) {
                atomic_store(&c->failed, 1);
                return NULL;
            }
            c->last_seq = s;
        }
    }
    // changed from b12x (ABI 6): stopped, a ring of four still posts the relays whose source arrived (for at most a
    // second): a neighbour may wait for this rank's last relay after this rank's own last op completed
    if (c->ring4) {
        struct timespec t0, t;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (;;) {
            int due = progress_relays(c);
            for (int h = 0; h < c->n_hca; h++) {
                if (drain_cq(c, h) != 0) due = -1;
            }
            // changed from b12x (idle park): a failure here, or relays still due after the second, is kept as the
            // context's failure (a parked rank checks it before it sleeps; roce_start refuses a failed context)
            if (due < 0) {
                if (!atomic_load(&c->failed)) snprintf(c->err, sizeof(c->err), "stopping: a relay or a completion failed");
                atomic_store(&c->failed, 1);
                break;
            }
            if (due == 0) break;
            clock_gettime(CLOCK_MONOTONIC, &t);
            if ((t.tv_sec - t0.tv_sec) * 1000000000L + (t.tv_nsec - t0.tv_nsec) > 1000000000L) {
                if (!atomic_load(&c->failed)) snprintf(c->err, sizeof(c->err), "stopping: %d relays still due after 1 s", due);
                atomic_store(&c->failed, 1);
                break;
            }
        }
    }
    return NULL;
}

// changed from b12x (idle park): the thread id back to 0 however the thread ends (the roce view reads it)
static void *proxy_main(void *arg) {
    roce_ctx_t *c = (roce_ctx_t *)arg;
    void *r = proxy_run(c);
    atomic_store(&c->tid, 0);
    return r;
}

int roce_start(roce_ctx_t *c) {
    if (atomic_load(&c->running)) {
        return 0;
    }
    if (!c->started) {
        // A restart continues from the last posted sequence so ops that rang
        // the doorbell while the thread was stopped are still posted.
        volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
        c->last_seq = ctrl[0];
        c->started = 1;
    }
    // changed from b12x (idle park): a failed context stays failed (its QPs, CQs and slots are not reset here), so
    // it is not started again; the failure was set to 0 here before
    if (atomic_load(&c->failed)) {
        return -1;
    }
    atomic_store(&c->running, 1);
    int rc = pthread_create(&c->thread, NULL, proxy_main, c);
    if (rc != 0) {
        atomic_store(&c->running, 0);
        set_err(c, "pthread_create", rc);
        return -1;
    }
    return 0;
}

void roce_stop(roce_ctx_t *c) {
    if (atomic_exchange(&c->running, 0)) {
        pthread_join(c->thread, NULL);
    }
}

// changed from b12x (ABI 7): the running proxy thread kept on one CPU (it spins while ops flow, so the scheduler's
// choice may be a fast core it heats). 0, or the error number; the thread goes on where it is on an error.
int roce_pin(roce_ctx_t *c, int cpu) {
    if (!atomic_load(&c->running)) {
        return EINVAL;
    }
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
        return EINVAL;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(c->thread, sizeof(set), &set);
}

// changed from b12x (ABI 7): the running proxy thread's id (0 before it runs), for /proc/self/task/<id>/stat
int roce_tid(roce_ctx_t *c) { return atomic_load(&c->tid); }

int roce_failed(roce_ctx_t *c) { return atomic_load(&c->failed); }

const char *roce_error(roce_ctx_t *c) { return c->err; }

uint64_t roce_stat(roce_ctx_t *c, int which) {
    switch (which) {
    case 0:
        return c->ops_posted;
    case 1:
        return c->writes_completed;
    case 2:
        return c->last_seq;
    case 3:  /* changed from b12x (ABI 6): relays posted (a ring of four) */
        return c->relays_posted;
    default:
        return 0;
    }
}

uint64_t roce_hca_stat(roce_ctx_t *c, int hca, int which) {
    if (hca < 0 || hca >= c->n_hca) {
        return -1;
    }
    switch (which) {
    case 0:
        return c->hca[hca].writes_completed;
    case 1:
        return c->hca[hca].bytes_posted;
    default:
        return 0;
    }
}

void roce_destroy(roce_ctx_t *c) {
    if (c == NULL) {
        return;
    }
    roce_stop(c);
    for (int h = 0; h < ROCE_MAX_HCAS; h++) {
        roce_hca_t *hca = &c->hca[h];
        for (int p = 0; p < ROCE_MAX_PEERS; p++) {
            if (hca->qp[p] != NULL) {
                ibv_destroy_qp(hca->qp[p]);
            }
        }
        if (hca->cq != NULL) {
            ibv_destroy_cq(hca->cq);
        }
        if (hca->mr != NULL) {
            ibv_dereg_mr(hca->mr);
        }
        if (hca->pd != NULL) {
            ibv_dealloc_pd(hca->pd);
        }
        if (hca->ctx != NULL) {
            ibv_close_device(hca->ctx);
        }
    }
    free(c);
}

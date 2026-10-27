# The release archive as an image: the archive's files in /opt/glm53 (the server, its library, tf-glm53, the control
# panel's launcher and page, the documents) and what the server needs at run time: cuBLAS 13, libibverbs with its
# providers, NCCL 2.30.7. Built in the unpacked archive:
#   docker build -f docker/Dockerfile -t glm53-tensorfold:VERSION .
# The NVIDIA driver comes from the host through the container toolkit (--gpus all); the weights and the expert banks
# stay on the host and are mounted read-only. tf-glm53 with RUNTIME=docker starts one container a rank (RUNTIME.md).
# Build in a freshly unpacked archive: the build refuses any file the archive's SHA256SUMS does not list (a glm53.conf,
# logs, a key file), so nothing of an installation goes into the image.
# Both stages start from one Ubuntu 24.04, pinned by its multi-arch index digest, so a rebuild starts from the same
# base. Move it deliberately (for Ubuntu's security updates) to the digest `docker buildx imagetools inspect
# ubuntu:24.04` gives, in both lines.
FROM ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c AS nccl
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip ca-certificates \
 && pip3 install --break-system-packages --no-deps --target /opt/nccl nvidia-nccl-cu13==2.30.7

FROM ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget libibverbs1 ibverbs-providers \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
 && apt-get update && apt-get install -y --no-install-recommends libcublas-13-0 \
 && apt-get purge -y wget && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
COPY --from=nccl /opt/nccl /opt/nccl
COPY . /opt/glm53
# exactly the files the archive lists, each as listed: no other file, none missing, none changed
RUN cd /opt/glm53 && sha256sum -c SHA256SUMS --quiet \
 && sed 's/^[0-9a-f]*  [*]\{0,1\}//; s|^\./||' SHA256SUMS | LC_ALL=C sort > /tmp/listed \
 && find . -type f ! -path ./SHA256SUMS | sed 's|^\./||' | LC_ALL=C sort > /tmp/present \
 && { diff /tmp/listed /tmp/present || { echo "files the archive does not list (build in a freshly unpacked archive)"; exit 1; }; } \
 && rm /tmp/listed /tmp/present
ENV TF_NCCL_LIB=/opt/nccl/nvidia/nccl/lib/libnccl.so.2 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENTRYPOINT ["/opt/glm53/docker/entry.sh"]
CMD ["help"]

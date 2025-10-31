import modal

# Default: https://hub.docker.com/layers/nvidia/cuda/12.6.3-devel-ubuntu24.04/images/sha256-badf6c452e8b1efea49d0bb956bef78adcf60e7f87ac77333208205f00ac9ade
# should be no greater than host CUDA version
# use cuda version 13 for all Blackwell features
cuda_version = "12.6.3"
# devel includes full CUDA toolkit
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


cuda_base_hopper_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    # .entrypoint([])  # remove verbose logging by base image on entry
    .pip_install(
        "ninja",
        "torch",
    )
)

// Modified by Fiore-lab from TensorFold 1.0.2 (NOTICE lists the changed files).
//! Backend-neutral engine parts: safetensors files and direct reads.
//! Fiore-lab: GLM-5.3 uses these two; TensorFold's checkpoint, draft-depth rule, copy index and ids JSON are gone.

pub const safetensors = @import("safetensors.zig");
pub const direct_io = @import("direct_io.zig");

test {
    _ = safetensors;
    _ = direct_io;
}

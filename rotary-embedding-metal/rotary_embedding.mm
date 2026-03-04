#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <torch/torch.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string>

// Include the auto-generated header with embedded metallib.
#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

void rotary_embedding(torch::Tensor &positions, torch::Tensor &query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor &cos_sin_cache, bool is_neox) {
  TORCH_CHECK(query.device().is_mps(), "query must be on MPS device");
  TORCH_CHECK(positions.device().is_mps(), "positions must be on MPS device");
  TORCH_CHECK(cos_sin_cache.device().is_mps(),
              "cos_sin_cache must be on MPS device");

  // Determine tensor dimensions.
  // positions: [num_tokens] or [batch, seq_len]
  // query:     [num_tokens, num_heads * head_size] or
  //            [num_tokens, num_heads, head_size]
  const int64_t num_tokens = positions.numel();

  // Flatten positions to 1D for kernel simplicity.
  torch::Tensor positions_flat = positions.reshape({-1});

  // Compute query/key strides along the token dimension.
  // Standard layout: [num_tokens, num_heads, head_size]
  // Batched layout: [batch, seq_len, num_heads, head_size]
  // The token dim is at index (positions.dim() - 1) in query after
  // accounting for batch dims, but we flatten positions so use stride(0)
  // relative to the flattened view.
  //
  // For standard [num_tokens, num_heads, head_size]:
  //   query_stride = num_heads * head_size (stride along dim 0)
  // For flattened [num_tokens, num_heads * head_size]:
  //   query_stride = num_heads * head_size (stride along dim 0)
  int64_t query_stride = query.stride(0);
  int64_t key_stride = key.has_value() ? key->stride(0) : 0;

  // Compute num_heads from tensor size. Works for both flat and split layouts.
  const int num_heads =
      static_cast<int>(query.numel() / (num_tokens * head_size));
  const int num_kv_heads =
      key.has_value()
          ? static_cast<int>(key->numel() / (num_tokens * head_size))
          : 0;

  const int rot_dim = cos_sin_cache.size(-1);
  const int embed_dim = rot_dim / 2;
  const int has_key = key.has_value() ? 1 : 0;

  @autoreleasepool {
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream, "Failed to get current MPS stream");

    id<MTLDevice> device = stream->device();

    // Load embedded Metal library.
    NSError *error = nil;
    id<MTLLibrary> lib =
        EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(lib, "Failed to create Metal library from embedded data",
                error ? [NSString stringWithFormat:@": %@",
                                                   error.localizedDescription]
                            .UTF8String
                      : "");

    // Select kernel variant based on dtype.
    NSString *kernName = nil;
    switch (query.scalar_type()) {
    case torch::kFloat:
      kernName = @"rotary_embedding_float";
      break;
    case torch::kHalf:
      kernName = @"rotary_embedding_half";
      break;
    case torch::kBFloat16:
      kernName = @"rotary_embedding_bfloat16_t";
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for rotary_embedding: ",
                  query.scalar_type());
    }

    // Set function constant for IS_NEOX.
    MTLFunctionConstantValues *constants =
        [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&is_neox type:MTLDataTypeBool atIndex:0];

    id<MTLFunction> fn = [lib newFunctionWithName:kernName
                                   constantValues:constants
                                            error:&error];
    TORCH_CHECK(fn, "Missing Metal kernel function: ", kernName.UTF8String,
                error ? [NSString stringWithFormat:@": %@",
                                                   error.localizedDescription]
                            .UTF8String
                      : "");

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&error];
    TORCH_CHECK(pso, "Failed to create pipeline state",
                error ? [NSString stringWithFormat:@": %@",
                                                   error.localizedDescription]
                            .UTF8String
                      : "");

    // Use stream->commandEncoder() to properly integrate with PyTorch's
    // MPS encoder lifecycle (kernel coalescing). Don't create encoders
    // directly on the command buffer or manually call endEncoding.
    dispatch_sync(stream->queue(), ^{
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      TORCH_CHECK(enc, "Failed to create compute encoder");

      [enc setComputePipelineState:pso];

      // Buffer 0: positions (flattened)
      [enc setBuffer:getMTLBufferStorage(positions_flat)
              offset:positions_flat.storage_offset() *
                     positions_flat.element_size()
             atIndex:0];

      // Buffer 1: query
      [enc setBuffer:getMTLBufferStorage(query)
              offset:query.storage_offset() * query.element_size()
             atIndex:1];

      // Buffer 2: key (or query as dummy if no key)
      if (key.has_value()) {
        [enc setBuffer:getMTLBufferStorage(*key)
                offset:key->storage_offset() * key->element_size()
               atIndex:2];
      } else {
        // Pass query buffer as dummy; has_key=0 ensures it's never accessed
        [enc setBuffer:getMTLBufferStorage(query)
                offset:query.storage_offset() * query.element_size()
               atIndex:2];
      }

      // Buffer 3: cos_sin_cache
      [enc setBuffer:getMTLBufferStorage(cos_sin_cache)
              offset:cos_sin_cache.storage_offset() *
                     cos_sin_cache.element_size()
             atIndex:3];

      // Scalar parameters via setBytes.
      const int32_t rot_dim_i32 = static_cast<int32_t>(rot_dim);
      [enc setBytes:&rot_dim_i32 length:sizeof(int32_t) atIndex:4];

      [enc setBytes:&query_stride length:sizeof(int64_t) atIndex:5];
      [enc setBytes:&key_stride length:sizeof(int64_t) atIndex:6];

      const int32_t head_size_i32 = static_cast<int32_t>(head_size);
      [enc setBytes:&head_size_i32 length:sizeof(int32_t) atIndex:7];

      const int32_t num_heads_i32 = static_cast<int32_t>(num_heads);
      [enc setBytes:&num_heads_i32 length:sizeof(int32_t) atIndex:8];

      const int32_t num_kv_heads_i32 = static_cast<int32_t>(num_kv_heads);
      [enc setBytes:&num_kv_heads_i32 length:sizeof(int32_t) atIndex:9];

      const int32_t has_key_i32 = static_cast<int32_t>(has_key);
      [enc setBytes:&has_key_i32 length:sizeof(int32_t) atIndex:10];

      // Dispatch: one threadgroup per token.
      const uint32_t threads_per_tg =
          std::min<uint32_t>(512, std::max(num_heads, num_kv_heads) * embed_dim);
      MTLSize grid = MTLSizeMake(num_tokens, 1, 1);
      MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);

      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    });

    stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
  }
}

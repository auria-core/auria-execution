// File: inference_pipeline.rs - Integration tests for auria-execution
// Tests the full inference pipeline from input to output
//
use auria_execution::*;
use auria_core::*;
use auria_router::*;
use auria_tensor::*;
use async_trait::async_trait;

#[tokio::test]
async fn test_execution_config() {
    let config = ExecutionConfig::default();
    
    assert_eq!(config.max_batch_size, 8);
    assert!(config.enable_caching);
    assert_eq!(config.cache_size, 10);
}

#[tokio::test]
async fn test_execution_engine_creation() {
    struct DummyBackend;
    
    #[async_trait]
    impl ExecutionBackend for DummyBackend {
        async fn execute_step(
            &self,
            input: Tensor,
            _experts: Vec<Tensor>,
            _state: ExecutionState,
        ) -> AuriaResult<ExecutionOutput> {
            Ok(ExecutionOutput {
                tokens: vec!["test".to_string()],
                usage: UsageStats { tokens_generated: 1 },
            })
        }
        
        fn backend_name(&self) -> &str { "dummy" }
        fn supported_tiers(&self) -> &[Tier] { &[Tier::Standard] }
    }
    
    let backend = DummyBackend;
    let engine = ExecutionEngine::new(backend);
    
    assert_eq!(engine.backend().backend_name(), "dummy");
}

#[tokio::test]
async fn test_token_embedding() {
    let embedding = TokenEmbedding::new(1000, 128);
    
    let token_emb = embedding.embed_token(42);
    assert_eq!(token_emb.len(), 128);
    
    let tokens = embedding.embed_tokens(&[1, 2, 3]);
    assert_eq!(tokens.len(), 128 * 3);
}

#[tokio::test]
async fn test_token_embedding_unembed() {
    let embedding = TokenEmbedding::new(1000, 64);
    
    let hidden = vec![0.1; 64];
    let unembedded = embedding.unembed(&hidden);
    
    assert_eq!(unembedded.len(), 64);
}

#[tokio::test]
async fn test_kv_cache_basic() {
    let mut cache = KvCache::new(10, 128);
    
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    
    cache.append(&[1.0; 128], &[2.0; 128]);
    
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
}

#[tokio::test]
async fn test_kv_cache_overflow() {
    let mut cache = KvCache::new(2, 128);
    
    cache.append(&[1.0; 128], &[2.0; 128]);
    cache.append(&[3.0; 128], &[4.0; 128]);
    cache.append(&[5.0; 128], &[6.0; 128]);
    
    assert!(cache.len() <= 2);
}

#[tokio::test]
async fn test_kv_cache_as_tensors() {
    let mut cache = KvCache::new(10, 128);
    
    cache.append(&[1.0; 128], &[2.0; 128]);
    
    let (keys, values) = cache.as_tensors();
    
    assert_eq!(keys.shape.len(), 2);
    assert_eq!(values.shape.len(), 2);
}

#[tokio::test]
async fn test_gating_network_basic() {
    let gating = GatingNetwork::new(8, 2);
    
    let hidden = vec![0.1; 128];
    let gates = gating.compute_gates(&hidden);
    
    assert!(!gates.is_empty());
    assert_eq!(gates.len(), 2);
}

#[tokio::test]
async fn test_gating_network_deterministic() {
    let gating = GatingNetwork::new(8, 2);
    
    let hidden = vec![0.5; 64];
    let gates1 = gating.compute_gates(&hidden);
    let gates2 = gating.compute_gates(&hidden);
    
    assert_eq!(gates1.len(), gates2.len());
    for (id1, w1) in &gates1 {
        for (id2, w2) in &gates2 {
            if id1 == id2 {
                assert!((w1 - w2).abs() < 0.001);
            }
        }
    }
}

#[tokio::test]
async fn test_tensor_conversion_f16_to_f32() {
    let f32_data = vec![1.0, 2.0, 3.0, 4.0];
    let f16_data = convert_fp32_to_fp16(&f32_data);
    let converted = convert_fp16_to_fp32(&f16_data).unwrap();
    
    assert_eq!(converted.len(), f32_data.len());
    assert!((converted[0] - 1.0).abs() < 0.1);
}

#[tokio::test]
async fn test_tensor_conversion_roundtrip() {
    let original = vec![0.0, 0.5, 1.0, -0.5, 2.0, -2.0];
    
    let f16 = convert_fp32_to_fp16(&original);
    let recovered = convert_fp16_to_fp32(&f16).unwrap();
    
    for (o, r) in original.iter().zip(recovered.iter()) {
        assert!((o - r).abs() < 0.5, "Original: {}, Recovered: {}", o, r);
    }
}

#[tokio::test]
async fn test_matmul_basic() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    
    let c = matmul(&a, 2, 2, &b, 2);
    
    assert_eq!(c.len(), 4);
    assert!((c[0] - 19.0).abs() < 0.01);
    assert!((c[1] - 22.0).abs() < 0.01);
    assert!((c[2] - 43.0).abs() < 0.01);
    assert!((c[3] - 50.0).abs() < 0.01);
}

#[tokio::test]
async fn test_matmul_non_square() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    
    let c = matmul(&a, 2, 3, &b, 3);
    
    assert_eq!(c.len(), 6);
}

#[tokio::test]
async fn test_relu_activation() {
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = relu(&data);
    
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.0);
    assert_eq!(result[4], 2.0);
}

#[tokio::test]
async fn test_gelu_activation() {
    let data = vec![0.0, 1.0, -1.0, 10.0];
    let result = gelu(&data);
    
    assert!(result[0] >= 0.0);
    assert!(result[1] > 0.0);
}

#[tokio::test]
async fn test_silu_activation() {
    let data = vec![0.0, 1.0, -1.0];
    let result = silu(&data);
    
    assert!(result.len() == 3);
}

#[tokio::test]
async fn test_softmax_basic() {
    let data = vec![1.0, 2.0, 3.0];
    let result = softmax(&data);
    
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
    
    assert!(result[2] > result[1]);
    assert!(result[1] > result[0]);
}

#[tokio::test]
async fn test_softmax_stability() {
    let data = vec![1000.0, 1001.0, 1002.0];
    let result = softmax(&data);
    
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
}

#[tokio::test]
async fn test_rms_norm() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = rms_norm(&data, 4, 1e-5);
    
    assert_eq!(result.len(), data.len());
}

#[tokio::test]
async fn test_layer_norm() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = layer_norm(&data, 4, 1e-5);
    
    assert_eq!(result.len(), data.len());
}

#[tokio::test]
async fn test_attention_config() {
    let config = AttentionConfig::default();
    
    assert_eq!(config.num_heads, 8);
    assert_eq!(config.head_dim, 64);
    assert_eq!(config.dropout, 0.0);
}

#[tokio::test]
async fn test_router_deterministic() {
    let router = DeterministicRouter::new(1024);
    
    let decision1 = router.route(Tier::Standard, 0);
    let decision2 = router.route(Tier::Standard, 0);
    
    assert_eq!(decision1.expert_ids.len(), decision2.expert_ids.len());
}

#[tokio::test]
async fn test_router_different_tiers() {
    let router = DeterministicRouter::new(1024);
    
    let nano = router.route(Tier::Nano, 0);
    let max = router.route(Tier::Max, 0);
    
    assert!(nano.expert_ids.len() <= max.expert_ids.len());
}

#[tokio::test]
async fn test_router_expert_count_scaling() {
    let router_small = DeterministicRouter::new(8);
    let router_large = DeterministicRouter::new(1024);
    
    let small = router_small.route(Tier::Standard, 0);
    let large = router_large.route(Tier::Standard, 0);
    
    assert!(small.expert_ids.len() <= large.expert_ids.len());
}

#[tokio::test]
async fn test_create_tensor() {
    let tensor = create_tensor(&[2, 3], 0.5);
    
    assert_eq!(tensor.shape, vec![2, 3]);
    assert!(!tensor.data.is_empty());
}

#[tokio::test]
async fn test_create_tensor_from_vec() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = create_tensor_from_vec(&[2, 2], data);
    
    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.data.len(), 16);
}

#[tokio::test]
async fn test_tensor_to_vec() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = create_tensor_from_vec(&[2, 2], data);
    
    let recovered = tensor_to_vec(&tensor).unwrap();
    
    assert_eq!(recovered.len(), 8);
}

#[tokio::test]
async fn test_full_pipeline_tensor_flow() {
    let data = vec![0.5; 64];
    let input = create_tensor_from_vec(&[1, 64], data);
    
    let f32_data = tensor_to_vec(&input).unwrap();
    assert_eq!(f32_data.len(), 128);
    
    let normalized = layer_norm(&f32_data, 128, 1e-5);
    assert_eq!(normalized.len(), 128);
    
    let activated = relu(&normalized);
    assert_eq!(activated.len(), 128);
    
    let output = softmax(&activated);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
}

#[tokio::test]
async fn test_moe_gating_integration() {
    let gating = GatingNetwork::new(4, 2);
    let embedding = TokenEmbedding::new(100, 64);
    
    let input = embedding.embed_token(50);
    let gates = gating.compute_gates(&input);
    
    assert_eq!(gates.len(), 2);
    
    let mut expert_outputs = Vec::new();
    for _ in 0..4 {
        expert_outputs.push(create_tensor_from_vec(&[1, 64], input.clone()));
    }
    
    let mut combined = vec![0.0f32; 64];
    for (idx, weight) in gates {
        if idx < expert_outputs.len() {
            let expert = tensor_to_vec(&expert_outputs[idx]).unwrap();
            for (i, v) in expert.iter().enumerate() {
                if i < 64 {
                    combined[i] += v * weight;
                }
            }
        }
    }
    
    assert_eq!(combined.len(), 64);
}

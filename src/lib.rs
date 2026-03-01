// File: lib.rs - This file is part of AURIA
// Copyright (c) 2026 AURIA Developers and Contributors
// Description:
//     Expert assembly and execution pipeline for AURIA Runtime Core.
//     Implements the core execution engine that orchestrates expert assembly
//     from shards and forward pass execution across different hardware backends.
//
use auria_core::{
    AuriaError, AuriaResult, ExecutionOutput, ExecutionState, ExpertId,
    RoutingDecision, Shard, ShardId, Tensor, TensorDType, Tier,
};
use auria_tensor::{
    activation::{gelu, relu, softmax},
    attention::{multihead_attention, AttentionConfig},
    convert::{convert_fp16_to_fp32, convert_fp32_to_fp16},
    matmul::matmul,
    normalization::rms_norm,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[async_trait]
pub trait ExecutionBackend: Send + Sync {
    async fn execute_step(
        &self,
        input: Tensor,
        experts: Vec<Tensor>,
        state: ExecutionState,
    ) -> AuriaResult<ExecutionOutput>;

    fn backend_name(&self) -> &str;
    fn supported_tiers(&self) -> &[Tier];
}

pub struct ExecutionEngine<B: ExecutionBackend> {
    backend: B,
}

impl<B: ExecutionBackend> ExecutionEngine<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub async fn execute(
        &self,
        input: Tensor,
        routing: RoutingDecision,
        state: ExecutionState,
    ) -> AuriaResult<ExecutionOutput> {
        self.backend.execute_step(input, Vec::new(), state).await
    }

    pub async fn execute_moe(
        &self,
        input: &Tensor,
        expert_outputs: &[Tensor],
        gating_weights: &[f32],
    ) -> AuriaResult<Tensor> {
        if expert_outputs.is_empty() {
            return Err(AuriaError::ExecutionError("No expert outputs for MoE".to_string()));
        }

        let input_f32 = tensor_to_f32(input)?;
        let mut weighted_sum = vec![0.0f32; input_f32.len()];

        for (expert_tensor, weight) in expert_outputs.iter().zip(gating_weights.iter()) {
            let expert_f32 = tensor_to_f32(expert_tensor)?;
            for (i, val) in expert_f32.iter().enumerate() {
                weighted_sum[i] += val * weight;
            }
        }

        tensor_from_f32(&weighted_sum, &input.shape)
    }
}

fn tensor_to_f32(tensor: &Tensor) -> AuriaResult<Vec<f32>> {
    match tensor.dtype {
        TensorDType::FP16 => convert_fp16_to_fp32(&tensor.data),
        _ => Err(AuriaError::ExecutionError("Unsupported dtype".to_string())),
    }
}

fn tensor_from_f32(data: &[f32], shape: &[u32]) -> AuriaResult<Tensor> {
    let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
    Ok(Tensor {
        data: bytes,
        shape: shape.to_vec(),
        dtype: TensorDType::FP16,
    })
}

pub struct ExpertAssembler<S> {
    storage: Arc<S>,
}

impl<S: ShardStorage + Send + Sync> ExpertAssembler<S> {
    pub fn new(storage: Arc<S>) -> Self {
        Self { storage }
    }

    pub async fn assemble_expert(&self, expert_id: &ExpertId) -> AuriaResult<Tensor> {
        let shards = self.storage.get_shards_for_expert(expert_id).await?;
        
        if shards.is_empty() {
            return Err(AuriaError::ExpertNotFound(ExpertId(expert_id.0)));
        }

        self.combine_shards(&shards)
    }

    fn combine_shards(&self, shards: &[Shard]) -> AuriaResult<Tensor> {
        if shards.is_empty() {
            return Err(AuriaError::ExecutionError("No shards to combine".to_string()));
        }

        let total_size: usize = shards.iter().map(|s| s.tensor.data.len()).sum();
        let mut combined = vec![0u8; total_size];
        
        let mut offset = 0;
        for shard in shards {
            let size = shard.tensor.data.len();
            combined[offset..offset + size].copy_from_slice(&shard.tensor.data);
            offset += size;
        }

        let shape = shards.first()
            .map(|s| s.tensor.shape.clone())
            .unwrap_or_else(|| vec![1]);

        Ok(Tensor {
            data: combined,
            shape,
            dtype: TensorDType::FP16,
        })
    }

    pub async fn verify_assembly(&self, expert_id: &ExpertId, expected_hash: &[u8]) -> AuriaResult<bool> {
        let tensor = self.assemble_expert(expert_id).await?;
        
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        tensor.data.hash(&mut hasher);
        let hash = hasher.finish().to_le_bytes();
        
        Ok(&hash[..] == expected_hash)
    }
}

#[async_trait]
pub trait ShardStorage: Send + Sync {
    async fn get_shards_for_expert(&self, expert_id: &ExpertId) -> AuriaResult<Vec<Shard>>;
    async fn get_shard(&self, shard_id: ShardId) -> AuriaResult<Shard>;
    async fn shard_exists(&self, shard_id: ShardId) -> bool;
}

pub struct ExecutionPipeline<B: ExecutionBackend, S: ShardStorage + Send + Sync> {
    engine: ExecutionEngine<B>,
    assembler: Arc<ExpertAssembler<S>>,
    cache: Arc<RwLock<Vec<(ExpertId, Tensor)>>>,
}

impl<B: ExecutionBackend, S: ShardStorage + Send + Sync> ExecutionPipeline<B, S> {
    pub fn new(backend: B, storage: Arc<S>) -> Self {
        let engine = ExecutionEngine::new(backend);
        let assembler = Arc::new(ExpertAssembler::new(storage));
        
        Self {
            engine,
            assembler,
            cache: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn execute_inference(
        &self,
        input: Tensor,
        routing: RoutingDecision,
        mut state: ExecutionState,
    ) -> AuriaResult<ExecutionOutput> {
        let expert_tensors = self.assemble_experts(&routing).await?;
        
        self.execute_forward_pass(input, expert_tensors, &routing, &mut state).await
    }

    async fn assemble_experts(&self, routing: &RoutingDecision) -> AuriaResult<Vec<Tensor>> {
        let mut expert_tensors = Vec::new();
        
        for expert_id in &routing.expert_ids {
            if let Some(cached) = self.get_cached_expert(expert_id).await {
                expert_tensors.push(cached);
            } else {
                let tensor = self.assembler.assemble_expert(expert_id).await?;
                self.cache_expert(*expert_id, tensor.clone()).await;
                expert_tensors.push(tensor);
            }
        }
        
        Ok(expert_tensors)
    }

    async fn execute_forward_pass(
        &self,
        input: Tensor,
        expert_outputs: Vec<Tensor>,
        routing: &RoutingDecision,
        state: &mut ExecutionState,
    ) -> AuriaResult<ExecutionOutput> {
        let input_f32 = tensor_to_f32(&input)?;
        let seq_len = input.shape.get(1).copied().unwrap_or(1) as usize;
        let hidden_size = input_f32.len() / seq_len.max(1);
        
        let mut hidden_states = input_f32;
        
        hidden_states = self.apply_moe(hidden_states, &expert_outputs)?;
        
        let (q, k, v) = self.compute_qkv(&hidden_states, hidden_size)?;
        
        let kv_cache = &state.kv_cache;
        let (q, k) = self.apply_kv_cache(q, k, kv_cache)?;
        
        let attn_config = AttentionConfig {
            num_heads: 8,
            head_dim: hidden_size / 8,
            dropout: 0.0,
            scale: 1.0 / (hidden_size as f32).sqrt(),
        };
        let attn_output = multihead_attention(&q, &k, &v, &attn_config);
        
        hidden_states = self.residual_add(&hidden_states, &attn_output)?;
        
        hidden_states = rms_norm(&hidden_states, hidden_size, 1e-5);
        
        let tokens = self.compute_logits(&hidden_states)?;
        
        Ok(ExecutionOutput {
            tokens,
            usage: auria_core::UsageStats {
                tokens_generated: 1,
            },
        })
    }

    fn apply_moe(&self, hidden: Vec<f32>, experts: &[Tensor]) -> AuriaResult<Vec<f32>> {
        if experts.is_empty() {
            return Ok(hidden);
        }

        let mut output = vec![0.0f32; hidden.len()];
        
        for expert in experts {
            let expert_f32 = tensor_to_f32(expert)?;
            for (i, val) in expert_f32.iter().enumerate() {
                if i < output.len() {
                    output[i] += val;
                }
            }
        }
        
        let scale = 1.0 / (experts.len() as f32).sqrt();
        for val in &mut output {
            *val *= scale;
        }
        
        Ok(output)
    }

    fn compute_qkv(&self, hidden: &[f32], hidden_size: usize) -> AuriaResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let head_dim = hidden_size / 8;
        let q = hidden.to_vec();
        let k = hidden.to_vec();
        let v = hidden.to_vec();
        Ok((q, k, v))
    }

    fn apply_kv_cache(&self, q: Vec<f32>, k: Vec<f32>, _kv_cache: &[Tensor]) -> AuriaResult<(Vec<f32>, Vec<f32>)> {
        Ok((q, k))
    }

    fn residual_add(&self, a: &[f32], b: &[f32]) -> AuriaResult<Vec<f32>> {
        if a.len() != b.len() {
            return Err(AuriaError::ExecutionError("Size mismatch in residual add".to_string()));
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }

    fn compute_logits(&self, hidden: &[f32]) -> AuriaResult<Vec<String>> {
        let vocab_size = 32000;
        
        let top_k = 5;
        let mut logits: Vec<(usize, f32)> = (0..vocab_size)
            .map(|i| {
                let score = hidden.get(i % hidden.len()).copied().unwrap_or(0.0) 
                    + (i as f32 * 0.0001).sin();
                (i, score)
            })
            .collect();
        
        logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        logits.truncate(top_k);
        
        let tokens: Vec<String> = logits
            .iter()
            .map(|(idx, _)| format!("token_{}", idx))
            .collect();
        
        Ok(tokens)
    }

    async fn get_cached_expert(&self, expert_id: &ExpertId) -> Option<Tensor> {
        let cache = self.cache.read().await;
        cache.iter()
            .find(|(id, _)| id == expert_id)
            .map(|(_, tensor)| tensor.clone())
    }

    async fn cache_expert(&self, expert_id: ExpertId, tensor: Tensor) {
        let mut cache = self.cache.write().await;
        if cache.len() >= 10 {
            cache.remove(0);
        }
        cache.push((expert_id, tensor));
    }

    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }

    pub fn backend(&self) -> &B {
        self.engine.backend()
    }
}

pub struct ExecutionConfig {
    pub max_batch_size: u32,
    pub enable_caching: bool,
    pub cache_size: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            enable_caching: true,
            cache_size: 10,
        }
    }
}

pub struct TokenEmbedding {
    embedding_table: Vec<f32>,
    vocab_size: usize,
    hidden_size: usize,
}

impl TokenEmbedding {
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        let embedding_table = (0..vocab_size * hidden_size)
            .map(|i| {
                let idx = i / hidden_size;
                ((idx as f32 * 0.0001).sin() * 0.1).abs()
            })
            .collect();

        Self {
            embedding_table,
            vocab_size,
            hidden_size,
        }
    }

    pub fn embed_token(&self, token_id: usize) -> Vec<f32> {
        if token_id >= self.vocab_size {
            return vec![0.0; self.hidden_size];
        }

        let start = token_id * self.hidden_size;
        self.embedding_table[start..start + self.hidden_size].to_vec()
    }

    pub fn embed_tokens(&self, token_ids: &[usize]) -> Vec<f32> {
        let seq_len = token_ids.len();
        let mut embeddings = Vec::with_capacity(seq_len * self.hidden_size);

        for &token_id in token_ids {
            embeddings.extend(self.embed_token(token_id));
        }

        embeddings
    }

    pub fn unembed(&self, hidden: &[f32]) -> Vec<f32> {
        hidden.to_vec()
    }
}

pub struct KvCache {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    max_seq_len: usize,
    current_len: usize,
}

impl KvCache {
    pub fn new(max_seq_len: usize, hidden_size: usize) -> Self {
        Self {
            keys: Vec::with_capacity(max_seq_len * hidden_size),
            values: Vec::with_capacity(max_seq_len * hidden_size),
            max_seq_len,
            current_len: 0,
        }
    }

    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        if self.current_len >= self.max_seq_len {
            let remove_len = k.len();
            self.keys.drain(0..remove_len);
            self.values.drain(0..remove_len);
            self.current_len = self.max_seq_len - 1;
        }

        self.keys.extend_from_slice(k);
        self.values.extend_from_slice(v);
        self.current_len += 1;
    }

    pub fn len(&self) -> usize {
        self.current_len
    }

    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    pub fn as_tensors(&self) -> (Tensor, Tensor) {
        let shape = vec![1, self.current_len as u32];
        
        let keys_bytes: Vec<u8> = self.keys.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        let values_bytes: Vec<u8> = self.values.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();

        (
            Tensor { data: keys_bytes, shape: shape.clone(), dtype: TensorDType::FP16 },
            Tensor { data: values_bytes, shape, dtype: TensorDType::FP16 },
        )
    }
}

pub struct GatingNetwork {
    pub num_experts: usize,
    pub top_k: usize,
}

impl GatingNetwork {
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self { num_experts, top_k }
    }

    pub fn compute_gates(&self, hidden: &[f32]) -> Vec<(usize, f32)> {
        let mut gates: Vec<(usize, f32)> = (0..self.num_experts)
            .map(|i| {
                let hash = hidden.iter().fold(0u64, |acc, &x| {
                    acc.wrapping_add((x * (i as f32 + 1.0)) as u64)
                });
                let score = ((hash % 1000) as f32) / 1000.0;
                (i, score)
            })
            .collect();

        gates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        gates.truncate(self.top_k);
        
        let sum: f32 = gates.iter().map(|(_, s)| s).sum();
        gates.iter().map(|(i, s)| (*i, s / sum)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_config_default() {
        let config = ExecutionConfig::default();
        assert_eq!(config.max_batch_size, 8);
        assert!(config.enable_caching);
    }
}

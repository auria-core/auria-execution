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
use async_trait::async_trait;
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
        let experts: Vec<Tensor> = Vec::new();
        self.backend.execute_step(input, experts, state).await
    }
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

        self.engine.backend().execute_step(input, expert_tensors, state).await
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
